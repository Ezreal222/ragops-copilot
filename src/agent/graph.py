"""Single-step ReAct agent as a LangGraph graph (W6 D2).

This is the smallest "real" agent: an LLM that, per turn, either **calls a tool**
or **gives a final answer**. Unlike the fixed pipeline in src/generate.py
(embed -> retrieve -> generate, always), here the *decision to retrieve* is the
model's. For a normal lookup it should call `search_docs` once, read the
excerpts, then answer with citations — reproducing today's RAG, but chosen at
runtime instead of hard-wired.

The graph (see docs/agent_design.md §2):

    (START) -> agent --tool_calls?-- yes --> tools --+
                 ^                                    |
                 └───────── ToolMessage(s) ───────────┘
                 │
                 └── no tool_calls ──> (END)  final cited answer

Four LangGraph concepts, reused unchanged in D3 (multi-step):
  - State  : `MessagesState` — a `messages` list with the `add_messages` reducer,
             which APPENDS each new message (LLM requests + tool results) instead
             of overwriting, so the agent always sees the full trace.
  - Node `agent` : one LLM call with the tools bound; emits tool_calls or answer.
  - Node `tools` : a prebuilt `ToolNode` that runs the requested tool(s) and
             appends each result as a `ToolMessage`.
  - Conditional edge after `agent`: last message has tool_calls -> `tools`, else
             -> `END`. This edge IS the ReAct "keep going or stop" switch.
  - Edge `tools -> agent`: loop back so the LLM observes the tool output.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import compare, list_sources, search_docs
from src.config import LLM, LLMConfig
from src.generate import SYSTEM_PROMPT

# Load .env so the provider key is available. Safe to call repeatedly.
load_dotenv()

# The tools this agent may call. D3 = retrieval + list_sources + compare (the
# last drives genuine multi-step). Kept as one list so both the LLM binding and
# the ToolNode use the exact same set (a mismatch = the model calls a tool the
# node can't run).
TOOLS = [search_docs, list_sources, compare]

# Runaway-loop cap — the safety valve D2's 5x-search run motivated. A ReAct LLM
# can get stuck calling tools without converging, burning tokens/latency; this
# bounds how many agent<->tools super-steps one invoke may take, passed as
# LangGraph's `recursion_limit` (its default is 25). A proper AgentConfig plus a
# graceful "forced answer on cap" lands in D4; here hitting it just stops.
RECURSION_LIMIT = 8


def get_chat_model(config: LLMConfig = LLM) -> ChatOpenAI:
    """Build a LangChain chat model for the configured provider.

    We reuse the SAME OpenAI-compatible `LLMConfig` the fixed pipeline uses
    (DeepSeek by default) — only the wrapper differs: generate.py talks to the
    raw OpenAI SDK, while an agent needs a LangChain chat model so `bind_tools`
    can attach the tool schemas. `ChatOpenAI` speaks the OpenAI protocol, so
    pointing `base_url` at DeepSeek is all it takes.
    """
    if config.provider == "anthropic":
        raise NotImplementedError(
            "anthropic provider not wired for the agent yet — use ChatAnthropic, "
            "or set LLMConfig.provider to 'deepseek'/'openai'."
        )

    api_key = os.environ.get(config.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{config.api_key_env} is not set. Copy .env.example to .env and "
            f"fill in your {config.provider} key."
        )
    return ChatOpenAI(
        model=config.model,
        base_url=config.base_url or None,
        api_key=api_key,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )


def build_graph(config: LLMConfig = LLM):
    """Compile and return the single-step ReAct graph.

    Built lazily (as a function, not at import) so importing this module doesn't
    require the API key or construct the model until someone actually runs it.
    """
    llm_with_tools = get_chat_model(config).bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict:
        """Reason step: run the LLM over the conversation so far.

        We prepend the D6 system prompt fresh each turn (kept OUT of the
        accumulated state) so every LLM call — the first decision and the final
        answer after tool results — obeys the same grounded/cited/refuse rules.
        Returns the new AIMessage; the `add_messages` reducer appends it.
        """
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: MessagesState) -> str:
        """The ReAct switch: did the LLM ask to act, or is it done?

        If its last message carries tool_calls -> go run them (`tools`);
        otherwise it produced a final answer -> stop (`END`).
        """
        last = state["messages"][-1]
        return "tools" if last.tool_calls else END

    g = StateGraph(MessagesState)
    g.add_node("agent", agent_node)
    g.add_node("tools", ToolNode(TOOLS))  # executes the requested tool call(s)

    g.add_edge(START, "agent")
    # After the LLM: branch to tools (act) or END (answer). The dict maps the
    # string should_continue returns to the next node.
    g.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    # After tools: always loop back so the LLM observes the results and decides
    # the next step (in D2 that's just "write the final answer").
    g.add_edge("tools", "agent")

    return g.compile()


def print_trace(messages: list) -> None:
    """Pretty-print the ReAct message trace for manual inspection (Step 3).

    The whole point of the agent is that the control flow is visible: a healthy
    lookup reads HumanMessage -> AIMessage(tool_calls) -> ToolMessage ->
    AIMessage(final). This makes that sequence legible.
    """
    for m in messages:
        role = m.__class__.__name__
        if role == "AIMessage" and getattr(m, "tool_calls", None):
            calls = ", ".join(f"{tc['name']}({tc['args']})" for tc in m.tool_calls)
            print(f"  AIMessage    -> wants tool: {calls}")
        elif role == "ToolMessage":
            preview = " ".join((m.content or "").split())[:100]
            print(f"  ToolMessage  <- {m.name}: {preview}...")
        else:  # HumanMessage / plain AIMessage (final answer)
            preview = " ".join((m.content or "").split())[:160]
            print(f"  {role:<12} : {preview}")


if __name__ == "__main__":
    #   uv run python -m src.agent.graph
    # Live MULTI-STEP test (makes real LLM calls). Expect the agent to PLAN
    # implicitly — pick a tool, read the result, maybe call another — rather than
    # answer in one shot. recursion_limit is passed so a non-converging loop
    # can't run away.
    app = build_graph()
    questions = [
        "Compare continuous batching and PagedAttention in vLLM.",  # -> compare / 2x search + synthesis
        "What docs cover quantization?",                            # -> list_sources
    ]
    for q in questions:
        print(f"\n########## {q}")
        try:
            result = app.invoke(
                {"messages": [("user", q)]},
                {"recursion_limit": RECURSION_LIMIT},
            )
        except GraphRecursionError:
            print(f"  [hit recursion_limit={RECURSION_LIMIT}] safety valve stopped a runaway loop")
            continue

        # How many turns actually requested tools = how "multi-step" this was.
        tool_rounds = sum(
            1 for m in result["messages"] if getattr(m, "tool_calls", None)
        )
        print_trace(result["messages"])
        print(f"\n  steps (tool-call rounds): {tool_rounds}")
        print("  --- FINAL ANSWER ---")
        print(" ", result["messages"][-1].content)
