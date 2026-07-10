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
import re

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from src.agent.tools import compare, list_sources, search_docs
from src.config import AGENT, LLM, AgentConfig, LLMConfig
from src.generate import SYSTEM_PROMPT, _CITATION_RE, _cited_sources

# Load .env so the provider key is available. Safe to call repeatedly.
load_dotenv()

# The tools this agent may call. D3 = retrieval + list_sources + compare (the
# last drives genuine multi-step). Kept as one list so both the LLM binding and
# the ToolNode use the exact same set (a mismatch = the model calls a tool the
# node can't run).
TOOLS = [search_docs, list_sources, compare]

# D4 guardrail 1 — graceful degradation when the step cap is hit. The runaway
# cap itself now lives in AgentConfig.recursion_limit (D3 introduced it as a bare
# constant that just let GraphRecursionError propagate). Here we turn that crash
# into an honest fallback answer: a non-converging loop shouldn't dump a
# traceback on the user, it should say "I couldn't do this reliably" — same
# spirit as refusing rather than hallucinating.
FALLBACK_ANSWER = (
    "I tried several steps but couldn't work this out reliably. Please try "
    "rephrasing the question or narrowing it to a more specific vLLM topic."
)

# D6 convergence fix — the agent's system prompt = the shared grounding/citation
# rules (SYSTEM_PROMPT) PLUS a tool-use POLICY. Why a separate prompt: the D5
# agent eval and the D6 regression both showed the agent *non-converging* on
# simple questions — it fired search_docs repeatedly with reworded queries (often
# two in parallel per turn), burned the recursion_limit, and then refused an
# answerable question. The fixed RAG pipeline answers the same questions fine
# because it retrieves ONCE and commits. This policy makes the agent behave that
# way by default: retrieve once, then ANSWER. We keep it OUT of generate.py's
# SYSTEM_PROMPT so the fixed-RAG regression baseline is unchanged (a fair compare).
AGENT_SYSTEM_PROMPT = SYSTEM_PROMPT + (
    "\n\nTool-use policy (follow exactly):\n"
    "- For a normal question, call `search_docs` EXACTLY ONCE, then answer from "
    "the returned excerpts, citing each claim with [n]. One retrieval is enough.\n"
    "- Do NOT re-search with reworded queries to gather more — commit to an answer "
    "from what the first search returned. Prefer answering over searching again.\n"
    "- Issue tool calls ONE AT A TIME (never several search calls in parallel).\n"
    "- Only make a second tool call if the first returned NO relevant excerpts; "
    "then either try one alternative query, or refuse with the exact 'not found' "
    "sentence. Never exceed a couple of tool calls.\n"
    "- Use `compare` (once) for two-concept comparisons and `list_sources` (once) "
    'for "which docs cover X?" — same one-shot rule.'
)


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

        We prepend the AGENT system prompt fresh each turn (kept OUT of the
        accumulated state) so every LLM call — the first decision and the final
        answer after tool results — obeys the same grounded/cited/refuse rules
        PLUS the D6 tool-use policy (retrieve once, then answer) that keeps the
        agent from non-converging on simple questions.
        Returns the new AIMessage; the `add_messages` reducer appends it.
        """
        messages = [SystemMessage(content=AGENT_SYSTEM_PROMPT)] + state["messages"]
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


# Parses "[n] (source: X)" blocks out of a tool observation — the numbered
# context the tools handed the LLM. Group 1 = the citation number, group 2 = the
# source path, matching the "[n] (source: ...)" shape format_context / tools emit.
_SHOWN_RE = re.compile(r"\[(\d+)\]\s*\(source:\s*([^)]+)\)")


def _shown_context(messages: list) -> list[dict]:
    """Reconstruct the numbered context the LLM was shown, from the ToolMessages.

    Tools return "[n] (source: X)" blocks; we parse those back into a
    chunks-shaped list (index n-1 -> {source, title}) so we can feed it to
    generate._cited_sources and reuse the SAME citation-resolution logic the
    fixed pipeline uses. Limitation: if the agent made several retrieval calls
    that each number from [1], a later block overwrites an earlier one at the
    same n — fine for the common single-search question, approximate for
    multi-retrieval. Good enough to catch out-of-range (hallucinated) citations.
    """
    by_n: dict[int, str] = {}
    for m in messages:
        if m.__class__.__name__ != "ToolMessage":
            continue
        for num, source in _SHOWN_RE.findall(m.content or ""):
            by_n[int(num)] = source.strip()
    if not by_n:
        return []
    # Dense list up to the max shown number; gaps (unseen n) get a sentinel
    # source so a citation to them still counts as out-of-range/hallucinated.
    hi = max(by_n)
    return [{"source": by_n.get(i, "<unknown>"), "title": by_n.get(i, "<unknown>")}
            for i in range(1, hi + 1)]


def _is_refusal(answer: str) -> bool:
    """True if the answer is an honest 'not found' / fallback, not a real answer.

    A refusal legitimately has no citations, so we exclude it from the
    "answered but cited nothing" warning below.
    """
    a = answer.lower()
    return ("couldn't find this in the vllm docs" in a
            or answer.strip() == FALLBACK_ANSWER)


def validate_citations(answer: str, messages: list) -> dict:
    """Output guardrail (D4 guardrail 4): check the answer's [n] are real.

    Every [n] in a grounded answer should point at an excerpt the LLM was
    actually shown. Reusing generate._cited_sources, we split the answer's
    citations into:
      - `citations`   : [n] that resolve to a shown source (traceable, good),
      - `hallucinated`: [n] with no shown excerpt (e.g. [9] when 5 were shown) —
        a fabricated reference, the thing this guardrail exists to catch.
    We also warn when a NON-refusal answer cites nothing at all (possible bare /
    ungrounded answer). We MARK (report), not rewrite — the caller decides.
    """
    shown = _shown_context(messages)
    citations = _cited_sources(answer, shown)          # in-range -> real sources
    valid_ns = {c["n"] for c in citations}
    all_ns = {int(m) for m in _CITATION_RE.findall(answer)}
    hallucinated = sorted(all_ns - valid_ns)           # cited but never shown

    warnings = []
    if hallucinated:
        warnings.append(f"hallucinated citation(s) {hallucinated}: no such excerpt was retrieved")
    if not all_ns and not _is_refusal(answer):
        warnings.append("non-refusal answer contains no citations (possible ungrounded answer)")

    return {
        "citations": citations,
        "hallucinated": hallucinated,
        "warnings": warnings,
        "ok": not warnings,
    }


def run_agent(
    question: str,
    *,
    app=None,
    config: LLMConfig = LLM,
    agent_config: AgentConfig = AGENT,
) -> dict:
    """Answer `question` with the agent, guarded so a runaway loop can't crash.

    This is the guardrailed entry point (D4). Callers should use it instead of
    `app.invoke(...)` directly, because it enforces the step cap and turns the
    two "bad" outcomes into structured, degraded-but-safe results rather than
    exceptions:

      - the LLM keeps calling tools past `recursion_limit` -> LangGraph raises
        GraphRecursionError; we catch it and return `FALLBACK_ANSWER` with
        `degraded=True`, so the caller/user gets an honest message, not a stack
        trace.
      - normal case -> the final cited answer plus `steps` (how many agent turns
        actually requested tools). That count is a cheap agent-health signal: a
        healthy lookup is 1 step; consistently high counts mean the model is
        thrashing and the cap (or the tools/prompt) needs a look.

    `app` is injectable so a caller answering many questions builds the graph
    once; we compile lazily if none is passed.

    Returns: {answer: str, steps: int, degraded: bool, messages: list}.
    (`messages` is the full ReAct trace, or [] when we bailed on the cap.)
    """
    app = app or build_graph(config)
    try:
        result = app.invoke(
            {"messages": [("user", question)]},
            {"recursion_limit": agent_config.recursion_limit},
        )
    except GraphRecursionError:
        # Hit the runaway cap: degrade gracefully instead of propagating.
        return {
            "answer": FALLBACK_ANSWER,
            "steps": agent_config.recursion_limit,
            "degraded": True,
            "messages": [],
            "citations": [],
            "citation_report": validate_citations(FALLBACK_ANSWER, []),
        }

    # steps = how many turns asked to act. Same count print_trace surfaces; a
    # useful health metric to log/monitor later (W7).
    steps = sum(1 for m in result["messages"] if getattr(m, "tool_calls", None))
    answer = result["messages"][-1].content
    # Output guardrail: verify every [n] in the answer maps to a shown excerpt.
    citation_report = validate_citations(answer, result["messages"])
    return {
        "answer": answer,
        "steps": steps,
        "degraded": False,
        "messages": result["messages"],
        "citations": citation_report["citations"],
        "citation_report": citation_report,
    }


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
    # Live MULTI-STEP test (makes real LLM calls) via the guarded run_agent()
    # entry point. Expect the agent to PLAN implicitly — pick a tool, read the
    # result, maybe call another — rather than answer in one shot.
    app = build_graph()  # compile once, reuse across questions
    questions = [
        "Compare continuous batching and PagedAttention in vLLM.",  # -> compare / 2x search + synthesis
        "What docs cover quantization?",                            # -> list_sources
    ]
    for q in questions:
        print(f"\n########## {q}")
        result = run_agent(q, app=app)
        print_trace(result["messages"])
        print(f"\n  steps (tool-call rounds): {result['steps']}  degraded: {result['degraded']}")
        rep = result["citation_report"]
        print(f"  citations: {[c['n'] for c in rep['citations']]}  "
              f"citation_ok: {rep['ok']}  warnings: {rep['warnings']}")
        print("  --- FINAL ANSWER ---")
        print(" ", result["answer"])

    # D4 guardrail-1 fault injection: force the step cap absurdly low so any
    # tool-calling question overruns it, and confirm we get the graceful
    # FALLBACK_ANSWER (degraded=True) — NOT a GraphRecursionError traceback.
    print("\n########## [fault injection] recursion_limit=1 -> expect graceful fallback")
    tiny = AgentConfig(recursion_limit=1)
    result = run_agent("Compare continuous batching and PagedAttention.", app=app, agent_config=tiny)
    print(f"  degraded: {result['degraded']}  steps: {result['steps']}")
    print(f"  answer: {result['answer']}")

    # D4 guardrail-4 fault injection: hand validate_citations a fabricated answer
    # citing [9] when only 2 excerpts were shown, and confirm [9] is flagged as
    # hallucinated while [1] resolves to a real source. No LLM call needed.
    print("\n########## [fault injection] hallucinated citation -> expect [9] flagged")
    from langchain_core.messages import ToolMessage
    fake_msgs = [ToolMessage(
        content="[1] (source: docs/a.md)\nreal excerpt\n\n[2] (source: docs/b.md)\nreal excerpt",
        tool_call_id="x", name="search_docs",
    )]
    rep = validate_citations("vLLM does X [1] and also Y [9].", fake_msgs)
    print(f"  citations: {[c['n'] for c in rep['citations']]}  "
          f"hallucinated: {rep['hallucinated']}  ok: {rep['ok']}")
