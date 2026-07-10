"""Unified entrypoint — one `answer()` that routes to the agent OR fixed RAG (W6 D6).

Until today the system had two separate front doors:
  - `generate.ask()`      — the fixed retrieve->rerank->generate pipeline (W4-W5),
  - `agent.graph.run_agent()` — the guarded LangGraph agent (W6 D2-D4).

D6 makes the **agent the system's single entrypoint**: `answer()` sends every
question through `run_agent()` by default, so a simple lookup and a multi-step
compare share one path. A `use_agent` config flag (SERVING.use_agent) flips the
route back to the fixed pipeline — the production **degrade switch** for when the
agent is too slow / costly / loops (progressive rollout 101).

Whichever path runs, `answer()` returns the SAME shape so the frontend, eval
harness, and monitoring never special-case which engine produced the result:

    {
      "answer":     str,                        # the grounded, cited text
      "citations":  [{n, source, title}, ...],  # sources actually cited by [n]
      "tool_trace": [{name, args}, ...],        # tools called, in order
      "steps":      int,                        # tool-call rounds (fixed RAG = 1)
      "contexts":   [str, ...],                 # excerpts used as grounding (for eval/RAGAS)
      "mode":       "agent" | "rag",            # which engine answered
      "degraded":   bool,                       # agent hit its step-cap fallback
    }

Why this module (not generate.py)? `agent/graph.py` already imports from
`generate.py`; putting the router there too would create an import cycle. A thin
top module that imports BOTH sides is the clean seam.
"""

from __future__ import annotations

import re

from src.agent.graph import build_graph, run_agent
from src.config import AGENT, LLM, SERVING, AgentConfig, LLMConfig, ServingConfig
from src.generate import ask

# Parses one "[n] (source: X)\n<text>" excerpt block out of a tool observation —
# the same numbered format search_docs/compare emit (see generate.format_context
# / tools._numbered). Group 3 (the text) runs until the next "[n] (source:" header
# or end-of-string, so multi-line excerpts stay intact. DOTALL so "." spans lines.
_EXCERPT_RE = re.compile(
    r"\[(\d+)\]\s*\(source:\s*([^)]+)\)\s*\n(.*?)(?=\n\s*\[\d+\]\s*\(source:|\Z)",
    re.DOTALL,
)


def _agent_tool_trace(messages: list) -> list[dict]:
    """Flatten the ReAct message trace into an ordered list of tool calls.

    Walks the AIMessages that carried tool_calls and records each as
    {name, args} in the order the agent requested them — a compact, UI/monitoring
    friendly view of "what did the agent actually do" (vs the full message list).
    """
    trace = []
    for m in messages:
        for tc in getattr(m, "tool_calls", None) or []:
            trace.append({"name": tc["name"], "args": tc["args"]})
    return trace


def _agent_contexts(messages: list) -> list[str]:
    """Recover the excerpt TEXTS the agent was shown, from its ToolMessages.

    The tools return numbered "[n] (source: X)\\n<text>" blocks; we parse those
    back into the raw excerpt texts so `answer()` can expose `contexts` uniformly
    with the fixed pipeline's retrieved chunks — this is what RAGAS's
    context_precision / context_recall consume. Order-preserving de-dupe because a
    multi-search question (e.g. compare) can surface the same excerpt twice.
    """
    contexts: list[str] = []
    seen: set[str] = set()
    for m in messages:
        if m.__class__.__name__ != "ToolMessage":
            continue
        for match in _EXCERPT_RE.findall(m.content or ""):
            t = match[2].strip()  # group 3 = the excerpt text
            if t and t not in seen:
                seen.add(t)
                contexts.append(t)
    return contexts


def answer(
    question: str,
    *,
    use_agent: bool | None = None,
    serving: ServingConfig = SERVING,
    config: LLMConfig = LLM,
    agent_config: AgentConfig = AGENT,
    app=None,
    os_client=None,
    embedder=None,
    reranker=None,
    llm_client=None,
) -> dict:
    """Answer `question` via the agent (default) or fixed RAG, in a unified shape.

    Routing: `use_agent` (explicit arg) wins if given, else `serving.use_agent`.
    This is the one function the frontend / eval / monitoring should call.

    Heavy objects are injectable so a caller answering many questions (the eval
    harness) builds them once. They apply to whichever path is chosen:
      - agent path: pass a pre-compiled `app` (build_graph()) to reuse the graph;
        the tools hold their OWN cached embedder/OpenSearch client, so the
        os_client/embedder/reranker args below are ignored here by design.
      - fixed-RAG path: os_client/embedder/reranker/llm_client feed `ask()`.

    Returns the unified dict documented at the top of this module.
    """
    route_agent = serving.use_agent if use_agent is None else use_agent

    if route_agent:
        # --- Agent path: the LLM decides which tool(s) to call. run_agent already
        # enforces the D4 guardrails (step cap -> graceful fallback, citation
        # check) and returns answer/steps/degraded/messages/citations. We just
        # reshape it into the unified contract.
        r = run_agent(question, app=app, config=config, agent_config=agent_config)
        return {
            "answer": r["answer"],
            "citations": r["citations"],
            "tool_trace": _agent_tool_trace(r["messages"]),
            "steps": r["steps"],
            "contexts": _agent_contexts(r["messages"]),
            "mode": "agent",
            "degraded": r["degraded"],
        }

    # --- Fixed-RAG path: deterministic retrieve->rerank->generate. It always does
    # exactly one retrieval, so we represent that as a single synthetic
    # search_docs entry (steps=1). This is honest AND the D6 point in miniature: a
    # simple question is one search_docs call whether the agent or the fixed
    # pipeline runs it — the two should be near-equivalent.
    r = ask(
        question,
        os_client=os_client,
        embedder=embedder,
        reranker=reranker,
        llm_client=llm_client,
        config=config,
    )
    return {
        "answer": r["answer"],
        "citations": r["citations"],
        "tool_trace": [{"name": "search_docs", "args": {"query": question}}],
        "steps": 1,
        "contexts": [c["text"] for c in r["retrieved_chunks"]],
        "mode": "rag",
        "degraded": False,
    }


if __name__ == "__main__":
    #   uv run python -m src.answer
    # Smoke test: run the SAME simple lookup through both engines and confirm the
    # unified shape holds. On a simple question the agent should make ONE
    # search_docs call (steps=1), i.e. behave ~like the fixed pipeline — the D6
    # "no unnecessary complexity" expectation.
    q = "What is PagedAttention and what problem does it solve?"

    print(f"### agent (SERVING.use_agent default = {SERVING.use_agent})")
    app = build_graph()  # compile once, reuse
    a = answer(q, use_agent=True, app=app)
    print(f"  mode={a['mode']}  steps={a['steps']}  degraded={a['degraded']}")
    print(f"  tool_trace={a['tool_trace']}")
    print(f"  citations={[c['n'] for c in a['citations']]}  contexts={len(a['contexts'])}")
    print("  answer:", " ".join(a["answer"].split())[:160])

    print("\n### fixed RAG (use_agent=False)")
    r = answer(q, use_agent=False)
    print(f"  mode={r['mode']}  steps={r['steps']}  degraded={r['degraded']}")
    print(f"  tool_trace={r['tool_trace']}")
    print(f"  citations={[c['n'] for c in r['citations']]}  contexts={len(r['contexts'])}")
    print("  answer:", " ".join(r["answer"].split())[:160])
