"""D4 completion check: inject all four fault classes, confirm graceful degrade.

The D4 bar is "deliberately break the agent four ways; each time it should
degrade gracefully — no unhandled exception, a sensible output". This harness is
that check, made reproducible. Each block wraps its call in try/except and only
PASSes if (a) nothing raised out and (b) the expected fallback fired.

  1. step limit      — recursion_limit=1 forces a runaway; expect FALLBACK_ANSWER
  2. tool failure    — OpenSearch pointed at a dead port; expect a TOOL ERROR obs
  3. low-score       — off-topic question; expect a below-threshold refusal
  4. citation check  — fabricated [9] not in context; expect it flagged

Only fault 1 makes a (single) real LLM call; 2/3/4 are checked at the tool /
validator level, so this is cheap to re-run.

    uv run python -m eval.verify_guardrails
"""

from __future__ import annotations

from langchain_core.messages import ToolMessage
from opensearchpy import OpenSearch

import src.agent.tools as T
from src.agent.graph import build_graph, run_agent, validate_citations
from src.config import AgentConfig
from src.embeddings import Embedder

REFUSAL_MARK = "below the relevance threshold"


def check(name: str, fn) -> bool:
    """Run one fault block; PASS iff it returns True AND raised nothing."""
    try:
        ok = fn()
    except Exception as exc:  # an escaped exception IS the failure we test against
        print(f"[FAIL] {name}: unhandled {type(exc).__name__}: {exc}")
        return False
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def fault_step_limit(app) -> bool:
    # recursion_limit=1 -> any tool-calling question overruns -> graceful fallback
    r = run_agent("Compare continuous batching and PagedAttention.",
                  app=app, agent_config=AgentConfig(recursion_limit=1))
    print(f"       degraded={r['degraded']} steps={r['steps']} answer={r['answer'][:60]!r}")
    return r["degraded"] is True


def fault_tool_failure() -> bool:
    # Swap the cached OpenSearch client for one on a dead port; the embedder still
    # loads, retrieval then hits ConnectionError -> retries -> TOOL ERROR string.
    T._embedder = T._embedder or Embedder()
    T._os_client = OpenSearch(hosts=["http://localhost:9999"])
    try:
        out = T.search_docs.invoke({"query": "How does vLLM do continuous batching?"})
    finally:
        T._os_client = None  # reset so later checks rebuild against the real host
    print(f"       tool returned: {out[:70]!r}")
    return isinstance(out, str) and out.startswith("TOOL ERROR")


def fault_low_score() -> bool:
    # Clearly off-topic -> best chunk below AGENT.min_relevance_score -> refuse.
    T._embedder = T._embedder or Embedder()
    out = T.search_docs.invoke({"query": "How do I bake sourdough bread at home?"})
    print(f"       tool returned: {out[:70]!r}")
    return REFUSAL_MARK in out


def fault_hallucinated_citation() -> bool:
    # Answer cites [9] but only [1],[2] were shown -> [9] flagged, [1] resolves.
    msgs = [ToolMessage(
        content="[1] (source: docs/a.md)\nx\n\n[2] (source: docs/b.md)\ny",
        tool_call_id="x", name="search_docs",
    )]
    rep = validate_citations("vLLM does X [1] and also Y [9].", msgs)
    print(f"       citations={[c['n'] for c in rep['citations']]} "
          f"hallucinated={rep['hallucinated']} warnings={rep['warnings']}")
    return rep["hallucinated"] == [9] and [c["n"] for c in rep["citations"]] == [1]


if __name__ == "__main__":
    app = build_graph()
    results = [
        check("1 step-limit -> graceful fallback", lambda: fault_step_limit(app)),
        check("2 tool failure -> TOOL ERROR observation", fault_tool_failure),
        check("3 low-score -> refusal", fault_low_score),
        check("4 hallucinated citation -> flagged", fault_hallucinated_citation),
    ]
    print(f"\n{sum(results)}/{len(results)} guardrails degraded gracefully")
    raise SystemExit(0 if all(results) else 1)
