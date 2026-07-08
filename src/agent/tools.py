"""Agent tools — the functions the LLM is allowed to call (W6 D2).

An agent is just an LLM that can *choose* to call tools. A "tool" here is a
plain Python function wrapped with LangChain's `@tool` decorator, which reads
its **name, argument signature, and docstring** and turns them into the JSON
schema the model sees. That schema is the ENTIRE basis on which the LLM decides
whether to call the tool and what to pass — so the docstring is not a comment,
it is the tool's spec sheet written *for the model*. Vague description => the
model calls it at the wrong time or with the wrong args.

D2 shipped `search_docs` (retrieval). D3 adds two more:
  - `list_sources(topic)` — which doc pages cover a topic (scope / "is this
    answerable?"), a one-shot lookup.
  - `compare(a, b)` — retrieve two concepts so the LLM can synthesize a
    side-by-side; this is the tool that makes the agent genuinely *multi-step*.
`estimate_cost` from docs/agent_design.md stays optional/deferred for now.

Design note on tool selection: with several tools bound, the LLM picks between
them purely from their descriptions, so each tool here is kept **single-purpose
with a distinct docstring** — overlapping semantics would make it choose wrong.
"""

from __future__ import annotations

import functools
import time

from langchain_core.tools import tool
from opensearchpy.exceptions import ConnectionError as OSConnectionError
from opensearchpy.exceptions import ConnectionTimeout

from src.config import AGENT, RETRIEVAL
from src.embeddings import Embedder
from src.generate import format_context
from src.opensearch_client import get_client
from src.retrieve import search

# --- Shared, lazily-built resources -----------------------------------------
# `retrieve.search()` will build an Embedder (loads the bge model) and an
# OpenSearch client on every call if we don't hand it ones. The agent may call
# search_docs several times per question, so we build both ONCE here and reuse
# them across calls — same "build heavy objects once" pattern the eval harness
# uses. Lazy so importing this module (e.g. for its docstring/schema) doesn't
# force the model to load.
_embedder: Embedder | None = None
_os_client = None


def _resources():
    """Return the cached (embedder, opensearch_client), building on first use."""
    global _embedder, _os_client
    if _embedder is None:
        _embedder = Embedder()
    if _os_client is None:
        _os_client = get_client()
    return _embedder, _os_client


# --- D4 guardrail 2: tools must survive transient failures, and NEVER crash the
# graph. Two layers below:
#   (a) _search_with_retry — retry the retrieval call on *transient* errors only,
#       with linear backoff. Retrying a deterministic error (bad query) is
#       pointless, so we only catch the connection/timeout family.
#   (b) @safe_tool — a backstop that converts ANY exception escaping a tool into a
#       readable observation string. Principle: a tool failure becomes something
#       the LLM can READ and route around ("docs unavailable -> tell the user"),
#       not a traceback that kills the whole agent run.

# Transient = worth retrying: OpenSearch down / wrong port / slow (exactly the D4
# fault-injection case). NOT here: RequestError (malformed query, 400) — that's
# deterministic, retrying just wastes time. (429/503 could be added later by
# inspecting TransportError.status_code.)
_TRANSIENT_ERRORS = (OSConnectionError, ConnectionTimeout)


def _search_with_retry(*args, **kwargs) -> list[dict]:
    """Call retrieve.search(), retrying only transient failures with backoff.

    Makes up to `1 + AGENT.tool_max_retries` attempts; sleeps
    `attempt * tool_retry_backoff_s` between them (linear backoff — enough to
    ride out a container restart without hammering). Re-raises the last transient
    error if all attempts fail; `@safe_tool` turns that into an observation.
    """
    attempts = AGENT.tool_max_retries + 1
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return search(*args, **kwargs)
        except _TRANSIENT_ERRORS as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(AGENT.tool_retry_backoff_s * attempt)
    raise last_exc  # exhausted retries; safe_tool will catch and report it


def safe_tool(fn):
    """Wrap a tool so it NEVER raises out of the graph (D4 guardrail 2 backstop).

    Any exception escaping `fn` — retries exhausted, or an unexpected bug — is
    caught and returned as a "TOOL ERROR: ..." observation. The LLM reads that
    like any other tool result and can degrade honestly ("I can't reach the
    docs") instead of the agent crashing. `functools.wraps` keeps fn's name,
    docstring and signature so LangChain's `@tool` still builds the right schema.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _TRANSIENT_ERRORS as exc:
            return (
                f"TOOL ERROR: the vLLM docs search service is unavailable "
                f"({type(exc).__name__}) after retries. Tell the user you "
                f"can't reach the documentation right now; do not fabricate an answer."
            )
        except Exception as exc:  # last-resort backstop — never let it hit the graph
            return (
                f"TOOL ERROR: '{fn.__name__}' failed unexpectedly "
                f"({type(exc).__name__}). Do not retry it; tell the user the "
                f"lookup failed and you can't answer reliably."
            )

    return wrapper


# --- D4 guardrail 3: low-score refusal ---------------------------------------
def _below_threshold(chunks: list[dict]) -> bool:
    """True if retrieval found nothing clearing the relevance bar.

    Pure k-NN always returns top-k, so "no chunks" almost never happens — the
    real "miss" signal is a low BEST score. We gate on the top-1 score (the most
    relevant hit): if even that is below AGENT.min_relevance_score, everything
    retrieved is noise, so the tool should report "not found" and let the LLM
    refuse rather than ground an answer on irrelevant text. Threshold basis: see
    AgentConfig.min_relevance_score / eval/analyze_score_threshold.py.
    """
    if not chunks:
        return True
    return max(c["score"] for c in chunks) < AGENT.min_relevance_score


@tool
@safe_tool
def search_docs(query: str) -> str:
    """Search the vLLM documentation for excerpts relevant to a query.

    Use this to find grounding for ANY factual/technical question about vLLM
    (e.g. how continuous batching works, what PagedAttention is, how to set a
    config flag). It returns the top matching documentation excerpts, each
    numbered and tagged with its source URL, so you can answer from them and
    cite each claim with the excerpt's [n].

    Args:
        query: A natural-language search query describing what to look for in
            the vLLM docs. Prefer the user's own technical terms.

    Returns:
        A numbered, source-tagged context block ("[1] (source: ...)\\n<text>"),
        one entry per excerpt — or a short "no relevant excerpts" note if the
        index has nothing on the query (treat that as: not in the docs).
    """
    embedder, client = _resources()

    # Same retrieval the fixed RAG pipeline uses: bi-encoder k-NN, rerank off
    # (per the W5 D6 ablation). top_k comes from config so the agent and the
    # fixed pipeline stay in lockstep.
    chunks = _search_with_retry(
        query,
        k=RETRIEVAL.top_k,
        client=client,
        embedder=embedder,
        use_reranker=RETRIEVAL.use_reranker,
    )

    if _below_threshold(chunks):
        # No chunk clears the relevance bar -> treat as "not in the docs" and
        # give the model an explicit refuse signal, rather than feeding it
        # top-k noise it might stitch into a confident-but-wrong answer.
        return (
            "No sufficiently relevant excerpts found in the vLLM docs for this "
            "query (best match below the relevance threshold). Treat this as: "
            "the answer is not in the docs — refuse rather than guess."
        )

    # Reuse the exact numbered format the D6 generator feeds the LLM, so a
    # citation [n] in the answer maps to excerpt n here — consistent across the
    # fixed pipeline and the agent.
    return format_context(chunks)


# A numbered formatter with an OFFSET, used by `compare`. format_context() in
# generate.py always numbers from [1]; when we return TWO context groups we need
# one continuous numbering ([1..a] then [a+1..]) so each citation [n] is
# unambiguous. Same "[n] (source: ...)\n<text>" shape as format_context.
def _numbered(chunks: list[dict], start: int) -> list[str]:
    return [
        f"[{i}] (source: {c['source']})\n{c['text']}"
        for i, c in enumerate(chunks, start)
    ]


# How many chunks list_sources scans to decide which pages cover a topic. Wider
# than the answer top_k (5) because here we want page COVERAGE, not the single
# best passage.
_LIST_SOURCES_K = 20


@tool
@safe_tool
def list_sources(topic: str) -> str:
    """List the vLLM documentation pages that discuss a given topic.

    Use this to answer "which docs cover X?" or to judge whether a topic is even
    in the corpus BEFORE trying to answer it. Returns the distinct doc pages
    whose excerpts match the topic, each with how many matching excerpts it has
    (more = more central to that page). This does NOT return excerpt text — call
    `search_docs` when you need the actual content to answer from.

    Args:
        topic: The subject to look up, e.g. "quantization" or "LoRA adapters".

    Returns:
        A bullet list of "<title> (<source>) — <n> matching excerpt(s)", most
        relevant pages first — or a note that no page covers the topic.
    """
    embedder, client = _resources()
    chunks = _search_with_retry(
        topic, k=_LIST_SOURCES_K, client=client, embedder=embedder,
        use_reranker=False,
    )
    if not chunks:
        return f'No vLLM documentation pages found for topic "{topic}".'

    # Collapse the matching chunks to their distinct source pages, counting hits
    # per page as a rough relevance signal.
    by_source: dict[str, dict] = {}
    for c in chunks:
        entry = by_source.setdefault(c["source"], {"title": c["title"], "n": 0})
        entry["n"] += 1

    lines = [f'vLLM documentation pages related to "{topic}":']
    for source, info in sorted(by_source.items(), key=lambda kv: -kv[1]["n"]):
        lines.append(f"- {info['title']} ({source}) — {info['n']} matching excerpt(s)")
    return "\n".join(lines)


@tool
@safe_tool
def compare(concept_a: str, concept_b: str) -> str:
    """Retrieve documentation for TWO vLLM concepts for a side-by-side comparison.

    Use this when the user asks how two things differ or which to use (e.g.
    "compare continuous batching and PagedAttention"). It searches the docs for
    each concept separately and returns both context groups under one continuous
    numbering, so you can synthesize a single cited comparison — cite each claim
    with the [n] of the supporting excerpt.

    Args:
        concept_a: The first vLLM concept to look up.
        concept_b: The second vLLM concept to look up.

    Returns:
        Two labeled, numbered context groups (one per concept) sharing one [n]
        sequence — or a note if neither concept is found in the docs.
    """
    embedder, client = _resources()
    a = _search_with_retry(concept_a, k=RETRIEVAL.top_k, client=client,
                           embedder=embedder, use_reranker=False)
    b = _search_with_retry(concept_b, k=RETRIEVAL.top_k, client=client,
                           embedder=embedder, use_reranker=False)

    # Apply the same relevance bar per concept; a concept whose best hit is below
    # threshold is treated as "not in the docs". If neither concept clears it,
    # there's nothing worth comparing -> refuse.
    if _below_threshold(a) and _below_threshold(b):
        return (
            "No sufficiently relevant excerpts found in the vLLM docs for either "
            "concept (both below the relevance threshold). Treat this as: not in "
            "the docs — refuse rather than guess."
        )

    # Continuous numbering: A gets [1..len(a)], B continues from len(a)+1.
    parts = [
        f'## Concept A: "{concept_a}"\n' + "\n\n".join(_numbered(a, 1)),
        f'## Concept B: "{concept_b}"\n' + "\n\n".join(_numbered(b, len(a) + 1)),
    ]
    return "\n\n".join(parts)


if __name__ == "__main__":
    #   uv run python -m src.agent.tools
    # Smoke-test each tool in isolation (no agent yet). `.invoke({...})` is how
    # you call a LangChain @tool by hand.
    print("=== search_docs")
    print(search_docs.invoke({"query": "How does vLLM do continuous batching?"})[:400])
    print("\n=== list_sources")
    print(list_sources.invoke({"topic": "quantization"}))
    print("\n=== compare")
    print(compare.invoke(
        {"concept_a": "continuous batching", "concept_b": "PagedAttention"}
    )[:500])
