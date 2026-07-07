"""Agent tools — the functions the LLM is allowed to call (W6 D2).

An agent is just an LLM that can *choose* to call tools. A "tool" here is a
plain Python function wrapped with LangChain's `@tool` decorator, which reads
its **name, argument signature, and docstring** and turns them into the JSON
schema the model sees. That schema is the ENTIRE basis on which the LLM decides
whether to call the tool and what to pass — so the docstring is not a comment,
it is the tool's spec sheet written *for the model*. Vague description => the
model calls it at the wrong time or with the wrong args.

D2 exposes exactly one tool: `search_docs` (retrieval). The other three tools
in docs/agent_design.md (`list_sources`, `compare`, `estimate_cost`) land in D3.
"""

from __future__ import annotations

from langchain_core.tools import tool

from src.config import RETRIEVAL
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


@tool
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
    chunks = search(
        query,
        k=RETRIEVAL.top_k,
        client=client,
        embedder=embedder,
        use_reranker=RETRIEVAL.use_reranker,
    )

    if not chunks:
        # Give the model an explicit signal rather than an empty string, so it
        # can honestly refuse instead of guessing. (Hard fallback wiring is D4.)
        return "No relevant excerpts found in the vLLM docs for this query."

    # Reuse the exact numbered format the D6 generator feeds the LLM, so a
    # citation [n] in the answer maps to excerpt n here — consistent across the
    # fixed pipeline and the agent.
    return format_context(chunks)


if __name__ == "__main__":
    #   uv run python -m src.agent.tools
    # Smoke-test the tool in isolation (no agent yet): confirm it retrieves for
    # a real question and returns the "nothing found" signal for gibberish.
    # `.invoke({...})` is how you call a LangChain @tool by hand.
    for q in [
        "How does vLLM do continuous batching?",
        "asdfghjkl zxcvbnm not a real vllm topic",
    ]:
        print(f"=== {q}")
        print(search_docs.invoke({"query": q})[:600])
        print()
