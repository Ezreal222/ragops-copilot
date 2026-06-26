"""query the index: embed a question, k-NN search, return chunks.

This is the online half of RAG. Given a question we:
  1. embed it with the SAME model used for the corpus (so they share a space),
  2. ask OpenSearch for the `k` nearest chunk vectors (HNSW approximate search),
  3. return each hit's score + stored fields (text/title/source) for display,
     downstream answer-generation, and citation.
"""

from __future__ import annotations

from src.config import INDEX, RETRIEVAL
from src.embeddings import Embedder
from src.opensearch_client import get_client


def search(
    query: str,
    k: int = RETRIEVAL.top_k,
    *,
    client=None,
    embedder: Embedder | None = None,
) -> list[dict]:
    """Return the top-`k` chunks nearest to `query`.

    `client`/`embedder` are injectable so callers can
    build them once and reuse across many queries instead of reloading the model
    per call.

    Each result: {score, chunk_id, title, source, section, text}.
    """
    client = client or get_client()
    embedder = embedder or Embedder()

    qv = embedder.encode_query(query)
    body = {
        "size": k,
        "query": {"knn": {"embedding": {"vector": qv, "k": k}}},
        # Don't ship the 384-float embedding back over the wire — we only need
        # the human-readable fields for each hit.
        "_source": {"excludes": ["embedding"]},
    }
    res = client.search(index=INDEX.index_name, body=body)

    hits = []
    for h in res["hits"]["hits"]:
        src = h["_source"]
        hits.append(
            {
                "score": h["_score"],
                "chunk_id": src["chunk_id"],
                "title": src["title"],
                "source": src["source"],
                "section": src["section"],
                "text": src["text"],
            }
        )
    return hits


if __name__ == "__main__":
    #   uv run python -m src.retrieve
    # Build the model + client once, reuse for a couple of smoke queries.
    embedder = Embedder()
    client = get_client()
    print(f"index holds {client.count(index=INDEX.index_name)['count']} docs\n")

    for q in [
        "How does vLLM use continuous batching?",
        "What is PagedAttention and what problem does it solve?",
    ]:
        print(f"=== {q}")
        for i, hit in enumerate(search(q, client=client, embedder=embedder), 1):
            snippet = hit["text"][:120].replace("\n", " ")
            print(f"  {i}. {hit['score']:.3f}  {hit['title']}  [{hit['source']}]")
            print(f"      {snippet}")
        print()
