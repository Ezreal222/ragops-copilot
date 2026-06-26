"""OpenSearch client + k-NN index definition.

This is the "vector database" half of the corpus. We define an index with:
  - one `knn_vector` field holding the 384-dim chunk embedding, searched with
    HNSW (approximate nearest neighbor) under cosine similarity, and
  - the original chunk fields (text/source/title/section/chunk_id) so a hit can
    be shown to the user and cited.

Why ANN/HNSW instead of brute force? Brute force scores the query against every
vector — O(N·d) per query, which doesn't scale. HNSW searches a navigable graph
in sub-linear time, trading a tiny bit of recall for a large speedup. That's the
whole reason a vector index exists.

Host comes from `OPENSEARCH_HOST` env var if set (e.g. AWS later), else the
local Docker default in `IndexConfig`.
"""

from __future__ import annotations

import os

from opensearchpy import OpenSearch

from src.config import INDEX, IndexConfig


def get_client(config: IndexConfig = INDEX) -> OpenSearch:
    """Connect to OpenSearch. Local dev runs with the security plugin disabled,
    so no auth is needed; for AWS later, point OPENSEARCH_HOST at the endpoint.
    """
    host = os.getenv("OPENSEARCH_HOST", config.host)
    return OpenSearch(hosts=[host])


def index_body(config: IndexConfig = INDEX, dimension: int = 384) -> dict:
    """The create-index request body: k-NN settings + field mappings.

    `index.knn: true` turns on the k-NN plugin for this index. The embedding
    field declares its dimension (must match the model), the similarity metric
    (`space_type`), and the ANN method (HNSW on the chosen engine).
    """
    return {
        "settings": {
            "index": {
                "knn": True,
            }
        },
        "mappings": {
            "properties": {
                "embedding": {
                    "type": "knn_vector",
                    "dimension": dimension,
                    "space_type": config.space_type,
                    "method": {
                        "name": "hnsw",
                        "engine": config.engine,
                        "parameters": {
                            "ef_construction": config.ef_construction,
                            "m": config.m,
                        },
                    },
                },
                # Stored payload — returned with each hit for display + citation.
                "chunk_id": {"type": "keyword"},
                "text": {"type": "text"},
                "source": {"type": "keyword"},
                "title": {"type": "text"},
                "section": {"type": "keyword"},
            }
        },
    }


def create_index(
    client: OpenSearch,
    config: IndexConfig = INDEX,
    dimension: int = 384,
    recreate: bool = True,
) -> None:
    """Create the index. If `recreate`, drop any existing one first.

    Dropping + recreating keeps ingestion idempotent: re-running always yields a
    clean index built from the current mapping, never a half-stale mix.
    """
    if recreate and client.indices.exists(index=config.index_name):
        client.indices.delete(index=config.index_name)
    if not client.indices.exists(index=config.index_name):
        client.indices.create(
            index=config.index_name,
            body=index_body(config, dimension),
        )


if __name__ == "__main__":
    #   uv run python -m src.opensearch_client
    client = get_client()
    info = client.info()
    print(f"connected: {info['version']['distribution']} {info['version']['number']}")
    create_index(client, dimension=384, recreate=True)
    mapping = client.indices.get_mapping(index=INDEX.index_name)
    emb = mapping[INDEX.index_name]["mappings"]["properties"]["embedding"]
    print(f"index '{INDEX.index_name}' created; embedding field -> {emb}")
