"""Step 3 (D3) — embed every chunk and bulk-load it into OpenSearch.

Pipeline: read data/chunks.jsonl -> batch-embed the text on the GPU -> write all
docs with the OpenSearch bulk API. Re-runnable: it recreates the index first, so
running it twice gives the same result (no duplicates).

Why batch + bulk? Two separate throughput wins:
  - batch encode: the GPU embeds 64 chunks per forward pass, far faster than one
    at a time.
  - bulk write: one HTTP request ships hundreds of docs, vs one round-trip each.

Each doc's `_id` is the chunk_id, so a re-ingest overwrites the same doc rather
than appending a copy — idempotent at the document level too.

Run with:  uv run python -m src.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from opensearchpy.helpers import bulk

from src.config import EMBED, INDEX
from src.embeddings import Embedder
from src.ingestion.build_chunks import CHUNKS_PATH
from src.opensearch_client import create_index, get_client

# How many docs per bulk request. 500 keeps each request a reasonable size
# (~500 * 384 floats) while still cutting round-trips dramatically.
BULK_BATCH = 500


def load_chunks(path: Path = CHUNKS_PATH) -> list[dict]:
    """Read the JSONL produced by D2. One dict per line:
    {chunk_id, text, source, title, section}.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python -m src.ingestion.build_chunks` first."
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _actions(chunks: list[dict], vectors):
    """Yield one bulk action per chunk, pairing it with its embedding vector."""
    for chunk, vec in zip(chunks, vectors):
        yield {
            "_index": INDEX.index_name,
            "_id": chunk["chunk_id"],  # stable id -> re-ingest overwrites
            "_source": {
                "embedding": vec.tolist(),  # numpy row -> json-able list
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
                "source": chunk["source"],
                "title": chunk["title"],
                "section": chunk["section"],
            },
        }


def ingest() -> int:
    """Build the index from scratch and load all chunks. Returns the doc count."""
    chunks = load_chunks()
    print(f"loaded {len(chunks)} chunks from {CHUNKS_PATH.name}")

    embedder = Embedder()
    print(f"embedding with {EMBED.model_name} on {embedder.device} ...")
    vectors = embedder.encode_passages([c["text"] for c in chunks])

    client = get_client()
    create_index(client, dimension=EMBED.dimension, recreate=True)
    print(f"(re)created index '{INDEX.index_name}', bulk-writing ...")

    success, errors = bulk(
        client,
        _actions(chunks, vectors),
        chunk_size=BULK_BATCH,
        request_timeout=120,
    )
    if errors:
        # bulk returns a list of per-item errors when raise_on_error=False;
        # by default it raises, so reaching here with errors is unexpected.
        print(f"WARNING: {len(errors)} bulk errors (first: {errors[0]})")

    # refresh makes the just-written docs searchable immediately (otherwise we
    # wait for the periodic refresh interval).
    client.indices.refresh(index=INDEX.index_name)
    count = client.count(index=INDEX.index_name)["count"]
    print(f"done: {success} indexed, index now holds {count} docs")
    return count


if __name__ == "__main__":
    ingest()
