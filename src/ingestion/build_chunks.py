"""Step 3 — materialize chunks to data/chunks.jsonl for embedding.

This is the end of the offline ingestion stage: load -> clean -> chunk ->
**persist**. Reads `chunks.jsonl` directly to embed + index, so it doesn't
re-run loading/chunking every time.

Format: JSON Lines (one self-contained JSON object per line). Easy to stream,
diff-friendly, and append-free here (we overwrite, keeping ingestion idempotent —
re-running produces the same file, never duplicates).

Each line:
    {"chunk_id": "<source>::<n>", "text": "...", "source": "...",
     "title": "...", "section": "..."}
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import CHUNK
from src.ingestion.chunk import chunk_docs
from src.ingestion.loader import REPO_ROOT, load_docs

# data/ is gitignored, so this intermediate artifact stays local.
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"


def build_chunks(out_path: Path = CHUNKS_PATH) -> Path:
    """Load -> chunk -> write JSONL. Returns the output path.

    Overwrites `out_path` so the run is idempotent.
    """
    docs = load_docs()
    chunks = chunk_docs(docs)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            row = {
                "chunk_id": c.metadata["chunk_id"],
                "text": c.page_content,
                "source": c.metadata["source"],
                "title": c.metadata["title"],
                "section": c.metadata["section"],
            }
            # ensure_ascii=False keeps non-ASCII (e.g. ≈, —) readable on disk.
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return out_path


if __name__ == "__main__":
    # Run with:  uv run python -m src.ingestion.build_chunks
    docs = load_docs()
    chunks = chunk_docs(docs)
    out = build_chunks()

    lengths = sorted(len(c.page_content) for c in chunks)
    n = len(lengths)
    print(f"wrote {len(chunks)} chunks from {len(docs)} docs -> {out}")
    print(f"config: chunk_size={CHUNK.chunk_size} overlap={CHUNK.chunk_overlap}")
    print(
        f"chunk length chars: min={lengths[0]} median={lengths[n // 2]} "
        f"max={lengths[-1]} avg={sum(lengths) // n} | avg chunks/doc={n / len(docs):.1f}"
    )
    print(f"file size: {out.stat().st_size / 1024:.0f} KiB")
