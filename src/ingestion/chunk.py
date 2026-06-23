"""Step 2 — chunk the loaded docs into retrieval-sized pieces.

Why chunk at all? Embedding + retrieval operate on *passages*. A whole doc
(some are 50k+ chars) would (a) blow past the embedding model's input limit and
(b) dilute the signal — the matching sentence gets averaged together with
everything else, so retrieval becomes imprecise. Splitting into ~paragraph-sized
chunks lets us retrieve the specific passage that answers a question.

We use LangChain's `RecursiveCharacterTextSplitter`: it tries to split on the
biggest natural boundary first (paragraph -> line -> word -> char), so it avoids
cutting sentences in half where it can. Size/overlap come from `src.config`.

Each chunk inherits the parent doc's metadata (source/title/section) and gets a
stable `chunk_id` of the form "<source>::<n>" — traceable back to its file and
idempotent across re-runs (doc order is sorted, split is deterministic).

Output: a list[Document] ready for embedding. Nothing written to disk here.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK, ChunkConfig
from src.ingestion.loader import load_docs


def chunk_docs(
    docs: list[Document],
    config: ChunkConfig = CHUNK,
) -> list[Document]:
    """Split each Document into overlapping chunks with a stable chunk_id.

    Args:
        docs: cleaned documents from the loader.
        config: chunk_size / chunk_overlap (defaults to src.config.CHUNK).

    Returns:
        list[Document]; each chunk's metadata carries the parent's
        source/title/section plus a new `chunk_id`.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        # Defaults: separators ["\n\n", "\n", " ", ""] -> paragraph, line,
        # word, then hard char split as a last resort. length_function=len
        # (character count), matching how we reason about chunk_size.
    )

    chunks: list[Document] = []
    # Split per-document so chunk_id indices are local to each source file.
    for doc in docs:
        source = doc.metadata["source"]
        pieces = splitter.split_documents([doc])  # metadata is copied into each
        for i, piece in enumerate(pieces):
            piece.metadata["chunk_id"] = f"{source}::{i}"
            chunks.append(piece)

    return chunks


if __name__ == "__main__":
    # Quick manual check (W4 D2 validation skeleton). Run with:
    #   uv run python -m src.ingestion.chunk
    docs = load_docs()
    chunks = chunk_docs(docs)
    print(
        f"docs: {len(docs)} | chunks: {len(chunks)} "
        f"| chunk_size={CHUNK.chunk_size} overlap={CHUNK.chunk_overlap}"
    )

    # Length distribution (chars) — watch for many tiny or over-long chunks.
    lengths = sorted(len(c.page_content) for c in chunks)
    n = len(lengths)
    print(
        f"chunk length chars: min={lengths[0]} "
        f"median={lengths[n // 2]} max={lengths[-1]} "
        f"| avg chunks/doc={len(chunks) / len(docs):.1f}"
    )

    # Eyeball a sample: is it cut mid-sentence? is metadata + chunk_id intact?
    sample = chunks[0]
    print("\n--- sample chunk ---")
    print("metadata:", sample.metadata)
    print("content:\n", sample.page_content[:400])
