"""Unit tests for src.ingestion.chunk.chunk_docs.

Chunking has two properties the rest of the pipeline relies on and that a
refactor could quietly break:

  1. **Stable, traceable chunk_id** of the form "<source>::<n>" — this is what
     lets a citation point back to a specific piece of a specific file, and what
     makes re-ingestion idempotent (same input -> same ids).
  2. **Metadata inheritance** — each chunk keeps its parent doc's source/title,
     so a retrieved chunk still knows where it came from.

No embedding, no OpenSearch: chunk_docs is pure text splitting, so it's a clean
hermetic unit test. We build tiny Documents by hand instead of loading the real
corpus so the assertions don't depend on the current docs on disk.
"""

from __future__ import annotations

from langchain_core.documents import Document

from src.config import ChunkConfig
from src.ingestion.chunk import chunk_docs

# A small config so a short synthetic doc still splits into several chunks —
# lets us assert on multi-chunk behaviour (ids, overlap) without a huge fixture.
SMALL = ChunkConfig(chunk_size=40, chunk_overlap=10)


def _doc(source: str, text: str, title: str = "T") -> Document:
    return Document(page_content=text, metadata={"source": source, "title": title})


def test_chunk_ids_are_stable_and_source_scoped():
    doc = _doc("a.md", "word " * 60)  # ~300 chars -> several 40-char chunks
    chunks = chunk_docs([doc], config=SMALL)

    assert len(chunks) > 1, "fixture should split into multiple chunks"
    ids = [c.metadata["chunk_id"] for c in chunks]
    # Ids are "<source>::<n>", numbered from 0, contiguous, in order.
    assert ids == [f"a.md::{i}" for i in range(len(chunks))]


def test_ids_reset_per_source_file():
    # Indices are LOCAL to each file, so two files both start at ::0. This is what
    # keeps ids stable when an unrelated doc is added/removed elsewhere.
    docs = [_doc("a.md", "word " * 60), _doc("b.md", "word " * 60)]
    chunks = chunk_docs(docs, config=SMALL)

    assert "a.md::0" in {c.metadata["chunk_id"] for c in chunks}
    assert "b.md::0" in {c.metadata["chunk_id"] for c in chunks}


def test_metadata_is_inherited_by_every_chunk():
    doc = _doc("a.md", "word " * 60, title="PagedAttention")
    chunks = chunk_docs([doc], config=SMALL)

    assert chunks, "expected at least one chunk"
    for c in chunks:
        assert c.metadata["source"] == "a.md"
        assert c.metadata["title"] == "PagedAttention"


def test_determinism_same_input_same_output():
    # Idempotent ingestion depends on this: chunking the same doc twice must give
    # byte-identical content and ids.
    doc = _doc("a.md", "word " * 60)
    first = chunk_docs([doc], config=SMALL)
    second = chunk_docs([doc], config=SMALL)

    assert [(c.page_content, c.metadata["chunk_id"]) for c in first] == [
        (c.page_content, c.metadata["chunk_id"]) for c in second
    ]
