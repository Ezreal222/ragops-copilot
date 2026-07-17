"""Unit tests for the citation layer in src.generate.

Citations are the product's core trust guarantee: every [n] in an answer must
resolve to a real retrieved source, and numbers that don't must be dropped
rather than shown. `_cited_sources` is the pure function that enforces that, so
it's exactly the kind of logic CI should protect — no network, no LLM, just
string -> mapping.

`format_context` is the other half of the contract: it numbers chunks 1-based so
that citation [1] refers to chunks[0]. If that numbering ever drifts, every
citation silently points at the wrong source, so we pin it too.

Note: importing src.generate pulls in the embedding stack (torch via
sentence-transformers) at module load. That's why CI installs the `cpu` group —
these tests need the import to resolve, not a GPU.
"""

from __future__ import annotations

from src.generate import _cited_sources, format_context


def _chunk(source: str, title: str, text: str = "body") -> dict:
    """Minimal chunk dict shaped like what the retriever hands to generate()."""
    return {"source": source, "title": title, "text": text}


CHUNKS = [
    _chunk("a.md", "Alpha"),
    _chunk("b.md", "Bravo"),
    _chunk("c.md", "Charlie"),
]


def test_maps_markers_to_sources_in_order():
    cited = _cited_sources("First [2], then [1].", CHUNKS)
    # Order of FIRST appearance, not sorted numerically: [2] then [1].
    assert [c["n"] for c in cited] == [2, 1]
    assert cited[0] == {"n": 2, "source": "b.md", "title": "Bravo"}
    assert cited[1] == {"n": 1, "source": "a.md", "title": "Alpha"}


def test_out_of_range_markers_are_ignored():
    # The model wrote [9] but only 3 chunks exist — it must not resolve to
    # anything (no IndexError, no phantom citation).
    cited = _cited_sources("See [9] and [2].", CHUNKS)
    assert [c["n"] for c in cited] == [2]


def test_duplicate_markers_are_deduped():
    cited = _cited_sources("[1] and again [1] and [1].", CHUNKS)
    assert [c["n"] for c in cited] == [1]


def test_no_markers_yields_no_citations():
    assert _cited_sources("A plain answer with no brackets.", CHUNKS) == []


def test_format_context_is_1_based_and_source_tagged():
    # Citation [1] must map to chunks[0]; the block for the first chunk therefore
    # starts with "[1]" and carries its source tag.
    rendered = format_context(CHUNKS)
    assert rendered.startswith("[1] (source: a.md)")
    assert "[2] (source: b.md)" in rendered
    assert "[3] (source: c.md)" in rendered
    # 1-based means there is no [0].
    assert "[0]" not in rendered
