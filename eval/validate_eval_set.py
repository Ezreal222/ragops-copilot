"""Validate the hand-labeled eval set and print its statistics.

The eval set (`eval/eval_set.jsonl`) is the *ground truth* of the whole project:
every W5 metric is graded against it. A single typo'd chunk_id silently turns a
correct retrieval into a "miss" and corrupts recall@k. This script is the guard
that catches those mistakes before they poison the numbers.

What it checks, per row:
  - required fields are present and well-typed (q / gold_chunk_ids /
    gold_sources / reference);
  - every gold_chunk_id actually EXISTS in data/chunks.jsonl (catches typos);
  - every gold_source is a REAL source path in the corpus;
  - gold_chunk_ids and gold_sources agree (each gold chunk's document is listed
    in gold_sources) — they describe the same answer at two granularities;
  - "no-answer" questions (empty gold, used to test refusal / anti-hallucination)
    still carry a hand-written `reference` saying the docs don't cover it.

What it prints (the D2 stats to paste into notes/w5.md):
  - question count, # answerable vs # no-answer (refusal tests);
  - mean gold chunks / sources per answerable question;
  - distribution of gold over top-level doc areas (docs/<area>/...), to spot
    topic coverage gaps.

This is a labeling aid, NOT a generator: it never invents gold or references —
it only verifies what a human wrote. Run it repeatedly while labeling.

Run with:  uv run python -m eval.validate_eval_set
Exit code is non-zero if any hard error is found (so it can gate a commit/CI).
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = REPO_ROOT / "data" / "chunks.jsonl"
EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"

# Required keys every row must have. gold_* may be empty lists (no-answer
# questions), but the keys must be present so the schema is uniform.
REQUIRED_FIELDS = ("q", "gold_chunk_ids", "gold_sources", "reference")


def load_corpus_ids(path: Path = CHUNKS_PATH) -> tuple[set[str], set[str]]:
    """Return (all chunk_ids, all source paths) present in the corpus.

    We validate gold against these sets so a mistyped id can't slip through.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python -m src.ingestion.build_chunks` first."
        )
    chunk_ids: set[str] = set()
    sources: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            chunk_ids.add(chunk["chunk_id"])
            sources.add(chunk["source"])
    return chunk_ids, sources


def load_rows(path: Path = EVAL_PATH) -> list[dict]:
    """Read eval_set.jsonl into a list of dicts (one per question)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — write the eval set first.")
    rows = []
    for i, line in enumerate(path.open(encoding="utf-8"), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"line {i}: invalid JSON — {e}") from e
    return rows


def validate(rows: list[dict], chunk_ids: set[str], sources: set[str]) -> tuple[list[str], list[str]]:
    """Check every row against the schema and the corpus.

    Returns (errors, warnings). Errors are hard failures (bad/missing data that
    would corrupt metrics); warnings are smells worth a human glance but not
    fatal (e.g. a suspiciously short reference).
    """
    errors: list[str] = []
    warnings: list[str] = []
    seen_questions: set[str] = set()

    for idx, r in enumerate(rows, 1):
        tag = f"Q{idx}"  # stable label for messages (q text may be long)

        # 1. Required fields present.
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            errors.append(f"{tag}: missing field(s) {missing}")
            continue  # can't check further without the fields

        q = r["q"]
        gold_chunks = r["gold_chunk_ids"]
        gold_sources = r["gold_sources"]
        reference = r["reference"]

        # 2. Types.
        if not isinstance(q, str) or not q.strip():
            errors.append(f"{tag}: 'q' must be a non-empty string")
        if not isinstance(gold_chunks, list):
            errors.append(f"{tag}: 'gold_chunk_ids' must be a list")
            gold_chunks = []
        if not isinstance(gold_sources, list):
            errors.append(f"{tag}: 'gold_sources' must be a list")
            gold_sources = []
        if not isinstance(reference, str):
            errors.append(f"{tag}: 'reference' must be a string")
            reference = ""

        # 3. Duplicate question text (likely an accidental copy-paste).
        if q in seen_questions:
            warnings.append(f"{tag}: duplicate question text — {q!r}")
        seen_questions.add(q)

        # 4. Every gold chunk_id must exist in the corpus.
        for cid in gold_chunks:
            if cid not in chunk_ids:
                errors.append(f"{tag}: gold_chunk_id not in chunks.jsonl: {cid!r}")

        # 5. Every gold source must be a real source path.
        for src in gold_sources:
            if src not in sources:
                errors.append(f"{tag}: gold_source not in corpus: {src!r}")

        # 6. Cross-check: each gold chunk's document should be listed in
        #    gold_sources (they describe the same answer at two granularities).
        chunk_docs = {cid.split("::")[0] for cid in gold_chunks}
        for doc in chunk_docs:
            if doc not in set(gold_sources):
                warnings.append(
                    f"{tag}: doc {doc!r} appears in gold_chunk_ids but not in gold_sources"
                )

        # 7. No-answer questions (refusal tests): empty gold is allowed, but a
        #    reference is still required (the ideal "not in the docs" answer).
        is_no_answer = not gold_chunks and not gold_sources
        if not reference.strip():
            errors.append(f"{tag}: 'reference' is empty (write the ideal answer by hand)")
        elif is_no_answer and len(reference.strip()) < 10:
            warnings.append(f"{tag}: no-answer question with a very short reference")

        # 8. Partial labeling smell: one of the two gold granularities filled
        #    but not the other (likely forgot to fill the second).
        if bool(gold_chunks) != bool(gold_sources):
            warnings.append(
                f"{tag}: only one of gold_chunk_ids / gold_sources is filled "
                "(fill both, or leave both empty for a no-answer question)"
            )

    return errors, warnings


def print_stats(rows: list[dict]) -> None:
    """Print the D2 statistics block for notes/w5.md."""
    # Use .get with safe defaults so stats still print on a partially-labeled
    # set (this runs even after validation errors, as a progress readout).
    n = len(rows)
    no_answer = [r for r in rows if not r.get("gold_chunk_ids") and not r.get("gold_sources")]
    answerable = [r for r in rows if r not in no_answer]
    n_ans = len(answerable)

    avg_chunks = (
        sum(len(r.get("gold_chunk_ids") or []) for r in answerable) / n_ans if n_ans else 0.0
    )
    avg_sources = (
        sum(len(r.get("gold_sources") or []) for r in answerable) / n_ans if n_ans else 0.0
    )

    # Topic coverage by top-level doc area, e.g. docs/design/... -> "design".
    def area(source: str) -> str:
        parts = source.split("/")
        return parts[1] if len(parts) > 2 else (parts[0] if parts else source)

    area_counts: Counter[str] = Counter()
    for r in answerable:
        for src in set(r.get("gold_sources") or []):
            area_counts[area(src)] += 1

    print("\n=== eval set stats ===")
    print(f"  questions:            {n}")
    print(f"  answerable:           {n_ans}")
    print(f"  no-answer (refusal):  {len(no_answer)}")
    print(f"  avg gold chunks / answerable Q:   {avg_chunks:.2f}")
    print(f"  avg gold sources / answerable Q:  {avg_sources:.2f}")
    print("  gold coverage by doc area:")
    for a, c in area_counts.most_common():
        print(f"    {a:<20} {c}")


def main() -> None:
    chunk_ids, sources = load_corpus_ids()
    rows = load_rows()

    if not rows:
        print("eval_set.jsonl is empty — nothing to validate yet.")
        sys.exit(1)

    errors, warnings = validate(rows, chunk_ids, sources)

    if warnings:
        print(f"\n⚠️  {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print(f"\n❌ {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        print("\nschema validation FAILED — fix the errors above.")
        print_stats(rows)
        sys.exit(1)

    print("\n✓ schema validation passed — all gold ids/sources exist, all fields present.")
    print_stats(rows)


if __name__ == "__main__":
    main()
