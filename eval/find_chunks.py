"""Find candidate gold chunks by keyword — a labeling aid for the eval set.

for each eval question, decide
which chunk(s) actually contain the answer and record their `chunk_id`. This
helper just makes that lookup fast. It does a plain case-insensitive substring
search over data/chunks.jsonl (NO embeddings, NO model load) grep the
corpus for a phrase and copy the right chunk_id into eval/eval_set.jsonl.

Why keyword (not semantic) search here? For *labeling* you want the literal
chunk that states the answer, found deterministically — not whatever the
retriever ranks highest. The semantic retriever is the thing we're measuring;
gold must be chosen independently of it, or recall@k just grades the model
against itself.

Usage:
    uv run python eval/find_chunks.py "continuous batching"
    uv run python eval/find_chunks.py "paged attention" --limit 5 --width 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

CHUNKS_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks.jsonl"


def search(keyword: str, limit: int, width: int) -> None:
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"{CHUNKS_PATH} not found. Run `uv run python -m src.ingestion.build_chunks` first."
        )

    needle = keyword.lower()
    found = 0
    with CHUNKS_PATH.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            chunk = json.loads(line)
            if needle not in chunk["text"].lower():
                continue

            found += 1
            # Show a snippet centred on the first match so you can confirm the
            # chunk really answers the question before copying its id.
            text = chunk["text"]
            pos = text.lower().find(needle)
            start = max(0, pos - width // 2)
            snippet = text[start : start + width].replace("\n", " ")

            print(f"\n[{found}] {chunk['chunk_id']}")
            print(f"    title:   {chunk['title']}")
            print(f"    source:  {chunk['source']}  (section: {chunk['section']})")
            print(f"    snippet: ...{snippet}...")

            if found >= limit:
                print(f"\n(stopped at --limit {limit}; refine the keyword to narrow down)")
                return

    print(f"\n{found} chunk(s) matched '{keyword}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keyword search over the chunk corpus to pick gold chunk_ids.")
    parser.add_argument("keyword", help="substring to search for (case-insensitive)")
    parser.add_argument("--limit", type=int, default=10, help="max matches to print (default 10)")
    parser.add_argument("--width", type=int, default=160, help="snippet width in chars (default 160)")
    args = parser.parse_args()
    search(args.keyword, args.limit, args.width)
