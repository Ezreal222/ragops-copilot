"""Measure retrieval quality on the hand-labeled eval set: recall@1/3/5/10.

it turns "retrieval" into a number we can
compare against late.

recall@k (per the spec) = fraction of questions for which AT LEAST ONE gold
chunk appears in the top-k retrieved chunks. It answers "did we retrieve the
right doc at all within the top k?" — the ceiling on how well the LLM can
answer, since it can only ground on what we retrieve.

How it works:
  - Build the embedder + OpenSearch client ONCE (reused across all questions, so
    we don't reload the model per query).
  - For each question, retrieve the top-10 chunk_ids a single time, then compute
    recall@k by slicing that ranked list to the first k — no need to re-query
    per k.
  - Also record per-query latency (mean + median) — retrieval speed matters for
    serving later.

A question counts as a "hit@k" if any of its gold_chunk_ids is in the top-k.

Run with:  uv run python -m eval.eval_retrieval
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path
from time import perf_counter

# Make `src` importable whether run as a module or a script.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import Embedder  # noqa: E402
from src.opensearch_client import get_client  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.retrieve import search  # noqa: E402

EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"

# Retrieve this many per query; recall@k is computed for each k in K_VALUES by
# slicing the single top-RETRIEVE_K result list. K_VALUES must all be <= this.
RETRIEVE_K = 10
K_VALUES = (1, 3, 5, 10)


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    """Read eval/eval_set.jsonl -> [{q, gold_chunk_ids}, ...]."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — write the eval set first.")
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    for r in rows:
        if not r.get("gold_chunk_ids"):
            raise ValueError(f"question has no gold_chunk_ids: {r.get('q')!r}")
    return rows


def evaluate(rows: list[dict], use_reranker: bool = False) -> dict:
    """Retrieve for every question, return metrics + per-question records.

    `use_reranker=False` is the bi-encoder baseline (D4); True adds the D5
    cross-encoder rerank stage. The reranker is built once and injected so the
    measured latency reflects steady-state serving, not a per-query model load.
    """
    # Build heavy objects once and reuse — the whole point of search()'s
    # injectable client/embedder/reranker.
    embedder = Embedder()
    client = get_client()
    reranker = Reranker() if use_reranker else None

    records = []
    latencies_ms = []
    for r in rows:
        t0 = perf_counter()
        hits = search(
            r["q"],
            k=RETRIEVE_K,
            client=client,
            embedder=embedder,
            use_reranker=use_reranker,
            reranker=reranker,
        )
        latencies_ms.append((perf_counter() - t0) * 1000)

        retrieved_ids = [h["chunk_id"] for h in hits]
        gold = set(r["gold_chunk_ids"])
        # rank (1-based) of the first gold chunk in the result list, or None.
        rank = next((i for i, cid in enumerate(retrieved_ids, 1) if cid in gold), None)

        # Document-level: did we surface the right DOC, ignoring which chunk of
        # it? chunk_id is "path::N", so the doc is everything before "::". This
        # is fairer to the reranker, which often promotes a sibling chunk of the
        # gold doc over the exact gold chunk our labels happen to name.
        gold_docs = {cid.split("::")[0] for cid in gold}
        doc_rank = next(
            (i for i, cid in enumerate(retrieved_ids, 1) if cid.split("::")[0] in gold_docs),
            None,
        )
        records.append(
            {
                "q": r["q"],
                "gold": r["gold_chunk_ids"],
                "retrieved": retrieved_ids,
                "first_gold_rank": rank,
                "first_gold_doc_rank": doc_rank,
            }
        )

    # recall@k = (# questions whose first gold lands within top-k) / total.
    # Computed both at chunk granularity (exact chunk_id) and doc granularity.
    n = len(records)
    recall = {
        k: sum(1 for rec in records if rec["first_gold_rank"] and rec["first_gold_rank"] <= k) / n
        for k in K_VALUES
    }
    recall_doc = {
        k: sum(1 for rec in records if rec["first_gold_doc_rank"] and rec["first_gold_doc_rank"] <= k) / n
        for k in K_VALUES
    }
    return {
        "n": n,
        "recall": recall,
        "recall_doc": recall_doc,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
        },
        "records": records,
    }


def main() -> None:
    # `--rerank` flips on the D5 cross-encoder stage so the same script produces
    # both sides of the before/after comparison.
    use_reranker = "--rerank" in sys.argv
    mode = "bi-encoder + rerank" if use_reranker else "bi-encoder only (baseline)"

    rows = load_eval_set()
    res = evaluate(rows, use_reranker=use_reranker)

    print(f"\neval set: {res['n']} questions  (retrieve top-{RETRIEVE_K})  | mode: {mode}\n")
    print("recall@k       chunk-level   doc-level")
    for k in K_VALUES:
        print(f"  recall@{k:<2}     {res['recall'][k]:.3f}         {res['recall_doc'][k]:.3f}")
    lat = res["latency_ms"]
    print(f"\nlatency/query: mean {lat['mean']:.1f} ms | median {lat['median']:.1f} ms")

    # Failure analysis input: questions whose gold never made the top-5. These
    # are the cases D5's reranker / W5 ablations need to fix.
    misses = [r for r in res["records"] if not r["first_gold_rank"] or r["first_gold_rank"] > 5]
    print(f"\nmisses @5 ({len(misses)}):")
    for r in misses:
        rank = r["first_gold_rank"] or "not in top-10"
        print(f"  - first gold rank: {rank}")
        print(f"    q:    {r['q']}")
        print(f"    gold: {r['gold']}")
        print(f"    top3: {r['retrieved'][:3]}")


if __name__ == "__main__":
    main()
