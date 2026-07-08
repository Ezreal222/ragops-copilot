"""Derive the low-score refusal threshold from the eval set (W6 D4, guardrail 3).

Pure k-NN ALWAYS returns top-k — there is no "empty" result, so an off-topic
question still gets 5 chunks back, and if we feed them to the LLM it may stitch
together a confident-but-wrong answer. The fix is a *relevance threshold*: if the
best retrieved chunk scores below some cutoff, treat retrieval as "nothing
relevant found" and refuse instead of grounding on noise.

The spec says: don't pick the cutoff by gut — look at the score distribution of
real hits vs misses and choose a boundary. This script produces that evidence:

  - ON-TOPIC : the 36 hand-written eval questions (known to be answerable). Their
    top-1 k-NN score is what a genuine hit looks like.
  - OFF-TOPIC: queries with no answer in the vLLM docs. Their top-1 score is what
    noise looks like — the score a chunk gets just for being the "least far" of
    an irrelevant bunch.

A clean threshold sits ABOVE the off-topic scores and BELOW the on-topic ones.
We print min/median/max for each group plus a suggested cutoff; the chosen value
goes into AgentConfig.min_relevance_score with this file cited as the basis.

Score note: OpenSearch's lucene k-NN with space_type=cosinesimil returns
_score = 1 / (1 + (1 - cosine)) = 1 / (2 - cosine), so identical vectors -> 1.0
and orthogonal -> 0.5. Thresholds below live in that 0.5..1.0 band.

    uv run python -m eval.analyze_score_threshold
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.config import RETRIEVAL
from src.embeddings import Embedder
from src.opensearch_client import get_client
from src.retrieve import search

EVAL_SET = Path(__file__).with_name("eval_set.jsonl")

# Queries with NO answer in the vLLM docs — the "miss" class. A mix of totally
# unrelated topics and one that name-drops vLLM but asks nonsense, so the cutoff
# has to reject even superficially on-brand noise.
OFF_TOPIC = [
    "What is the capital of France?",
    "How do I bake sourdough bread at home?",
    "Explain how photosynthesis works in plants.",
    "What are the best stocks to buy this year?",
    "Can vLLM make me a cup of coffee?",
    "Write a poem about the ocean at sunset.",
    "How do I potty train a puppy?",
    "What is the offside rule in soccer?",
]


def top1_scores(queries: list[str], embedder: Embedder, client) -> list[float]:
    """Return the top-1 retrieval score for each query (top of the k-NN list)."""
    scores = []
    for q in queries:
        hits = search(q, k=RETRIEVAL.top_k, client=client, embedder=embedder,
                      use_reranker=False)
        scores.append(hits[0]["score"] if hits else 0.0)
    return scores


def summary(name: str, scores: list[float]) -> dict:
    scores = sorted(scores)
    stats = {
        "n": len(scores),
        "min": min(scores),
        "median": statistics.median(scores),
        "max": max(scores),
    }
    print(f"{name:<10} n={stats['n']:<3} "
          f"min={stats['min']:.3f}  median={stats['median']:.3f}  max={stats['max']:.3f}")
    return stats


if __name__ == "__main__":
    embedder = Embedder()      # build heavy objects once
    client = get_client()

    on_q = [json.loads(line)["q"] for line in EVAL_SET.read_text().splitlines() if line.strip()]

    print(f"Scoring {len(on_q)} on-topic + {len(OFF_TOPIC)} off-topic queries "
          f"(top-1 k-NN score, use_reranker=False)\n")
    on = top1_scores(on_q, embedder, client)
    off = top1_scores(OFF_TOPIC, embedder, client)

    print("=== top-1 score distribution ===")
    on_s = summary("on-topic", on)
    off_s = summary("off-topic", off)

    # A safe cutoff sits in the gap between the highest miss and the lowest hit.
    # If they overlap, split the difference toward the miss side (favor recall:
    # better to answer a borderline real question than to over-refuse).
    gap_lo, gap_hi = off_s["max"], on_s["min"]
    if gap_hi > gap_lo:
        suggested = round((gap_lo + gap_hi) / 2, 3)
        note = f"clean gap [{gap_lo:.3f}, {gap_hi:.3f}] -> midpoint"
    else:
        # Overlap: put the cutoff just above the off-topic median so typical noise
        # is refused while most real hits still pass.
        suggested = round(off_s["median"] + 0.01, 3)
        note = f"overlap (off max {gap_lo:.3f} >= on min {gap_hi:.3f}) -> off median + 0.01"

    print(f"\nsuggested min_relevance_score = {suggested}  ({note})")
    # How much of each class the suggested cutoff would keep/reject — the number
    # to record as the basis.
    on_pass = sum(1 for s in on if s >= suggested)
    off_pass = sum(1 for s in off if s >= suggested)
    print(f"  on-topic kept : {on_pass}/{len(on)}  (higher = fewer false refusals)")
    print(f"  off-topic kept: {off_pass}/{len(off)} (lower  = more noise refused)")
