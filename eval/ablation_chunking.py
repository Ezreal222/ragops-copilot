"""W5 D5 — chunking ablation: vary chunk_size/overlap, re-ingest, re-eval.

This is the first proper ablation study of the project: change ONE thing at a
time (chunk_size / overlap) and measure the effect on retrieval, to attack D3's
weakest number — context_recall (0.53).

For each (chunk_size, overlap) group we:
  1. re-chunk the whole corpus with that config and write data/chunks.jsonl,
  2. re-ingest — recreate the OpenSearch index and re-embed every chunk
     (idempotent: recreate=True, so no leftovers from the previous group),
  3. run ONE retrieval pass over the frozen eval set (reranker OFF), and from
     that single pass derive BOTH
       - doc-level recall@1/3/5  (local, no LLM), and
       - the RAGAS context inputs (retrieved chunk texts + gold reference),
  4. score RAGAS context_recall + context_precision on those inputs.

Why this isolates the variable:
  - Only chunk_size/overlap move. Same embedding model, same k, reranker OFF,
    same frozen eval set, same judge (temperature 0). So any metric delta is
    attributable to chunking — that's what makes an ablation an ablation.
  - We use DOC-level recall (gold_sources), not chunk-level, because chunk_ids
    are re-minted every time we re-chunk — "docs/foo.md::5" means something
    different at 400 chars than at 1200. The gold DOC is stable; the gold chunk
    id is not. (This is the W4 lesson baked in.)

Why no generation: context_recall / context_precision only need the retrieved
contexts + the gold reference answer — NOT our system's generated answer. So we
skip the expensive per-question LLM generation entirely and just retrieve. The
only LLM cost is the RAGAS judge scoring the two context metrics.

Run:
    uv run python -m eval.ablation_chunking --smoke   # baseline only, 3 Q — wiring test
    uv run python -m eval.ablation_chunking           # full grid (4 groups)
    uv run python -m eval.ablation_chunking --no-ragas # doc-recall only (fast, no judge)

Leaves the index built from the LAST group in the grid. Step 3 (pick best) will
re-ingest the chosen config and update config.py.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# --- compat shim (must run BEFORE importing ragas) -------------------------
# Same reason as run_ragas.py: ragas imports a langchain-community Vertex shim
# that no longer exists in our pinned stack. We never use Vertex, so stub it.
_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertex)
# ---------------------------------------------------------------------------

import csv  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
from time import perf_counter  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ChunkConfig, RETRIEVAL  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.ingest import ingest  # noqa: E402
from src.ingestion.build_chunks import build_chunks  # noqa: E402
from src.opensearch_client import get_client  # noqa: E402
from src.retrieve import search  # noqa: E402

# Reuse the frozen judge builder from the D3 harness, so the ablation scores with
# the EXACT same judge model/temperature as the baseline.
from eval.run_ragas import JUDGE_MODEL, build_judge  # noqa: E402

EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "ablation_chunking.csv"


def load_answerable() -> list[dict]:
    """Load the eval set, keeping only questions that HAVE a gold document.

    The eval set mixes in "no-answer" questions (out-of-scope; the system should
    refuse). Those have no gold_sources, so retrieval recall is undefined for
    them — including them would just floor every group's recall by the same
    constant and add noise. A chunking ablation measures RETRIEVAL, so we score
    only the answerable questions here. (Refusal quality is a generation concern,
    measured elsewhere.)
    """
    rows = [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8") if line.strip()]
    return [r for r in rows if r.get("gold_sources")]

# The ablation grid (from the W5 D5 spec). Keep it small (4 groups) so the whole
# sweep is a coffee break, not an afternoon. label, chunk_size, overlap.
GRID: list[tuple[str, int, int]] = [
    ("A baseline", 800, 100),
    ("B small", 400, 50),
    ("C large", 1200, 200),
    ("D big-overlap", 800, 200),
]

# Retrieve this many per query; doc-recall@k slices the top of this list. The
# first K_RAGAS are handed to RAGAS as the retrieved context (matches serving
# top_k so the numbers reflect what a user would actually get).
RETRIEVE_K = 10
K_VALUES = (1, 3, 5)
K_RAGAS = RETRIEVAL.top_k  # 5


def reingest(size: int, overlap: int) -> int:
    """Re-chunk the corpus at (size, overlap) and rebuild the index.

    Returns the number of chunks now in the index. build_chunks writes
    data/chunks.jsonl for this config; ingest() recreates the index and embeds
    every chunk, so nothing survives from the previous group.
    """
    cfg = ChunkConfig(chunk_size=size, chunk_overlap=overlap)
    build_chunks(config=cfg)
    return ingest()  # recreate=True inside — idempotent rebuild


def retrieve_pass(rows: list[dict], embedder: Embedder, client) -> tuple[dict, list[dict], float]:
    """One retrieval pass over the eval set (reranker OFF).

    From the single ranked list per question we derive two things at once:
      - doc-level recall@k  (did any gold SOURCE appear in the top-k hits?)
      - RAGAS inputs        (the top-K_RAGAS chunk texts + the gold reference)

    Returns (recall_doc_by_k, ragas_samples, mean_latency_ms).
    """
    doc_ranks: list[int | None] = []
    ragas_samples: list[dict] = []
    latencies_ms: list[float] = []

    for r in rows:
        t0 = perf_counter()
        hits = search(r["q"], k=RETRIEVE_K, client=client, embedder=embedder, use_reranker=False)
        latencies_ms.append((perf_counter() - t0) * 1000)

        # Doc-level: rank (1-based) of the first hit whose SOURCE is a gold doc.
        gold_sources = set(r.get("gold_sources", []))
        rank = next((i for i, h in enumerate(hits, 1) if h["source"] in gold_sources), None)
        doc_ranks.append(rank)

        ragas_samples.append(
            {
                "user_input": r["q"],
                "retrieved_contexts": [h["text"] for h in hits[:K_RAGAS]],
                "reference": r["reference"],
            }
        )

    n = len(rows)
    recall_doc = {
        k: sum(1 for rank in doc_ranks if rank and rank <= k) / n for k in K_VALUES
    }
    return recall_doc, ragas_samples, statistics.mean(latencies_ms)


def score_context_metrics(samples: list[dict], judge) -> dict:
    """Run RAGAS context_recall + context_precision on the retrieval inputs.

    These two are the ONLY metrics chunking directly moves, and neither needs
    the generated answer — just (question, retrieved_contexts, reference). We
    import ragas lazily (after the Vertex shim above) and reuse the frozen judge.
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.metrics import context_precision, context_recall
    from ragas.run_config import RunConfig

    ds = EvaluationDataset.from_list(samples)
    # Low workers + generous timeout: DeepSeek rate-limits; the judge fires many
    # small calls. Same knobs as run_ragas.py for comparability.
    run_config = RunConfig(timeout=180, max_workers=4)
    result = evaluate(
        ds,
        metrics=[context_precision, context_recall],
        llm=judge,
        run_config=run_config,
    )
    df = result.to_pandas()
    return {
        "context_recall": float(df["context_recall"].mean()),
        "context_precision": float(df["context_precision"].mean()),
    }


def main() -> None:
    smoke = "--smoke" in sys.argv
    run_ragas = "--no-ragas" not in sys.argv

    rows = load_answerable()
    grid = GRID[:1] if smoke else GRID
    if smoke:
        rows = rows[:3]

    mode = "SMOKE" if smoke else "FULL"
    ragas_note = f"+ RAGAS (judge={JUDGE_MODEL})" if run_ragas else "(doc-recall only)"
    print(f"chunking ablation — {mode} | {len(grid)} group(s) x {len(rows)} Q | reranker OFF | {ragas_note}\n")

    # The judge is built ONCE and reused across every group — identical scoring.
    judge = build_judge() if run_ragas else None
    # Embedder is rebuilt per group by ingest(); for retrieval we build one here
    # and reuse it across all questions of all groups (query embedding is the
    # same model regardless of how the corpus was chunked).
    embedder = Embedder()
    client = get_client()

    results: list[dict] = []
    for label, size, overlap in grid:
        print(f"=== {label}  (size={size}, overlap={overlap}) ===")
        n_chunks = reingest(size, overlap)
        recall_doc, samples, mean_lat = retrieve_pass(rows, embedder, client)

        row = {
            "group": label,
            "chunk_size": size,
            "overlap": overlap,
            "n_chunks": n_chunks,
            **{f"recall@{k}_doc": recall_doc[k] for k in K_VALUES},
            "mean_latency_ms": round(mean_lat, 1),
        }
        if run_ragas:
            print("  scoring RAGAS context metrics ...")
            row.update(score_context_metrics(samples, judge))
        results.append(row)

        rr = "  ".join(f"r@{k}={recall_doc[k]:.3f}" for k in K_VALUES)
        extra = (
            f"  ctx_recall={row['context_recall']:.3f}  ctx_prec={row['context_precision']:.3f}"
            if run_ragas
            else ""
        )
        print(f"  chunks={n_chunks}  {rr}{extra}  lat={mean_lat:.0f}ms\n")

    _report(results, run_ragas, smoke)


def _report(results: list[dict], run_ragas: bool, smoke: bool) -> None:
    """Print the ablation table and (on a full run) write the CSV."""
    cols = ["group", "chunk_size", "overlap", "n_chunks",
            "recall@1_doc", "recall@3_doc", "recall@5_doc"]
    if run_ragas:
        cols += ["context_recall", "context_precision"]
    cols += ["mean_latency_ms"]

    print("=== ablation table ===")
    header = "  ".join(f"{c:>15}" for c in cols)
    print(header)
    for r in results:
        cells = []
        for c in cols:
            v = r[c]
            cells.append(f"{v:>15.3f}" if isinstance(v, float) else f"{str(v):>15}")
        print("  ".join(cells))

    if run_ragas:
        best = max(results, key=lambda r: r["context_recall"])
        print(f"\nhighest context_recall: {best['group']} "
              f"(size={best['chunk_size']}, overlap={best['overlap']}) "
              f"-> {best['context_recall']:.3f}")

    if not smoke:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows({c: r[c] for c in cols} for r in results)
        print(f"\ntable -> {OUT_CSV}")


if __name__ == "__main__":
    main()
