"""W5 D6 — retrieval/rerank ablation: sweep query-time knobs → before/after table.

This is the SECOND ablation of the project and the week's core deliverable. D5
locked the corpus chunking (1200/200, now sitting in the index), so here we hold
the corpus fixed and move only the ONLINE knobs — one variable at a time:

  - top_k   (3 / 5 / 10)          how many chunks the LLM ultimately sees,
  - rerank  (off / on)            add the cross-encoder second stage,
  - top_N   (20 / 50 / 100)       stage-1 candidate pool fed to the reranker.

Because chunking is fixed, we do NOT re-ingest here (unlike ablation_chunking).
Every config queries the SAME index, so any metric delta is attributable to the
retrieval knob — that's what makes it an ablation.

Per config we collect:
  - doc-level recall@1  — did the gold DOC land in position 1? (ordering signal:
    this is what a reranker is supposed to move),
  - doc-level recall@k  — did the gold DOC make it into the k chunks the LLM
    sees? (coverage: W4 showed this is near-saturated, so rerank may not move it),
  - RAGAS context_precision / context_recall on exactly those k contexts,
  - mean RETRIEVAL latency (ms) — rerank and a bigger top_N/k all cost time; the
    quality-vs-latency trade is the whole point of the table,
  - on 3 representative configs only (baseline / +rerank / k=10), we also GENERATE
    answers and score RAGAS faithfulness + answer_relevancy — to see whether a
    retrieval change actually reaches the final answer. Generation costs real LLM
    calls, so we don't run it on all six rows.

Why doc-level recall (not chunk-level): same W4 lesson as the chunking ablation —
chunk ids are unstable, the gold DOC is stable. gold_sources is the ground truth.

Everything is frozen for comparability: same eval set, same judge (temp 0), same
local bge embeddings for answer_relevancy — identical to run_ragas.py so these
numbers sit next to the D3 baseline.

Run:
    uv run python -m eval.ablation_retrieval --smoke   # 3 Q, no gen, no ragas — wiring test
    uv run python -m eval.ablation_retrieval --no-ragas # doc-recall + latency only (fast, free)
    uv run python -m eval.ablation_retrieval           # full grid (6 rows, gen on 3)
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

# --- compat shim (must run BEFORE importing ragas) -------------------------
# Same reason as run_ragas.py / ablation_chunking.py: ragas imports a
# langchain-community Vertex shim that no longer exists in our pinned stack. We
# never use Vertex, so stub it before ragas is imported anywhere.
_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertex)
# ---------------------------------------------------------------------------

import csv  # noqa: E402
import json  # noqa: E402
import statistics  # noqa: E402
from time import perf_counter  # noqa: E402

from dotenv import load_dotenv  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.embeddings import Embedder  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from src.generate import get_client as get_llm_client  # noqa: E402
from src.opensearch_client import get_client  # noqa: E402
from src.reranker import Reranker  # noqa: E402
from src.retrieve import search  # noqa: E402

# Reuse the D3 harness's frozen judge + local-embeddings adapter, so this
# ablation scores with the EXACT same judge model/temperature and the same
# answer_relevancy embeddings as the baseline.
from eval.run_ragas import JUDGE_MODEL, BgeEmbeddings, build_judge  # noqa: E402

load_dotenv()  # generation needs the provider key from .env

EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "ablation_retrieval.csv"

# The retrieval/rerank grid (from the W5 D6 spec). One variable moves per row vs
# the baseline. `gen=True` marks the representative configs we also generate
# answers for (faithfulness/answer_relevancy) — kept to 3 to bound LLM cost.
#   name, k, rerank, top_n (None = n/a for rerank-off), gen
CONFIGS: list[dict] = [
    {"name": "baseline",        "k": 5,  "rerank": False, "top_n": None, "gen": True},
    {"name": "k=3",             "k": 3,  "rerank": False, "top_n": None, "gen": False},
    {"name": "k=10",            "k": 10, "rerank": False, "top_n": None, "gen": True},
    {"name": "+rerank N=50",    "k": 5,  "rerank": True,  "top_n": 50,   "gen": True},
    {"name": "+rerank N=20",    "k": 5,  "rerank": True,  "top_n": 20,   "gen": False},
    {"name": "+rerank N=100",   "k": 5,  "rerank": True,  "top_n": 100,  "gen": False},
]


def load_answerable() -> list[dict]:
    """Load the eval set, keeping only questions that HAVE a gold document.

    The eval set mixes in out-of-scope "no-answer" questions (the system should
    refuse). Those have no gold_sources, so retrieval recall is undefined for
    them. This ablation measures RETRIEVAL, so we score only the answerable
    questions — same choice as the chunking ablation. (Refusal quality is a
    generation concern, measured elsewhere.)
    """
    rows = [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8") if line.strip()]
    return [r for r in rows if r.get("gold_sources")]


def retrieve_pass(rows: list[dict], cfg: dict, embedder, client, reranker):
    """One retrieval pass over the eval set under config `cfg`.

    Returns (recall@1, recall@k, records, mean_latency_ms) where each record is
    {"q", "hits", "reference"} — we keep the hit dicts (not just texts) because
    the gen configs need them to build the LLM context.
    """
    records: list[dict] = []
    ranks: list[int | None] = []
    latencies_ms: list[float] = []

    for r in rows:
        t0 = perf_counter()
        hits = search(
            r["q"],
            k=cfg["k"],
            client=client,
            embedder=embedder,
            use_reranker=cfg["rerank"],
            reranker=reranker,
            rerank_top_n=cfg["top_n"],
        )
        latencies_ms.append((perf_counter() - t0) * 1000)

        # Doc-level rank (1-based) of the first hit whose SOURCE is a gold doc.
        gold_sources = set(r.get("gold_sources", []))
        rank = next((i for i, h in enumerate(hits, 1) if h["source"] in gold_sources), None)
        ranks.append(rank)
        records.append({"q": r["q"], "hits": hits, "reference": r["reference"]})

    n = len(rows)
    # recall@1 = ordering (gold ranked first); recall@k = coverage (gold in the
    # k chunks the LLM sees). search() returns exactly k hits, so recall@k is
    # "gold appeared at all" for this config's k.
    recall_at_1 = sum(1 for rk in ranks if rk and rk <= 1) / n
    recall_at_k = sum(1 for rk in ranks if rk and rk <= cfg["k"]) / n
    return recall_at_1, recall_at_k, records, statistics.mean(latencies_ms)


def build_samples(records: list[dict], *, with_response: bool, llm_client) -> list[dict]:
    """Turn retrieval records into RAGAS samples.

    Always includes (user_input, retrieved_contexts, reference) — enough for the
    context metrics. When `with_response`, also GENERATE an answer per question
    (the expensive part) and attach it as `response`, enabling faithfulness +
    answer_relevancy for that config.
    """
    samples = []
    for rec in records:
        sample = {
            "user_input": rec["q"],
            "retrieved_contexts": [h["text"] for h in rec["hits"]],
            "reference": rec["reference"],
        }
        if with_response:
            sample["response"] = generate_answer(rec["q"], rec["hits"], client=llm_client)
        samples.append(sample)
    return samples


def score_ragas(samples: list[dict], metrics, judge, embeddings) -> dict:
    """Score `samples` with the given RAGAS metrics; return {metric_name: mean}.

    Imported lazily (after the Vertex shim) and reuses the frozen judge +
    local-bge embeddings so every config is scored identically. Same conservative
    concurrency/timeout as run_ragas.py (DeepSeek rate-limits under many small
    judge calls).
    """
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    ds = EvaluationDataset.from_list(samples)
    run_config = RunConfig(timeout=180, max_workers=4)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )
    df = result.to_pandas()
    return {m.name: float(df[m.name].mean()) for m in metrics}


def main() -> None:
    smoke = "--smoke" in sys.argv
    run_ragas = "--no-ragas" not in sys.argv and not smoke

    rows = load_answerable()
    configs = CONFIGS[:1] if smoke else CONFIGS
    if smoke:
        rows = rows[:3]

    mode = "SMOKE" if smoke else "FULL"
    ragas_note = f"+ RAGAS (judge={JUDGE_MODEL})" if run_ragas else "(doc-recall + latency only)"
    print(
        f"retrieval ablation — {mode} | {len(configs)} config(s) x {len(rows)} Q "
        f"| corpus fixed (1200/200) | {ragas_note}\n"
    )

    # Build every heavy object ONCE and reuse across all configs/questions:
    # embedder (query encoding), OpenSearch client, cross-encoder reranker, the
    # frozen judge, the answer_relevancy embeddings, and the generation client.
    embedder = Embedder()
    client = get_client()
    reranker = Reranker()  # loaded once; only used by rerank-on configs
    judge = build_judge() if run_ragas else None
    embeddings = None
    llm_client = None
    if run_ragas:
        # RAGAS wants its own embeddings wrapper; reuse the same local bge model.
        from ragas.embeddings import LangchainEmbeddingsWrapper

        embeddings = LangchainEmbeddingsWrapper(BgeEmbeddings())
        llm_client = get_llm_client()

    # Only import metric objects when scoring (keeps --no-ragas import-light).
    context_metrics = gen_metrics = []
    if run_ragas:
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        context_metrics = [context_precision, context_recall]
        gen_metrics = [faithfulness, answer_relevancy]

    results: list[dict] = []
    for cfg in configs:
        tag = f"{cfg['name']}  (k={cfg['k']}, rerank={cfg['rerank']}, N={cfg['top_n']})"
        print(f"=== {tag} ===")
        recall_at_1, recall_at_k, records, mean_lat = retrieve_pass(
            rows, cfg, embedder, client, reranker
        )

        row = {
            "name": cfg["name"],
            "k": cfg["k"],
            "rerank": cfg["rerank"],
            "top_n": cfg["top_n"] if cfg["top_n"] is not None else "",
            "recall@1": recall_at_1,
            "recall@k": recall_at_k,
            "context_precision": "",
            "context_recall": "",
            "faithfulness": "",
            "answer_relevancy": "",
            "mean_latency_ms": round(mean_lat, 1),
        }

        if run_ragas:
            gen = cfg["gen"]
            print(f"  building samples{' + generating answers' if gen else ''} ...")
            samples = build_samples(records, with_response=gen, llm_client=llm_client)
            metrics = context_metrics + (gen_metrics if gen else [])
            print(f"  scoring RAGAS ({', '.join(m.name for m in metrics)}) ...")
            row.update(score_ragas(samples, metrics, judge, embeddings))

        results.append(row)

        extra = ""
        if run_ragas:
            extra = f"  ctx_prec={row['context_precision']:.3f}  ctx_recall={row['context_recall']:.3f}"
            if cfg["gen"]:
                extra += f"  faith={row['faithfulness']:.3f}  ans_rel={row['answer_relevancy']:.3f}"
        print(f"  r@1={recall_at_1:.3f}  r@k={recall_at_k:.3f}{extra}  lat={mean_lat:.0f}ms\n")

    _report(results, run_ragas, smoke)


def _fmt(v) -> str:
    """Format a cell: 3-dp floats, blanks for missing metrics, str otherwise."""
    if isinstance(v, float):
        return f"{v:>17.3f}"
    return f"{str(v):>17}"


def _report(results: list[dict], run_ragas: bool, smoke: bool) -> None:
    """Print the before/after table and (on a full run) write the CSV."""
    cols = ["name", "k", "rerank", "top_n", "recall@1", "recall@k"]
    if run_ragas:
        cols += ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
    cols += ["mean_latency_ms"]

    print("=== retrieval/rerank ablation table ===")
    print("  ".join(f"{c:>17}" for c in cols))
    for r in results:
        print("  ".join(_fmt(r[c]) for c in cols))

    if not smoke:
        with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            writer.writerows({c: r[c] for c in cols} for r in results)
        print(f"\ntable -> {OUT_CSV}")


if __name__ == "__main__":
    main()
