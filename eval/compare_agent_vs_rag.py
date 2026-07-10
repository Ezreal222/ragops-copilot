"""Regression: agent entrypoint vs fixed RAG on the same eval set (W6 D6, step 2).

D6 makes the agent the system's default entrypoint (src/answer.py). Before we
trust that default we must prove it does NOT degrade quality vs the fixed
retrieve->rerank->generate pipeline it replaces. This is "changing the
architecture needs a regression test" applied to RAG: run BOTH engines over the
SAME 36-question eval set and compare, per engine:

  - the four RAGAS metrics (faithfulness / answer_relevancy / context_precision /
    context_recall) — same judge + wiring as eval/run_ragas.py, and
  - refusal accuracy — does the engine refuse exactly the no-answer questions?

Expectation (see W6_D6 §③): on simple questions the agent's default move is a
single search_docs call, so it should be ~equivalent to the fixed pipeline; the
agent version's metrics should be >= (or within noise of) fixed RAG. A metric
that DROPS is a finding to attribute (agent over-searching adds context noise? a
different prompt path?), not something to wave through.

Why regenerate both sides (not reuse eval/ragas_inputs.jsonl)? A fair head-to-head
needs both engines on the CURRENT index + config. The frozen baseline predates
the W5 chunking/rerank changes (it refuses "continuous batching", which is
answerable) — reusing it would compare a fresh agent against a stale RAG.

Two phases, so we don't re-pay generation while iterating on the table:
  collect  — run answer(use_agent=T/F) over the set, freeze four-tuples + booking
             to eval/compare_inputs.jsonl (the expensive LLM step),
  score    — load the frozen file, run RAGAS + refusal accuracy, print the Δ
             table, write eval/compare_agent_vs_rag.csv.

Run:
    uv run python -m eval.compare_agent_vs_rag --smoke 3   # 3 Qs, confirm wiring
    uv run python -m eval.compare_agent_vs_rag             # full 36-Q regression
    uv run python -m eval.compare_agent_vs_rag --score-only  # re-score frozen inputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing run_ragas runs its langchain-community vertex shim BEFORE ragas loads
# (must happen first) and gives us the exact same judge + metrics + embeddings the
# baseline uses — so this comparison is scored on identical footing. main() there
# is __main__-guarded, so nothing executes on import.
from eval.run_ragas import (  # noqa: E402
    METRICS,
    RAGAS_KEYS,
    BgeEmbeddings,
    build_judge,
)
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.run_config import RunConfig  # noqa: E402

from src.agent.graph import build_graph  # noqa: E402
from src.answer import answer  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.generate import get_client  # noqa: E402
from src.opensearch_client import get_client as get_os_client  # noqa: E402
from src.reranker import Reranker  # noqa: E402

EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"
INPUTS_PATH = REPO_ROOT / "eval" / "compare_inputs.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "compare_agent_vs_rag.csv"

ENGINES = ("rag", "agent")  # use_agent = False / True


def load_eval_set() -> list[dict]:
    if not EVAL_PATH.exists():
        raise FileNotFoundError(f"{EVAL_PATH} not found — build the eval set first.")
    return [json.loads(line) for line in EVAL_PATH.open(encoding="utf-8") if line.strip()]


def _declined(response: str, degraded: bool) -> bool:
    """Did the engine decline to give a real cited answer?

    Two ways an engine says "I'm not answering this": the designed refusal
    ("I couldn't find this in the vLLM docs.") — same string collect_ragas_inputs
    keys on — OR the agent's step-cap fallback (degraded=True), which is also a
    non-answer. We fold both into one signal so refusal accuracy penalises an
    agent that BAILS on an answerable question (the PagedAttention degrade we saw),
    not just explicit refusals.
    """
    return "couldn't find this" in response.lower() or degraded


def collect_engine(rows: list[dict], *, use_agent: bool, app, rag_kwargs: dict) -> list[dict]:
    """Run one engine over every question; return RAGAS four-tuples + bookkeeping."""
    mode = "agent" if use_agent else "rag"
    samples = []
    n = len(rows)
    for i, r in enumerate(rows, 1):
        q = r["q"]
        is_no_answer = not r.get("gold_sources") and not r.get("gold_chunk_ids")
        t0 = perf_counter()
        # The whole point of D6: one call, one shape, whichever engine. Agent path
        # reuses the pre-compiled graph; RAG path reuses the heavy retrieval objects.
        res = answer(q, use_agent=use_agent, app=app, **({} if use_agent else rag_kwargs))
        dt = perf_counter() - t0

        samples.append(
            {
                "mode": mode,
                # --- the four fields RAGAS consumes ---
                "user_input": q,
                "response": res["answer"],
                "retrieved_contexts": res["contexts"],
                "reference": r["reference"],
                # --- bookkeeping (RAGAS ignores; we use it for refusal acc + attribution) ---
                "is_no_answer": is_no_answer,
                "refused": _declined(res["answer"], res["degraded"]),
                "steps": res["steps"],
                "degraded": res["degraded"],
                "latency_s": round(dt, 2),
            }
        )
        flag = "  [DEGRADED]" if res["degraded"] else ""
        preview = res["answer"].replace("\n", " ")[:56]
        print(f"  [{mode:>5} {i:>2}/{n}] {dt:5.1f}s steps={res['steps']}  "
              f"{q[:40]:<40} -> {preview}{flag}")
    return samples


def collect_all(rows: list[dict]) -> list[dict]:
    """Generate for BOTH engines, building each engine's heavy objects once."""
    # Fixed-RAG heavy objects (reused across its 36 questions), same as
    # collect_ragas_inputs.py. Agent tools hold their OWN cached embedder/client,
    # so the agent path only needs the compiled graph.
    rag_kwargs = {
        "os_client": get_os_client(),
        "embedder": Embedder(),
        "reranker": Reranker(),
        "llm_client": get_client(),
    }
    app = build_graph()  # compile the agent graph once

    print(f"\n=== collecting fixed RAG (use_agent=False) over {len(rows)} questions ===")
    rag = collect_engine(rows, use_agent=False, app=app, rag_kwargs=rag_kwargs)
    print(f"\n=== collecting agent (use_agent=True) over {len(rows)} questions ===")
    agent = collect_engine(rows, use_agent=True, app=app, rag_kwargs=rag_kwargs)
    return rag + agent


def freeze(samples: list[dict]) -> None:
    with INPUTS_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\nfroze {len(samples)} samples ({len(ENGINES)} engines) -> {INPUTS_PATH}")


def load_frozen() -> list[dict]:
    if not INPUTS_PATH.exists():
        raise FileNotFoundError(
            f"{INPUTS_PATH} not found — run without --score-only first to generate it."
        )
    return [json.loads(line) for line in INPUTS_PATH.open(encoding="utf-8") if line.strip()]


def refusal_accuracy(samples: list[dict]) -> float:
    """Fraction of questions where the engine's refuse/answer choice was correct.

    Correct = it declined exactly the no-answer questions and answered the rest:
    `refused == is_no_answer`. This is the anti-hallucination score — a wrong
    refusal (bailing on an answerable Q) AND a wrong answer (answering an
    out-of-docs Q) both count against it.
    """
    if not samples:
        return 0.0
    correct = sum(1 for s in samples if s["refused"] == s["is_no_answer"])
    return correct / len(samples)


def score_ragas(samples: list[dict], judge, embeddings) -> dict:
    """Run RAGAS over one engine's samples; return {metric_name: mean_score}."""
    ds = EvaluationDataset.from_list([{k: s[k] for k in RAGAS_KEYS} for s in samples])
    # Same conservative concurrency as run_ragas — DeepSeek rate-limits.
    result = evaluate(
        ds, metrics=METRICS, llm=judge, embeddings=embeddings,
        run_config=RunConfig(timeout=180, max_workers=4),
    )
    df = result.to_pandas()
    return {m.name: float(df[m.name].mean()) for m in METRICS if m.name in df.columns}


def report(samples: list[dict]) -> None:
    """Score both engines and print the RAG-vs-agent Δ table (+ write CSV)."""
    by_mode = {mode: [s for s in samples if s["mode"] == mode] for mode in ENGINES}

    # Build the judge + embeddings ONCE, reuse for both engines' scoring.
    judge = build_judge()
    embeddings = LangchainEmbeddingsWrapper(BgeEmbeddings())

    scores: dict[str, dict] = {}
    for mode in ENGINES:
        rows = by_mode[mode]
        if not rows:
            continue
        print(f"\n=== scoring RAGAS: {mode} ({len(rows)} samples) ===")
        s = score_ragas(rows, judge, embeddings)
        s["refusal_acc"] = refusal_accuracy(rows)
        s["degraded_n"] = sum(1 for r in rows if r["degraded"])
        s["avg_steps"] = sum(r["steps"] for r in rows) / len(rows)
        scores[mode] = s

    # --- the comparison table: RAG vs agent, with Δ = agent - rag ---
    metric_order = [m.name for m in METRICS] + ["refusal_acc"]
    rag_s, agent_s = scores.get("rag", {}), scores.get("agent", {})
    print("\n" + "=" * 66)
    print(f"{'metric':<22}{'RAG':>10}{'Agent':>10}{'Δ (agent-rag)':>18}")
    print("-" * 66)
    csv_lines = ["metric,rag,agent,delta"]
    for m in metric_order:
        rv, av = rag_s.get(m), agent_s.get(m)
        if rv is None or av is None:
            continue
        d = av - rv
        flag = "  <- DROP" if d < -0.02 else ""
        print(f"{m:<22}{rv:>10.3f}{av:>10.3f}{d:>+18.3f}{flag}")
        csv_lines.append(f"{m},{rv:.4f},{av:.4f},{d:+.4f}")
    print("=" * 66)
    # Health context: how much extra work did the agent do?
    print(f"agent avg tool-call steps: {agent_s.get('avg_steps', 0):.2f}  "
          f"(rag is 1 by construction)")
    print(f"agent degraded (step-cap fallback): {agent_s.get('degraded_n', 0)}/"
          f"{len(by_mode['agent'])}")

    OUT_CSV.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
    print(f"\ncomparison table -> {OUT_CSV}")


def main() -> None:
    args = sys.argv[1:]
    score_only = "--score-only" in args
    smoke_n = None
    if "--smoke" in args:
        idx = args.index("--smoke")
        smoke_n = int(args[idx + 1]) if idx + 1 < len(args) else 3

    if score_only:
        print("--score-only: re-scoring frozen inputs (no generation).")
        samples = load_frozen()
    else:
        rows = load_eval_set()
        if smoke_n:
            rows = rows[:smoke_n]
            print(f"--smoke: first {smoke_n} questions only.")
        samples = collect_all(rows)
        freeze(samples)

    report(samples)


if __name__ == "__main__":
    main()
