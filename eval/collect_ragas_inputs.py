"""Collect RAGAS inputs by running the RAG pipeline once over the eval set.

RAGAS grades each question from a four-tuple:
    user_input          = the question
    response            = what OUR system answered
    retrieved_contexts  = the chunk TEXTS we fed the LLM as context (not ids)
    reference           = the hand-written gold answer (from eval_set.jsonl)

Generation is the expensive part (one LLM call per question, on a thinking
model), and we'll iterate on the RAGAS metric wiring many times. So we run the
pipeline ONCE here, freeze the four-tuples to `eval/ragas_inputs.jsonl`, and let
`run_ragas.py` read that file repeatedly without re-calling the LLM.

This is the system-under-test snapshot: `ask()` is the real production path
(bi-encoder retrieve -> cross-encoder rerank -> grounded, cited generation), so
the RAGAS baseline reflects what a user would actually get.

We also keep some non-RAGAS bookkeeping per row (gold ids, retrieved ids,
citations) so a low score can be traced to "retrieval missed it" vs "the answer
drifted from context" later.

Run with:  uv run python -m eval.collect_ragas_inputs
Heavy objects (OpenSearch client, embedder, reranker, LLM client) are built once
and reused across all questions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from time import perf_counter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import RETRIEVAL  # noqa: E402
from src.embeddings import Embedder  # noqa: E402
from src.generate import ask, get_client  # noqa: E402
from src.opensearch_client import get_client as get_os_client  # noqa: E402
from src.reranker import Reranker  # noqa: E402

EVAL_PATH = REPO_ROOT / "eval" / "eval_set.jsonl"
OUT_PATH = REPO_ROOT / "eval" / "ragas_inputs.jsonl"


def load_eval_set(path: Path = EVAL_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — build the eval set first.")
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def collect(rows: list[dict]) -> list[dict]:
    """Run ask() for each question and assemble the RAGAS four-tuples."""
    # Build the heavy objects once — reused for every question. ask() always
    # reranks (it's the production answer path), so this mirrors what a user gets.
    os_client = get_os_client()
    embedder = Embedder()
    reranker = Reranker()
    llm_client = get_client()

    samples = []
    n = len(rows)
    for i, r in enumerate(rows, 1):
        q = r["q"]
        t0 = perf_counter()
        result = ask(
            q,
            k=RETRIEVAL.top_k,
            os_client=os_client,
            embedder=embedder,
            reranker=reranker,
            llm_client=llm_client,
        )
        dt = perf_counter() - t0

        chunks = result["retrieved_chunks"]
        samples.append(
            {
                # --- the four fields RAGAS consumes ---
                "user_input": q,
                "response": result["answer"],
                "retrieved_contexts": [c["text"] for c in chunks],
                "reference": r["reference"],
                # --- bookkeeping for failure attribution (RAGAS ignores these) ---
                "gold_chunk_ids": r.get("gold_chunk_ids", []),
                "gold_sources": r.get("gold_sources", []),
                "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
                "citations": result["citations"],
                "latency_s": round(dt, 2),
            }
        )
        # Progress: a short preview so you can eyeball that answers look sane and
        # that no-answer questions are being refused, not fabricated.
        preview = result["answer"].replace("\n", " ")[:70]
        print(f"  [{i:>2}/{n}]  {dt:5.1f}s  {q[:48]:<48}  -> {preview}")

    return samples


def main() -> None:
    rows = load_eval_set()
    print(f"running ask() over {len(rows)} questions (this calls the LLM once each)...\n")
    samples = collect(rows)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    refused = sum(1 for s in samples if "couldn't find this" in s["response"].lower())
    print(f"\nwrote {len(samples)} samples to {OUT_PATH}")
    print(f"  refused ('not in docs'): {refused}  (expect ~the 4 no-answer questions)")


if __name__ == "__main__":
    main()
