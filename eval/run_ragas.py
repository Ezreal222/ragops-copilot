"""Run the RAGAS baseline (four metrics) over the frozen eval inputs.

This is the D3 deliverable: turn our RAG system's answers into four numbers.

    faithfulness        — is the answer supported by the retrieved context?
                          (anti-hallucination)         needs: response + contexts
    answer_relevancy    — does the answer actually address the question?
                          (on-topic)                   needs: question + response + embeddings
    context_precision   — is the retrieved context relevant + well-ranked?
                          (retrieval signal/noise)     needs: question + contexts + reference
    context_recall      — did retrieval bring back the info the reference needs?
                          (retrieval coverage)         needs: contexts + reference

All scores are 0..1 (1 best). The ABSOLUTE values matter less than the relative
change across ablations (W5 D5/D6) under one fixed setup — so we freeze the
judge (temperature=0) for reproducibility.

Judge LLM: DeepSeek `deepseek-v4-flash` (cheap/fast — RAGAS calls the LLM once
per sample per LLM-metric, so a thinking model would be slow + costly).
Embeddings: our own local bge model (reused, free), needed by answer_relevancy.

Inputs come from `eval/ragas_inputs.jsonl` (built once by collect_ragas_inputs.py)
so we never re-pay for generation while iterating on metrics.

Run:
    uv run python -m eval.run_ragas --smoke   # 3 samples, to confirm wiring
    uv run python -m eval.run_ragas           # full 36-question baseline
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path

# --- compat shim (must run BEFORE importing ragas) -------------------------
# ragas unconditionally does `from langchain_community.chat_models.vertexai
# import ChatVertexAI`, but that shim was deleted in langchain-community 0.4.x
# (and our project pins langchain 1.x). We never use Vertex AI, so inject a
# stub module to satisfy the import instead of downgrading the whole stack.
_vertex = types.ModuleType("langchain_community.chat_models.vertexai")
_vertex.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules.setdefault("langchain_community.chat_models.vertexai", _vertex)
# ---------------------------------------------------------------------------

import json  # noqa: E402

from dotenv import load_dotenv  # noqa: E402
from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from ragas import EvaluationDataset, evaluate  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import LLM  # noqa: E402
from src.embeddings import Embedder  # noqa: E402

load_dotenv()

INPUTS_PATH = REPO_ROOT / "eval" / "ragas_inputs.jsonl"
OUT_CSV = REPO_ROOT / "eval" / "ragas_baseline.csv"

# Cheap/fast judge — NOT the thinking model used for generation. Same DeepSeek
# endpoint + key as generation (config.LLM), only the model id differs.
JUDGE_MODEL = "deepseek-v4-flash"

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# RAGAS needs these 4 keys per sample; bookkeeping fields are ignored.
RAGAS_KEYS = ("user_input", "response", "retrieved_contexts", "reference")


class BgeEmbeddings(Embeddings):
    """Adapt our local bge Embedder to the langchain Embeddings interface.

    Reuses the exact model the retriever uses (no extra download, no API cost).
    answer_relevancy embeds generated questions vs the original to score topicality.
    """

    def __init__(self) -> None:
        self._e = Embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._e.encode_passages(texts, show_progress=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._e.encode_query(text)


def load_samples(limit: int | None = None) -> list[dict]:
    if not INPUTS_PATH.exists():
        raise FileNotFoundError(
            f"{INPUTS_PATH} not found — run `uv run python -m eval.collect_ragas_inputs` first."
        )
    rows = [json.loads(line) for line in INPUTS_PATH.open(encoding="utf-8") if line.strip()]
    if limit:
        rows = rows[:limit]
    # Keep only the keys RAGAS consumes.
    return [{k: r[k] for k in RAGAS_KEYS} for r in rows]


def build_judge() -> LangchainLLMWrapper:
    api_key = os.environ.get(LLM.api_key_env)
    if not api_key:
        raise RuntimeError(f"{LLM.api_key_env} not set — fill it in .env.")
    chat = ChatOpenAI(
        model=JUDGE_MODEL,
        base_url=LLM.base_url,
        api_key=api_key,
        temperature=0,  # reproducible scoring
    )
    return LangchainLLMWrapper(chat)


def main() -> None:
    smoke = "--smoke" in sys.argv
    samples = load_samples(limit=3 if smoke else None)
    mode = "SMOKE (3 samples)" if smoke else f"FULL ({len(samples)} samples)"
    print(f"RAGAS baseline — {mode} | judge={JUDGE_MODEL} | embeddings=bge (local)\n")

    ds = EvaluationDataset.from_list(samples)
    judge = build_judge()
    embeddings = LangchainEmbeddingsWrapper(BgeEmbeddings())

    # Conservative concurrency + generous timeout: DeepSeek rate limits, and the
    # judge does many small calls. max_workers low avoids 429s.
    run_config = RunConfig(timeout=180, max_workers=4)

    result = evaluate(
        ds,
        metrics=METRICS,
        llm=judge,
        embeddings=embeddings,
        run_config=run_config,
    )

    print("\n=== RAGAS baseline (mean over samples) ===")
    print(result)

    df = result.to_pandas()
    if not smoke:
        df.to_csv(OUT_CSV, index=False)
        print(f"\nper-question detail -> {OUT_CSV}")

    # Failure attribution: for each metric, show the lowest-scoring questions so
    # we can read off WHERE the system is weak (retrieval vs generation).
    metric_cols = [c for c in df.columns if c in {m.name for m in METRICS}]
    print("\n=== lowest-scoring questions per metric ===")
    for col in metric_cols:
        worst = df.nsmallest(3, col)[["user_input", col]]
        print(f"\n[{col}]")
        for _, row in worst.iterrows():
            print(f"  {row[col]:.3f}  {row['user_input'][:70]}")


if __name__ == "__main__":
    main()
