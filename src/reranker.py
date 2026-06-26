"""cross-encoder reranking with a sentence-transformers CrossEncoder.

Why this exists: the bi-encoder (src/embeddings.py) encodes the query and each
chunk SEPARATELY into vectors, then compares them by cosine. Fast and
pre-indexable, but the two never interact before scoring, so fine-grained
relevance gets blurred. A cross-encoder instead feeds the (query, chunk) pair
through the model TOGETHER with full attention and emits one relevance score —
much sharper, but it can't be pre-indexed (the score depends on the query), so
it only runs on a small candidate set the bi-encoder already narrowed down.

We load `BAAI/bge-reranker-base` via sentence-transformers' `CrossEncoder`
(robust + already a project dep) rather than FlagEmbedding, which had a
tokenizer incompatibility with the installed transformers. We wrap it in a
load-once class mirroring Embedder so callers (retrieval, eval) build it a
single time and reuse it across many queries.
"""

from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

from src.config import RERANK, RerankConfig
from src.embeddings import pick_device


class Reranker:
    """Loads bge-reranker once; scores/sorts (query, chunk) candidates."""

    def __init__(self, config: RerankConfig = RERANK) -> None:
        self.config = config
        self.device = pick_device()
        # fp16 only helps on a GPU; keep full precision on CPU/MPS.
        use_fp16 = config.use_fp16 and self.device == "cuda"
        model_kwargs = {"torch_dtype": torch.float16} if use_fp16 else None
        # First run downloads the model from HuggingFace (~280 MB), then caches.
        self.model = CrossEncoder(
            config.model_name, device=self.device, model_kwargs=model_kwargs
        )

    def score(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score for each (query, text) pair. Higher = more relevant.

        Only the ORDER of the scores matters for ranking.
        """
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        # predict() batches internally and returns a numpy array of scores.
        return [float(s) for s in self.model.predict(pairs)]

    def rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        """Re-score `candidates` by their "text", sort desc, return the top `k`.

        Each returned candidate gets a "rerank_score" field added so downstream
        code (and eval failure analysis) can see why the order changed.
        """
        scores = self.score(query, [c["text"] for c in candidates])
        for cand, s in zip(candidates, scores):
            cand["rerank_score"] = s
        ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
        return ranked[:k]


if __name__ == "__main__":
    #   uv run python -m src.reranker
    # Smoke test: the reranker should rank the obviously-relevant text first.
    rr = Reranker()
    print(f"model={rr.config.model_name} device={rr.device}")
    q = "What is PagedAttention?"
    docs = [
        "PagedAttention manages the KV cache in non-contiguous blocks like OS paging.",
        "Bananas are a good source of potassium.",
        "vLLM also supports speculative decoding for faster generation.",
    ]
    for s, d in sorted(zip(rr.score(q, docs), docs), key=lambda x: -x[0]):
        print(f"  {s:+.5f}  {d}")
