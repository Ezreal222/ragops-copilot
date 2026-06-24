"""Step (D3) — turn text into vectors with sentence-transformers.

What is an embedding? A function that maps text into a fixed-size vector so that
*semantically similar text lands close together* in that space. Retrieval then
becomes geometry: embed the question, find the chunk vectors nearest to it.

We wrap the model in a tiny class so the rest of the pipeline doesn't care about
device selection or bge's query/passage conventions:
  - encode_passages(): for the corpus chunks (no instruction prefix).
  - encode_query():    for a user question (optional bge query prefix).

Device is auto-detected (CLAUDE.md: don't hardcode CUDA) — CUDA on the 5080,
MPS on Mac, else CPU. Vectors are L2-normalized so cosine similarity in the
index behaves correctly.
"""

from __future__ import annotations

import torch
from sentence_transformers import SentenceTransformer

from src.config import EMBED, EmbeddingConfig


def pick_device() -> str:
    """Best available device, without hardcoding CUDA."""
    if torch.cuda.is_available():
        return "cuda"
    # Apple Silicon (Yang also runs on Mac).
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Embedder:
    """Loads the embedding model once and exposes passage/query encoders."""

    def __init__(
        self,
        config: EmbeddingConfig = EMBED,
        device: str | None = None,
    ) -> None:
        self.config = config
        self.device = device or pick_device()
        # First run downloads the model from HuggingFace (~130 MB), then caches.
        self.model = SentenceTransformer(config.model_name, device=self.device)

    def encode_passages(self, texts: list[str], show_progress: bool = True):
        """Embed corpus chunks. Returns an (N, dim) float32 numpy array.

        Batched on the GPU for throughput (one big matmul beats N tiny ones).
        """
        return self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    def encode_query(self, query: str) -> list[float]:
        """Embed one question -> a plain python list (ready for the k-NN body).

        Applies the (optional) bge query prefix; passages never get it.
        """
        text = self.config.query_prefix + query
        vec = self.model.encode(
            [text],
            normalize_embeddings=self.config.normalize,
            convert_to_numpy=True,
        )[0]
        return vec.tolist()


if __name__ == "__main__":
    #   uv run python -m src.embeddings
    emb = Embedder()
    print(f"model={emb.config.model_name} device={emb.device}")
    vecs = emb.encode_passages(["hello world", "vLLM serves LLMs fast"], show_progress=False)
    print(f"passage batch shape: {vecs.shape} (expect (2, {emb.config.dimension}))")
    qv = emb.encode_query("How does vLLM batch requests?")
    print(f"query vector len: {len(qv)} | first 3 dims: {[round(x, 3) for x in qv[:3]]}")
