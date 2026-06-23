"""Central, tweakable knobs for the RAG pipeline.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """Text-splitting parameters.

    Units are CHARACTERS (RecursiveCharacterTextSplitter measures length with
    `len` by default). Embeddings ultimately count tokens, but for English prose
    ~1 token ≈ 4 chars, so 800 chars ≈ 200 tokens — a sane starting point for
    technical docs.

    Starting values: 800 / 100  (~12% overlap). The overlap
    keeps an answer that straddles a chunk boundary from being lost.
    """

    chunk_size: int = 800
    chunk_overlap: int = 100


# The single instance the pipeline imports. Change values here (or override at
# call sites) to run an ablation.
CHUNK = ChunkConfig()
