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


@dataclass(frozen=True)
class EmbeddingConfig:
    """Sentence-embedding parameters.

    `dimension` MUST equal the model's output size AND the `knn_vector`
    dimension in the index mapping — a mismatch makes OpenSearch reject writes.
    bge-small-en-v1.5 outputs 384-dim vectors.

    `normalize` makes every vector unit-length so an inner-product / cosine
    space measures pure *direction* (semantic similarity), not magnitude — this
    is what `space_type=cosinesimil` in the index expects.

    bge models recommend prefixing the QUERY (not the passages) with a short
    instruction for retrieval. v1.5 mostly works without it, so we start empty
    and keep it as a later tuning knob.
    """

    model_name: str = "BAAI/bge-small-en-v1.5"
    dimension: int = 384
    batch_size: int = 64  # GPU batch — high throughput; drop on small VRAM
    normalize: bool = True
    query_prefix: str = ""  # e.g. "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class IndexConfig:
    """OpenSearch connection + k-NN index settings.

    `space_type=cosinesimil` → cosine similarity (direction), the usual choice
    for normalized text embeddings. `engine=lucene` is the simplest k-NN engine
    (ships with OpenSearch, no native libs). `method=hnsw` builds a graph for
    approximate nearest-neighbor search — sub-linear query time vs brute force.
    `ef_construction`/`m` are HNSW build-quality knobs (higher = better recall,
    slower build / more memory); the defaults are fine for ~2.7k vectors.
    """

    host: str = "http://localhost:9200"
    index_name: str = "vllm_docs"
    space_type: str = "cosinesimil"
    engine: str = "lucene"
    ef_construction: int = 128
    m: int = 16


@dataclass(frozen=True)
class RetrievalConfig:
    """Query-time settings."""

    top_k: int = 5  # how many chunks k-NN returns (recall@k uses k=1/3/5)


EMBED = EmbeddingConfig()
INDEX = IndexConfig()
RETRIEVAL = RetrievalConfig()
