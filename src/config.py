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
class RerankConfig:
    """Cross-encoder reranker settings.

    A reranker re-scores (query, chunk) PAIRS with full cross-attention — far
    more accurate than the bi-encoder's separate-vector cosine, but it can't be
    pre-indexed, so we only run it on a small candidate set.

    `model_name`: bge-reranker-base is the small/fast cross-encoder (~280 MB).
    `use_fp16`: half precision ~2x faster on a CUDA GPU; ignored off CUDA (CPU
    fp16 is slower, so reranker.py only enables it when a GPU is present).
    """

    model_name: str = "BAAI/bge-reranker-base"
    use_fp16: bool = True


@dataclass(frozen=True)
class RetrievalConfig:
    """Query-time settings.

    Two-stage retrieval: the bi-encoder k-NN casts a wide net (top-N =
    `rerank_top_n`), then the cross-encoder reranks those down to `top_k`.
    `use_reranker` toggles stage 2 so we can run the D4-vs-D5 before/after.
    """

    top_k: int = 5  # how many chunks we ultimately return (recall@k uses k=1/3/5)
    use_reranker: bool = False  # default off = pure bi-encoder (the D4 baseline)
    rerank_top_n: int = 50  # stage-1 candidates fed to the reranker (the "N")


@dataclass(frozen=True)
class LLMConfig:
    """Answer-generation (LLM) settings.

    Provider-agnostic via the OpenAI-compatible chat API. DeepSeek and OpenAI
    both speak this exact protocol, so switching providers is just swapping
    `base_url` + `model` + `api_key_env` — no code change in generate.py.

    Defaults target DeepSeek (the key we have now). To switch to OpenAI, set
    provider="openai", base_url="" (uses the SDK's default endpoint),
    model="gpt-4o-mini", api_key_env="OPENAI_API_KEY". Anthropic uses a
    different SDK; generate.py raises a clear error pointing that out until we
    wire it (the `anthropic` package is already installed for that day).

    `temperature=0` makes generation as deterministic as possible — for a
    grounded "answer only from the docs" assistant we want faithfulness, not
    creativity. `max_tokens` caps answer length (cost + latency).
    """

    provider: str = "deepseek"  # "deepseek" | "openai" | "anthropic"
    model: str = "deepseek-v4-pro"  # DeepSeek V4 (also available: deepseek-v4-flash)
    base_url: str = "https://api.deepseek.com"  # OpenAI-compatible endpoint
    api_key_env: str = "DEEPSEEK_API_KEY"  # which env var holds the key
    # deepseek-v4-pro is a *thinking* model: its hidden reasoning tokens count
    # against max_tokens too. Too small a budget gets eaten by reasoning, leaving
    # an empty/truncated answer (finish_reason="length"), so keep headroom for
    # reasoning + the visible answer.
    max_tokens: int = 2048
    temperature: float = 0.0


EMBED = EmbeddingConfig()
INDEX = IndexConfig()
RERANK = RerankConfig()
RETRIEVAL = RetrievalConfig()
LLM = LLMConfig()
