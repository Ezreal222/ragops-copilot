"""Central, tweakable knobs for the RAG pipeline.

"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ChunkConfig:
    """Text-splitting parameters.

    Units are CHARACTERS (RecursiveCharacterTextSplitter measures length with
    `len` by default). Embeddings ultimately count tokens, but for English prose
    ~1 token ≈ 4 chars, so 1200 chars ≈ 300 tokens.

    Values: 1200 / 200 (~17% overlap). These come from the W5 D5 chunking
    ablation (eval/ablation_chunking.py): sweeping size/overlap against the eval
    set, 1200/200 gave the best context_recall (0.69→0.79) AND context_precision
    (0.69→0.76) vs the old 800/100 baseline — larger chunks carry enough
    surrounding context to answer without losing retrieval precision. The overlap
    keeps an answer that straddles a chunk boundary from being lost.
    """

    chunk_size: int = 1200
    chunk_overlap: int = 200


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
    `use_reranker` toggles stage 2.

    Values come from the W5 D6 retrieval/rerank ablation
    (eval/ablation_retrieval.py), run on the D5 best chunking (1200/200):
      - `use_reranker=False`: the reranker LOST on every metric — recall@1
        0.59->0.44, context_precision 0.78->0.70, context_recall 0.79->0.70,
        faithfulness 0.84->0.79 — at 5-8x the latency (19ms->94/146ms). On this
        near-saturated doc corpus its reordering hurts more than it helps
        (a stronger repeat of the W4 finding). A clean negative result: off.
      - `top_k=5`: the balanced default. Best context_precision (0.777) at half
        the token cost of k=10. k=10 measurably lifts coverage (recall@k
        0.844->0.969) and end-to-end faithfulness/answer_relevancy, but at ~2x
        context tokens and lower precision — kept as a documented "recall-max"
        option, not the default (n=32, so the quality gains sit within noise).
    """

    top_k: int = 5  # chunks fed to the LLM; 10 = recall-max at ~2x token cost
    use_reranker: bool = False  # D6 ablation: reranker worse on every metric here
    rerank_top_n: int = 50  # stage-1 candidates if reranking is re-enabled (the "N")


@dataclass(frozen=True)
class AgentConfig:
    """Agent guardrail knobs (W6 D4).

    The agent hands control to a free-running LLM, so it needs *deterministic*
    safety edges around that freedom. Each field is one guardrail from the D4
    spec; they're gathered here (not scattered as literals in graph.py/tools.py)
    for the same reason the RAG knobs are — one place to see and tune the
    reliability posture.

    - `recursion_limit`: max agent<->tools super-steps per question, passed as
      LangGraph's `recursion_limit` (its own default is 25). Bounds a
      non-converging ReAct loop — the D2 "LLM called search_docs 5x" motivation.
      Hitting it is handled gracefully (a fallback answer), not by crashing.

    - `tool_max_retries` / `tool_retry_backoff_s`: guardrail 2. On a *transient*
      tool failure (OpenSearch unreachable / timeout) a tool retries this many
      extra times with linear backoff before giving up; deterministic errors
      (e.g. a malformed query) are NOT retried. When retries are exhausted the
      tool still doesn't raise — it returns a readable error the LLM can act on.

    - `min_relevance_score`: guardrail 3. Pure k-NN always returns top-k, so an
      off-topic question still gets chunks back; if the BEST chunk scores below
      this, the tool treats retrieval as "nothing relevant" and refuses instead
      of grounding on noise. Assumes the bi-encoder cosine score (reranker off,
      the prod default). Basis — eval/analyze_score_threshold.py over the 36-Q
      eval set: on-topic top-1 min 0.834 / median 0.917; off-topic top-1 median
      0.793 / max 0.854. 0.82 sits just under the on-topic minimum (0 false
      refusals on the eval set) and above the off-topic median. It favors recall
      (a false refusal is worse UX, and the system-prompt refusal + citation
      guardrail catch residual noise), so the lone off-topic outlier at 0.854
      that clears it is acceptable. OpenSearch lucene cosinesimil scores live in
      0.5 (orthogonal) .. 1.0 (identical).
    """

    recursion_limit: int = 8
    tool_max_retries: int = 2  # extra attempts after the first, on transient errors
    tool_retry_backoff_s: float = 0.3  # linear backoff: wait attempt*this between tries
    min_relevance_score: float = 0.82  # below this top-1 score -> refuse (eval-derived)


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


@dataclass(frozen=True)
class ServingConfig:
    """System-level serving knobs (W6 D6) — how the unified entrypoint answers.

    `use_agent` is the single routing switch for src.answer.answer():
      - True : route every question through the agent (run_agent) — the LLM decides
        which tool(s) to call, so simple lookups and multi-step compares share one
        path. This is the W6 architectural direction: an agentic RAG system.
      - False (default): use the fixed retrieve->rerank->generate pipeline (ask()).

    Keeping it a config flag (not a code fork) is the production **degrade switch**:
    the agent path exists and is the strategic entrypoint, but the RUNTIME default
    stays on the stable fixed pipeline — the basics of progressive rollout.

    Why default False (not True as D6 step 1 first set it): the D6 regression
    (eval/compare_agent_vs_rag.py, 36-Q) measured the agent NOT at parity with
    fixed RAG — context_precision 0.78->0.53 and 8/36 questions still hit the
    step-cap fallback (non-convergence), even after the D6 tool-use-policy fix cut
    avg tool calls 8.0->2.8. faithfulness/answer_relevancy/refusal_acc came within
    ~0.03-0.06 (noise), but 22% degraded answers is a real UX regression. So we do
    NOT flip the default to the agent yet: ship fixed RAG, keep the agent behind
    the flag until convergence improves. "Changing the architecture needs a
    regression test, and if it degrades you don't flip the default" — the lesson.
    """

    use_agent: bool = False


EMBED = EmbeddingConfig()
INDEX = IndexConfig()
RERANK = RerankConfig()
RETRIEVAL = RetrievalConfig()
AGENT = AgentConfig()
LLM = LLMConfig()
SERVING = ServingConfig()
