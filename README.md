# RAGOps Copilot

> A production-grade RAG (and Agent) assistant over the **vLLM documentation**.
> Ask a question → retrieve relevant doc chunks → (rerank) → LLM answers **with citations** →
> retrieval quality is **measured** (recall@k, then RAGAS). The focus is production engineering —
> evaluation, serving, monitoring — **not** a notebook chatbot.

**Why vLLM docs?** The corpus pairs with a companion project on vLLM / LLM-serving optimization:
this assistant answers questions *about* vLLM, while the serving work optimizes the infrastructure
that *runs* vLLM — using this RAG app as a realistic workload. Together they cover both building an
LLM application and making it efficient.

## Architecture

![Architecture](docs/architecture.svg)

- **① Offline ingestion** — vLLM docs → chunk → embed → index in OpenSearch.
- **② Online query** — question → semantic retrieval → (rerank) → LLM generation with citations.
- **③ Eval harness** (side) and **④ Serving / MLOps** (outer) are subsequent phases.

## Design decisions

| Decision | Choice | Why |
|---|---|---|
| Corpus | **vLLM official docs** (cloned from the GitHub `docs/` source) | Clean Markdown/RST, pinnable to a git SHA, idempotent re-ingestion vs. crawling HTML |
| Embeddings | **`BAAI/bge-small-en-v1.5`** (sentence-transformers) | Runs locally on CPU/GPU, reproducible, strong on English technical docs |
| Vector store | **OpenSearch** (local Docker) | Mature hybrid lexical + vector search; managed deployment is a later, serving-phase concern |
| LLM (generation) | **DeepSeek `deepseek-v4-pro`** (OpenAI-compatible) | Citation-faithful answering; provider-agnostic via `LLMConfig` — swap to OpenAI/Anthropic by config, not code |
| Retrieval metric | **recall@1 / recall@3 / recall@5** | Core measure of retrieval quality, reproducible from a fixed eval set |

## Tech stack

LangChain (splitters/retriever) · OpenSearch (vector index) · sentence-transformers (bge-small) ·
bge-reranker (cross-encoder) · DeepSeek (generation, OpenAI-compatible) · RAGAS · LangGraph · FastAPI · Docker · AWS.

## Repo layout

```
src/         ingestion (load/clean/chunk/embed) + retrieval + generation
eval/        eval set + metric scripts (recall@k)
data/        corpus + index data — GITIGNORED
docs/        architecture diagram, design notes
notebooks/   exploration
```

## Roadmap

| Phase | Milestone |
|---|---|
| **1 · Retrieval** ✅ | Ingest → chunk → embed → OpenSearch → semantic retrieval → **recall@1/3/5** + cross-encoder reranker + end-to-end LLM answers with citations. **Done (W4)** — see [Results](#results-w4-retrieval-baseline). |
| **2 · Evaluation** | RAGAS (faithfulness / answer relevancy / context precision-recall) + LLM-as-judge. |
| **3 · Agent** | LangGraph planning + tool loop + guardrails. |
| **4 · Serving & MLOps** | FastAPI · Docker · AWS · monitoring (latency / cost / failure rate) · CI/CD. |

## Results (W4 retrieval baseline)

Measured on a **fixed eval set of 20 hand-written questions** (gold `chunk_id`s labeled
independently of the retriever, via keyword search in `eval/find_chunks.py`) over a corpus of
~2,700 chunks. `chunk-level` recall requires the exact gold chunk; `doc-level` counts any chunk
from the same source document (`chunk_id.split("::")[0]`).

| recall@k | baseline (bi-encoder) chunk / doc | + reranker chunk / doc |
|---|---|---|
| @1 | 0.20 / 0.55 | **0.35** / 0.55 |
| @3 | 0.60 / 0.85 | 0.60 / **0.90** |
| @5 | 0.80 / **1.00** | 0.65 / 0.95 |
| @10 | 0.80 / **1.00** | 0.75 / 0.95 |
| latency / query | mean 38 ms / median 8 ms | mean 95 ms / median 65 ms |

**What the numbers say:**
- Baseline **recall@1 low (0.20) but recall@5 high (0.80)** → relevant content *is* retrieved,
  just not ranked first — exactly the gap a reranker exists to close.
- The cross-encoder reranker lifts **chunk recall@1 0.20 → 0.35** (+75% relative): its real payoff
  here is **ranking precision**, not coverage.
- Document-level recall is already **saturated (recall@5 = 1.00)** for the bi-encoder, so the
  reranker has no recall headroom — it can only reshuffle, occasionally pushing a gold doc out
  (doc@5 1.00 → 0.95). A textbook case of *"a reranker isn't worth it when recall is already
  saturated"*, at ~2.5× the latency.
- The chunk-vs-doc gap is largely a **labeling artifact** (gold pinned to overview `::0` chunks);
  W5 will move to multi-gold / document-level recall on a larger, non-saturated corpus.

**End-to-end (generation).** `ask()` runs retrieve (top-50) → rerank (top-k) → DeepSeek, returning
an answer with inline `[n]` citations mapped back to real sources. It **refuses rather than
hallucinates** — out-of-corpus questions ("Can vLLM make coffee?") and genuine retrieval misses
both return *"I couldn't find this in the vLLM docs."*, demonstrating that generation quality is
bounded by retrieval.

## How to run

Prerequisites: **Python 3.11**, [`uv`](https://docs.astral.sh/uv/), and Docker (for the local
OpenSearch index). Runs on Linux/WSL or macOS; uses a CUDA GPU when available, otherwise CPU.

```bash
# install dependencies into a local .venv
uv sync

# copy the env template and fill in your keys
cp .env.example .env        # then edit: DEEPSEEK_API_KEY, OPENSEARCH_PASSWORD

# 1. ingest: load & clean vLLM docs → chunk → embed → bulk-index in OpenSearch (idempotent)
uv run python -m src.ingest

# 2. retrieval smoke test (semantic top-k for a couple of queries)
uv run python -m src.retrieve

# 3. eval: recall@1/3/5/10 over the fixed eval set; add --rerank for the before/after comparison
uv run python eval/eval_retrieval.py
uv run python eval/eval_retrieval.py --rerank

# 4. end-to-end RAG: retrieve → rerank → grounded answer with [n] citations
uv run python -m src.generate
```

## Success criterion

For 20–30 questions about vLLM, the system retrieves the relevant chunk(s) and reports
**recall@1 / recall@3 / recall@5** — reproducibly, from a fixed eval set.
