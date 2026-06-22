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
| LLM (generation) | **Anthropic / Claude** | Citation-faithful answering grounded in retrieved context |
| Retrieval metric | **recall@1 / recall@3 / recall@5** | Core measure of retrieval quality, reproducible from a fixed eval set |

## Tech stack

LangChain (splitters/retriever) · OpenSearch (vector index) · sentence-transformers (bge-small) ·
bge-reranker · Anthropic Claude (generation) · RAGAS · LangGraph · FastAPI · Docker · AWS.

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
| **1 · Retrieval** | Ingest → chunk → embed → OpenSearch → semantic retrieval → **recall@1/3/5**, then add a reranker. |
| **2 · Evaluation** | RAGAS (faithfulness / answer relevancy / context precision-recall) + LLM-as-judge. |
| **3 · Agent** | LangGraph planning + tool loop + guardrails. |
| **4 · Serving & MLOps** | FastAPI · Docker · AWS · monitoring (latency / cost / failure rate) · CI/CD. |

## How to run

Prerequisites: **Python 3.11**, [`uv`](https://docs.astral.sh/uv/), and Docker (for the local
OpenSearch index). Runs on Linux/WSL or macOS; uses a CUDA GPU when available, otherwise CPU.

```bash
# install dependencies into a local .venv
uv sync

# copy the env template and fill in your keys
cp .env.example .env        # then edit: ANTHROPIC_API_KEY, OpenSearch creds

# ingestion / retrieval entry points live under src/
uv run python -m src.<module>
```

## Success criterion

For 20–30 questions about vLLM, the system retrieves the relevant chunk(s) and reports
**recall@1 / recall@3 / recall@5** — reproducibly, from a fixed eval set.
