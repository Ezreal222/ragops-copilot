"""Latency + cost profiler (W7 D6 · step 1) — where does a /ask spend time and money?

Before optimizing anything we PROFILE. The D2/D4 numbers already hinted that the
LLM call dominates and retrieval is a rounding error, but "先测再改" (measure before
you change) means proving it with a decomposition, not asserting it — otherwise you
risk optimizing the 6ms embedding while the 6-second LLM call sits untouched.

This script breaks one fixed-RAG answer into its stages and reports, per question:

  - embed_ms  : bi-encoder encodes the query to a 384-d vector (CPU/GPU, local)
  - knn_ms    : OpenSearch HNSW k-NN fetches top-k chunks (local network hop)
  - llm_ms    : the DeepSeek chat call — retrieval done, this is the wall the user waits on
  - prompt_tok / completion_tok : the two halves of the token bill
  - cost_usd  : prompt_tok*price_prompt + completion_tok*price_completion (same math as metrics.py)

The retrieval stages are timed REPEAT times each (they're cheap and local, so a
median squeezes out noise); the LLM is called once per question (it's the slow,
paid part — we don't burn N× the money to measure variance we already know is high).

Reuses one preloaded embedder / OpenSearch / LLM client, exactly like the API's
lifespan preload — so the numbers reflect steady-state serving, not cold model load.

Run:
    uv run python -m eval.profile_latency            # default: 8 eval-set questions
    uv run python -m eval.profile_latency --n 12     # profile more
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from src.config import INDEX, LLM, RETRIEVAL
from src.embeddings import Embedder
from src.generate import build_messages
from src.generate import get_client as get_llm_client
from src.opensearch_client import get_client as get_os_client

EVAL_SET = Path(__file__).parent / "eval_set.jsonl"
REPEAT = 5  # times to re-time each cheap local stage (embed, knn) → take the median


def _load_questions(n: int) -> list[str]:
    """First `n` questions from the frozen eval set (deterministic sample)."""
    qs = []
    with EVAL_SET.open() as f:
        for line in f:
            line = line.strip()
            if line:
                qs.append(json.loads(line)["q"])
    return qs[:n]


def _time_retrieval(
    question: str, embedder: Embedder, os_client, k: int
) -> tuple[float, float, list[dict]]:
    """Return (median embed_ms, median knn_ms, chunks) — mirrors retrieve.search().

    We inline the two lines of search() so we can put a clock around each half;
    the reranker is off in prod (RETRIEVAL.use_reranker=False), so fixed-RAG
    retrieval really is just encode + one k-NN query. `k` is passed in so the
    D6 top_k lever (5->3) can be profiled without editing config.
    """
    embed_ms, knn_ms = [], []
    chunks: list[dict] = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        qv = embedder.encode_query(question)
        embed_ms.append((time.perf_counter() - t0) * 1000)

        body = {
            "size": k,
            "query": {"knn": {"embedding": {"vector": qv, "k": k}}},
            "_source": {"excludes": ["embedding"]},
        }
        t0 = time.perf_counter()
        res = os_client.search(index=INDEX.index_name, body=body)
        knn_ms.append((time.perf_counter() - t0) * 1000)

        chunks = [
            {"source": h["_source"]["source"], "title": h["_source"]["title"], "text": h["_source"]["text"]}
            for h in res["hits"]["hits"]
        ]
    return statistics.median(embed_ms), statistics.median(knn_ms), chunks


def _time_llm(question: str, chunks: list[dict], llm_client) -> tuple[float, int, int]:
    """One real LLM call; return (llm_ms, prompt_tokens, completion_tokens).

    Calls the chat API directly (rather than generate_answer) so we can read the
    `usage` block — the exact token counts the provider bills — off the response.
    """
    messages = build_messages(question, chunks)
    t0 = time.perf_counter()
    resp = llm_client.chat.completions.create(
        model=LLM.model,
        messages=messages,
        max_tokens=LLM.max_tokens,
        temperature=LLM.temperature,
    )
    llm_ms = (time.perf_counter() - t0) * 1000
    usage = resp.usage
    return llm_ms, usage.prompt_tokens, usage.completion_tokens


def _cost_usd(prompt_tok: int, completion_tok: int) -> float:
    """Dollars for one call — identical formula to metrics.record_llm_usage()."""
    return (
        prompt_tok * LLM.price_prompt_usd_per_1m
        + completion_tok * LLM.price_completion_usd_per_1m
    ) / 1_000_000


def main() -> None:
    ap = argparse.ArgumentParser(description="Profile /ask latency + cost by stage.")
    ap.add_argument("--n", type=int, default=8, help="number of eval-set questions to profile")
    ap.add_argument(
        "--top-k", type=int, default=RETRIEVAL.top_k,
        help="chunks retrieved per query (default = configured RETRIEVAL.top_k)",
    )
    args = ap.parse_args()

    top_k = args.top_k
    questions = _load_questions(args.n)
    print(f"Profiling {len(questions)} questions | model={LLM.model} | top_k={top_k}")
    print("(retrieval stages = median of %d runs; LLM = 1 call/question)\n" % REPEAT)

    # Preload once, like the API lifespan — measure serving, not cold start.
    embedder = Embedder()
    os_client = get_os_client()
    llm_client = get_llm_client()

    rows = []
    header = f"{'#':>2}  {'embed':>7}  {'knn':>7}  {'llm':>9}  {'total':>9}  {'p_tok':>6}  {'c_tok':>6}  {'$/q':>9}  question"
    print(header)
    print("-" * len(header))
    for i, q in enumerate(questions, 1):
        embed_ms, knn_ms, chunks = _time_retrieval(q, embedder, os_client, top_k)
        llm_ms, p_tok, c_tok = _time_llm(q, chunks, llm_client)
        total_ms = embed_ms + knn_ms + llm_ms
        cost = _cost_usd(p_tok, c_tok)
        rows.append(
            {"embed": embed_ms, "knn": knn_ms, "llm": llm_ms, "total": total_ms,
             "p_tok": p_tok, "c_tok": c_tok, "cost": cost}
        )
        print(
            f"{i:>2}  {embed_ms:>6.1f}m  {knn_ms:>6.1f}m  {llm_ms:>8.0f}m  {total_ms:>8.0f}m  "
            f"{p_tok:>6}  {c_tok:>6}  ${cost:>8.5f}  {q[:44]}"
        )

    # --- Aggregates: the numbers that decide the optimization target -----------
    def agg(key: str) -> tuple[float, float]:
        vals = sorted(r[key] for r in rows)
        p95 = vals[min(len(vals) - 1, int(round(0.95 * (len(vals) - 1))))]
        return statistics.mean(vals), p95

    embed_mean, _ = agg("embed")
    knn_mean, _ = agg("knn")
    llm_mean, llm_p95 = agg("llm")
    total_mean, total_p95 = agg("total")
    p_tok_mean = statistics.mean(r["p_tok"] for r in rows)
    c_tok_mean = statistics.mean(r["c_tok"] for r in rows)
    cost_mean = statistics.mean(r["cost"] for r in rows)

    print("\n=== Latency breakdown (mean) ===")
    retrieval_mean = embed_mean + knn_mean
    print(f"  embed      {embed_mean:>8.1f} ms   ({embed_mean / total_mean * 100:>4.1f}%)")
    print(f"  knn        {knn_mean:>8.1f} ms   ({knn_mean / total_mean * 100:>4.1f}%)")
    print(f"  retrieval  {retrieval_mean:>8.1f} ms   ({retrieval_mean / total_mean * 100:>4.1f}%)  <- embed+knn")
    print(f"  LLM        {llm_mean:>8.1f} ms   ({llm_mean / total_mean * 100:>4.1f}%)  <- the bottleneck")
    print(f"  total      {total_mean:>8.1f} ms   (p95 {total_p95:.0f} ms; LLM p95 {llm_p95:.0f} ms)")

    print("\n=== Cost breakdown (mean per query) ===")
    prompt_cost = p_tok_mean * LLM.price_prompt_usd_per_1m / 1_000_000
    completion_cost = c_tok_mean * LLM.price_completion_usd_per_1m / 1_000_000
    print(f"  prompt      {p_tok_mean:>7.0f} tok  ->  ${prompt_cost:.6f}  ({prompt_cost / cost_mean * 100:>4.1f}%)")
    print(f"  completion  {c_tok_mean:>7.0f} tok  ->  ${completion_cost:.6f}  ({completion_cost / cost_mean * 100:>4.1f}%)")
    print(f"  $/query     ${cost_mean:.6f}   (prompt {p_tok_mean:.0f} + completion {c_tok_mean:.0f} tok)")

    print("\n=== Conclusion ===")
    print(f"  LLM is {llm_mean / retrieval_mean:.0f}x the retrieval latency and ~all of the cost.")
    print("  -> optimize the LLM side (context length, prompt, routing). Retrieval is a rounding error.")


if __name__ == "__main__":
    main()
