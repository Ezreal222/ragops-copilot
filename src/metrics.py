"""Prometheus metrics — the service's aggregatable health signals (W7 D4).

D1 already instrumented every /ask with a log line (mode, steps, degraded,
refused, latency_ms). That line is perfect for debugging ONE request and useless
for answering "is the system healthy?" — to get p95 latency out of logs you have
to grep, parse and sort them. **That's the logs-vs-metrics distinction**: a log is
a discrete event (rich, high-cardinality, read by a human after something broke);
a metric is a pre-aggregated time series (cheap, low-cardinality, read by a
dashboard and an alert rule, continuously). Production needs both. D4 adds the
second half, over the exact same events D1 was already logging.

What we track, and why each is the metric a RAG service actually lives or dies by:

  - `ask_latency_seconds{mode}` — **Histogram**, so Prometheus can compute
    p50/p95/p99. Not a Gauge of the mean: an average is dragged down by the many
    fast requests and hides the few users having an awful time. p95/p99 is the
    tail, and the tail is what an SLA is written in.
  - `ask_requests_total{mode,outcome}` — **Counter**. QPS is its rate(); the
    refusal rate is the `outcome="refused"` share. Refusal rate is RAG-specific
    and worth watching closely: a spike means retrieval stopped finding the docs
    (bad index, bad embedder) long before anyone files a bug.
  - `ask_exceptions_total` — **Counter**. The failure rate — 5xx that escaped.
  - `tool_errors_total{tool}` — **Counter**. Agent tool failures, which degrade
    an answer WITHOUT raising (see tools.safe_tool), so they'd be invisible in
    the exception count.
  - `llm_tokens_total{type}` + `llm_cost_usd_total` — **Counters**. The cost
    dimension that is unique to LLM apps: every request spends real money, and
    the only way to see it is to count tokens x price. This is the third axis of
    the quality/latency/**cost** triangle the whole project is about.
  - `agent_steps` — **Histogram** (distribution) + `agent_steps_last` **Gauge**
    (most recent). Tool-call rounds per question: a healthy lookup is 1, and a
    creeping distribution means the agent is thrashing.

That's all four Prometheus metric types in one file, each where it belongs:
**Counter** (monotonic, only goes up — requests, errors, tokens, dollars),
**Gauge** (goes up and down — last step count), **Histogram** (buckets the
observations, server-side quantiles — latency, steps). The fourth, **Summary**,
computes quantiles in the CLIENT and can't be aggregated across instances, which
is why latency here is a Histogram: with several API replicas you can still get a
true fleet-wide p95 from buckets, but you can NEVER average per-instance p95s.

Why this module is `src/metrics.py` and not `src/api/metrics.py`: token usage is
recorded inside generate.py and agent/graph.py, and tool errors inside
agent/tools.py. Those are pipeline modules — if they imported from `src.api.*`
the core would depend on its own transport layer, and the eval harness would drag
in FastAPI. Metrics are cross-cutting, so they live beside the pipeline; the API
just owns the /metrics endpoint that exposes them.

Concurrency/process note: prometheus_client's metrics are thread-safe, which is
what we need — FastAPI runs the sync `/ask` handler in a threadpool, so several
requests increment these concurrently. They are, however, PER-PROCESS: the
counters live in this process's memory. That's correct for our single-uvicorn-
worker deployment. The day we run multiple workers, each would expose its own
numbers and we'd need prometheus_client's multiprocess mode (or simply let
Prometheus scrape each worker as its own target and sum in the query).
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

from src.config import LLM, LLMConfig

# --- Latency ------------------------------------------------------------------
# Buckets are the one Histogram decision that matters: Prometheus computes
# quantiles by INTERPOLATING inside the bucket a quantile falls in, so a p95 is
# only as precise as the bucket boundaries around it. The defaults (.005s..10s)
# are tuned for fast web handlers and would lump nearly all of our traffic into
# one or two buckets — a RAG answer is an LLM call, i.e. seconds, and the
# thinking model can run much longer. These span 0.25s..60s with fine resolution
# where our answers actually land (2-15s), so p95 lands in a narrow bucket.
_LATENCY_BUCKETS = (0.25, 0.5, 1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 20, 30, 45, 60)

ask_latency_seconds = Histogram(
    "ask_latency_seconds",
    "End-to-end server time to answer a question, by engine.",
    labelnames=("mode",),  # agent | rag — the two engines cost very different time
    buckets=_LATENCY_BUCKETS,
)

# --- Request outcomes ---------------------------------------------------------
# `outcome` is deliberately a THREE-valued label, not a bool: "the request
# succeeded" is not the same as "the user got an answer". A refusal is a 200 OK
# and a correct behaviour (better than hallucinating), but a *rising* refusal rate
# is a retrieval regression. A degraded answer is the agent giving up after its
# step cap. Both are healthy-looking in HTTP terms and unhealthy in product terms,
# so they get their own outcome values rather than hiding inside "answered".
ask_requests_total = Counter(
    "ask_requests_total",
    "Answered questions, by engine and outcome.",
    labelnames=("mode", "outcome"),  # outcome: answered | refused | degraded
)

ask_exceptions_total = Counter(
    "ask_exceptions_total",
    "Requests that failed with an unhandled error (5xx), by exception class.",
    labelnames=("kind",),  # e.g. ConnectionError — which dependency broke
)

tool_errors_total = Counter(
    "tool_errors_total",
    "Agent tool invocations that failed and returned a TOOL ERROR observation.",
    labelnames=("tool",),
)

# --- Agent health -------------------------------------------------------------
# Two views of the same number, because they answer different questions.
# The Histogram answers "what does the step distribution look like over the last
# hour?" (the real health signal — is the agent creeping toward its cap?).
# The Gauge answers "what did the last request do?" — trivial to read on a panel,
# and the honest limitation of a Gauge here is that it's a sample of one: under
# concurrency it just shows whichever request finished last. Never alert on it.
_STEP_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10)

agent_steps = Histogram(
    "agent_steps",
    "Tool-call rounds per question (fixed RAG is always 1).",
    buckets=_STEP_BUCKETS,
)

agent_steps_last = Gauge(
    "agent_steps_last",
    "Tool-call rounds used by the most recently completed question.",
)

# --- LLM cost -----------------------------------------------------------------
llm_tokens_total = Counter(
    "llm_tokens_total",
    "LLM tokens consumed, by engine and direction.",
    labelnames=("mode", "type"),  # mode: agent|rag — type: prompt|completion
)

llm_cost_usd_total = Counter(
    "llm_cost_usd_total",
    "Cumulative LLM spend in USD, derived from token counts x configured price.",
    labelnames=("mode",),  # agent | rag
)


def record_llm_usage(
    prompt_tokens: int,
    completion_tokens: int,
    *,
    mode: str,
    config: LLMConfig = LLM,
) -> None:
    """Record one LLM call's token usage, and the dollars it cost.

    Called at every LLM boundary — generate.py (fixed-RAG path) and the agent
    node in graph.py (agent path, once per turn, so a 3-step question records 3
    times). Counting at the boundary rather than threading a token count back up
    through `answer()` is what keeps this cheap: Prometheus counters are global
    and additive, so the call sites just report and forget. The consequence to be
    aware of is that these are PROCESS-wide totals, not per-request — for
    $/query you divide the cost counter by the request counter (which is exactly
    what the Grafana panel does).

    Cost is computed HERE, at record time, rather than derived in the dashboard,
    for one reason: prices differ per model and change over time. Multiplying at
    the moment of spend means the counter is always denominated in the price that
    was actually in effect, and no PromQL query has to hardcode a rate.

    Both providers report usage on every response, so this needs no estimation —
    we're reading the number the vendor will bill us for, not a tokenizer guess.

    `mode` labels the spend by engine, which is what makes "how much more does the
    agent cost per question than fixed RAG?" a query instead of a guess. Each call
    site passes a CONSTANT: generate.py is only ever reached on the fixed-RAG path
    and passes "rag"; the agent node is only ever the agent path and passes
    "agent". Nothing has to thread routing state down here, and the label can't
    drift from reality.
    """
    if prompt_tokens:
        llm_tokens_total.labels(mode=mode, type="prompt").inc(prompt_tokens)
    if completion_tokens:
        llm_tokens_total.labels(mode=mode, type="completion").inc(completion_tokens)

    # Prices are per 1M tokens (how every provider quotes them); convert to
    # per-token here so config stays readable next to the vendor's pricing page.
    cost = (
        prompt_tokens * config.price_prompt_usd_per_1m
        + completion_tokens * config.price_completion_usd_per_1m
    ) / 1_000_000
    if cost:
        llm_cost_usd_total.labels(mode=mode).inc(cost)


def record_ask(*, mode: str, outcome: str, latency_s: float, steps: int) -> None:
    """Record one completed /ask — its latency, outcome, and step count.

    One function rather than four call sites in the handler, so "what do we
    measure per request" is defined in one place and can't drift between the two
    engines.
    """
    ask_latency_seconds.labels(mode=mode).observe(latency_s)
    ask_requests_total.labels(mode=mode, outcome=outcome).inc()
    agent_steps.observe(steps)
    agent_steps_last.set(steps)


def classify_outcome(*, degraded: bool, refused: bool) -> str:
    """Bucket a result into answered | refused | degraded (in that precedence).

    `degraded` wins over `refused` because the agent's step-cap fallback text is
    itself matched by the refusal check (`graph._is_refusal` treats FALLBACK_ANSWER
    as a refusal) — and the two mean very different things. A refusal is the
    system working correctly on an unanswerable question; a degraded answer is the
    agent failing to converge on one it should have handled. Collapsing them would
    hide agent non-convergence inside a metric that looks like healthy behaviour.
    """
    if degraded:
        return "degraded"
    if refused:
        return "refused"
    return "answered"
