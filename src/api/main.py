"""FastAPI service — the RAG/agent system as a callable HTTP API (W7 D1).

Run it:

    uv run uvicorn src.api.main:app --reload --port 8000
    # interactive docs: http://localhost:8000/docs

Everything below the transport layer already existed (retrieval, generation,
guardrails, the agent). What D1 adds is the four things that turn a library into
a *service*:

  1. **A contract** (schemas.py) — typed request/response, auto-validated,
     self-documenting via OpenAPI/Swagger.
  2. **Startup preloading** (deps.py) — the embedder / OpenSearch client / LLM
     client / agent graph are built ONCE in the lifespan hook and reused by every
     request. Rebuilding them per request would put seconds of model loading
     inside every user's latency.
  3. **Error handling that doesn't leak** — a client mistake is a 4xx, a
     dependency failure is a 5xx with a stable, boring message. Tracebacks, file
     paths and API keys stay on the server, in the log.
  4. **Instrumentation** — every /ask logs latency, mode, steps, degraded and
     refused. That log line is the raw material for the D4 monitoring work.

Sync vs async endpoints — a deliberate choice, not an oversight. `/ask` is
declared `def`, not `async def`, because the work it does is *blocking*: the bge
embedder is CPU/GPU-bound, and the OpenSearch and OpenAI SDKs are synchronous.
Blocking work inside an `async def` handler would run ON the event loop and stall
every other in-flight request. FastAPI runs a plain `def` handler in a threadpool
instead, so one slow answer can't freeze the server. (The genuinely right fix for
high concurrency is async HTTP clients for the two network calls plus a
threadpool for the embedder — a later optimization, and a real trade-off worth
naming rather than hiding.)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response
from opensearchpy.exceptions import ConnectionError as OSConnectionError
from opensearchpy.exceptions import ConnectionTimeout
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.agent.graph import _is_refusal
from src.answer import answer
from src.api.deps import build_deps, ping_opensearch
from src.api.schemas import AskRequest, AskResponse, HealthResponse
from src.config import LLM, SERVING
from src.metrics import ask_exceptions_total, classify_outcome, record_ask

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("ragops.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the heavy objects before the server accepts traffic; drop them after.

    This is the modern replacement for @app.on_event("startup"/"shutdown"): the
    code before `yield` runs once at boot, the code after runs once at shutdown.
    Anything expensive belongs here — see deps.py.
    """
    t0 = time.perf_counter()
    app.state.deps = build_deps()
    log.info("startup complete in %.1fs", time.perf_counter() - t0)
    yield
    # Nothing to close explicitly: the OpenSearch/OpenAI clients own pooled
    # connections that die with the process. Kept as the obvious seam for when
    # something DOES need releasing.
    app.state.deps = None


app = FastAPI(
    title="RAGOps Copilot",
    version="0.1.0",
    summary="Grounded, cited Q&A over the vLLM documentation.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Readiness probe: can this instance actually answer a question right now?

    Why every service needs one: a container orchestrator (Docker/ECS/K8s) and a
    load balancer have no way to know a process is *usable* — only that it's
    alive. /health is the contract that tells them, so a broken instance gets
    pulled from rotation instead of serving errors. It's also the first thing a
    deploy smoke test hits.

    We report `degraded` (503) rather than raising, so the failure is data, not an
    exception: the caller learns exactly WHICH dependency is down.
    """
    deps = request.app.state.deps
    reachable, count = ping_opensearch(deps)
    body = HealthResponse(
        status="ok" if reachable and count else "degraded",
        opensearch=reachable,
        indexed_chunks=count,
        embedder_loaded=deps.embedder is not None,
        model=LLM.model,
        default_mode="agent" if SERVING.use_agent else "rag",
    )
    if body.status != "ok":
        # 503 is the honest code: the process is up but not ready to serve.
        raise HTTPException(status_code=503, detail=body.model_dump())
    return body


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint — the whole registry, in its text format (D4).

    Prometheus is a PULL system: nothing here pushes anywhere. Every ~15s the
    Prometheus server GETs this path, parses the counters/histograms, and stores
    them as time series stamped with the scrape time. That inversion is the
    reason this endpoint is the entire integration — the app just has to keep
    honest numbers in memory and let itself be read. It also means metrics
    survive Prometheus being down (we keep counting; it backfills nothing, but
    the counters are cumulative so no *totals* are lost — only resolution).

    Why counters are cumulative-since-boot rather than per-interval: a restart
    resets them to 0, and Prometheus DETECTS that reset and handles it in
    `rate()`. Deltas computed by the app couldn't survive a missed scrape.

    `include_in_schema=False` keeps this out of the public /docs — it's an
    operator endpoint, not part of the product's API contract. It's also worth
    knowing this is an information leak in the making: it exposes traffic volume
    and spend to anyone who can reach it. Fine here (D3's Caddy only proxies
    /ask and /health publicly, so this is not internet-reachable), but a real
    deployment authenticates it or binds it to an internal interface.
    """
    # generate_latest() renders the default REGISTRY — every metric defined in
    # src/metrics.py registered itself there at import time — into the Prometheus
    # exposition format, which is why the media type must be theirs, not JSON.
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest, request: Request) -> AskResponse:
    """Answer a question over the vLLM docs, with citations.

    Routes to the agent or the fixed RAG pipeline per `use_agent` (defaulting to
    the server's SERVING.use_agent), reusing the preloaded heavy objects. The
    answer is grounded in retrieved excerpts and cites each claim with [n]; if the
    docs don't cover the question it refuses rather than guessing.
    """
    deps = request.app.state.deps
    t0 = time.perf_counter()

    try:
        result = answer(
            req.question,
            use_agent=req.use_agent,  # None -> the server's configured default
            k=req.k,
            app=deps.graph,
            os_client=deps.os_client,
            embedder=deps.embedder,
            reranker=deps.reranker,
            llm_client=deps.llm_client,
        )
    except (OSConnectionError, ConnectionTimeout) as exc:
        # A dependency is down — not the caller's fault. 503 tells the client this
        # is retryable. The exception TYPE goes to the log; the client gets a
        # plain sentence, never a traceback or a connection string.
        # The metric carries only that same exception CLASS as its label — never
        # the message. Label values become time series, so a message with a query
        # or an id in it would mint an unbounded number of them and melt
        # Prometheus (the classic "cardinality explosion"). Detail belongs in the
        # log line; the metric answers "how often, and which class".
        ask_exceptions_total.labels(kind=type(exc).__name__).inc()
        log.error("ask failed: OpenSearch unreachable (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail="The documentation search backend is unavailable. Please retry shortly.",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — the boundary: nothing escapes raw
        # exc_info puts the full traceback in the SERVER log, where it's useful;
        # the client gets a stable, information-free message. Leaking internals
        # here is how stack traces (and sometimes keys) end up in a user's browser.
        ask_exceptions_total.labels(kind=type(exc).__name__).inc()
        log.error("ask failed: unexpected error", exc_info=exc)
        raise HTTPException(
            status_code=500, detail="Internal error while answering the question."
        ) from exc

    latency_s = time.perf_counter() - t0
    latency_ms = latency_s * 1000

    # --- Instrumentation. These are the numbers that actually describe RAG
    # service health: how slow, which engine, how many tool rounds it needed, and
    # whether the user got a real answer or a refusal/fallback.
    #
    # D4 makes the same facts go two places, on purpose — the logs-vs-metrics
    # split (see src/metrics.py). The log line below keeps the per-request detail
    # (including the question text) for debugging one bad answer; record_ask()
    # feeds the aggregate view (p95, QPS, refusal rate) that a dashboard and an
    # alert read. Neither substitutes for the other: you can't alert on a log
    # line, and you can't debug a histogram bucket.
    refused = _is_refusal(result["answer"])
    record_ask(
        mode=result["mode"],
        outcome=classify_outcome(degraded=result["degraded"], refused=refused),
        latency_s=latency_s,
        steps=result["steps"],
    )
    log.info(
        'ask mode=%s steps=%d degraded=%s refused=%s citations=%d latency_ms=%.0f q="%s"',
        result["mode"],
        result["steps"],
        result["degraded"],
        refused,
        len(result["citations"]),
        latency_ms,
        " ".join(req.question.split())[:80],
    )

    return AskResponse(**result, latency_ms=latency_ms)
