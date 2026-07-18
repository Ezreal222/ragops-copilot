"""Server dependencies — build the heavy objects ONCE, at startup (W7 D1).

This module exists to answer one question: *what does a request need that is
expensive to create?*

  - the **embedder** (bge-small): loads ~130 MB of weights onto the GPU/CPU,
  - the **OpenSearch client**: opens a connection pool,
  - the **reranker** (bge-reranker, only if RETRIEVAL.use_reranker): ~280 MB,
  - the **LLM client** (OpenAI-compatible SDK): reads the key, sets up HTTP,
  - the compiled **agent graph**: builds the chat model and binds tool schemas.

Every one of those is *seconds* to build and *stateless* to use. Building them
per request would add that cost to every single answer (and, for the models,
re-allocate hundreds of MB each time). So the server builds them once in its
lifespan startup, stashes them on `app.state`, and every request borrows them.
This is the single most important performance property of the service, and the
D1 cold-vs-warm latency measurement exists to *prove* it happens only once.

The pipeline was already written to make this possible: `ask()`, `search()` and
`run_agent()` all take their heavy objects as injectable arguments (the eval
harness needed the same thing). `Deps` is just the bundle, and `answer()` takes
it apart again.

One subtlety: the agent's tools don't take injected resources — they read
module-level caches in `src.agent.tools`. So `build_deps()` calls
`tools.preload_resources(...)` to push OUR embedder/client into those globals.
Result: one embedder in the process, shared by both routing paths.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from src.agent import tools as agent_tools
from src.agent.graph import build_graph
from src.cache import ResponseCache
from src.config import CACHE, EMBED, INDEX, LLM, RERANK, RETRIEVAL, SERVING
from src.embeddings import Embedder
from src.generate import get_client as get_llm_client
from src.opensearch_client import get_client as get_os_client
from src.reranker import Reranker

log = logging.getLogger("ragops.api")


@dataclass
class Deps:
    """The warm, shared objects a request needs. Built once; never mutated."""

    embedder: Embedder
    os_client: Any  # opensearchpy.OpenSearch
    reranker: Reranker | None  # None when RETRIEVAL.use_reranker is False
    llm_client: Any  # openai.OpenAI — used by the fixed-RAG path
    graph: Any  # compiled LangGraph app — used by the agent path
    cache: ResponseCache  # W7 D6 — repeat /ask skips the LLM (see src/cache.py)


def build_deps() -> Deps:
    """Construct every heavy object. Call this exactly once, at startup.

    The reranker is built only if the config asks for it: the W5 D6 ablation
    turned it OFF (it lost on every metric at 5x the latency), so loading its
    280 MB unconditionally would mean paying for a model we deliberately don't use.
    """
    log.info("startup: loading embedder %s ...", EMBED.model_name)
    embedder = Embedder()

    # Loading the weights is not the whole cost. The FIRST encode also pays lazy
    # one-time init (CUDA context + kernel autotuning on GPU; graph warm-up on
    # CPU) — measured at ~480 ms vs ~12 ms for every encode after it. Preloading
    # the model but not exercising it would just move that 480 ms onto the first
    # unlucky user, so we burn one throwaway encode here, at boot.
    embedder.encode_query("warmup")

    log.info("startup: connecting to OpenSearch at %s ...", INDEX.host)
    os_client = get_os_client()

    reranker = None
    if RETRIEVAL.use_reranker:
        log.info("startup: loading reranker %s ...", RERANK.model_name)
        reranker = Reranker()

    log.info("startup: building LLM client (%s / %s) ...", LLM.provider, LLM.model)
    llm_client = get_llm_client(LLM)

    log.info("startup: compiling agent graph ...")
    graph = build_graph(LLM)

    # Hand the agent's tools the SAME embedder/client, so the process holds one
    # copy of the model rather than one per routing path (see module docstring).
    agent_tools.preload_resources(embedder=embedder, os_client=os_client)

    # The response cache is built unconditionally (it's a tiny dict); whether it's
    # consulted is CACHE.enabled, checked per request in the handler. That keeps
    # "is caching on?" a one-line config flip, not a wiring change.
    log.info(
        "startup: response cache %s (max_size=%d, ttl=%.0fs)",
        "ON" if CACHE.enabled else "OFF (config)",
        CACHE.max_size,
        CACHE.ttl_s,
    )
    cache = ResponseCache(max_size=CACHE.max_size, ttl_s=CACHE.ttl_s)

    log.info("startup: ready (default mode=%s)", "agent" if SERVING.use_agent else "rag")
    return Deps(
        embedder=embedder,
        os_client=os_client,
        reranker=reranker,
        llm_client=llm_client,
        graph=graph,
        cache=cache,
    )


def ping_opensearch(deps: Deps) -> tuple[bool, int | None]:
    """Is OpenSearch reachable, and how many chunks are indexed?

    Used by /health. We don't just ping the cluster — we `count` the index,
    because a reachable cluster with an EMPTY index still can't answer anything.
    Any failure is swallowed into (False, None): a health check must REPORT
    unhealthy, never raise.
    """
    try:
        return True, deps.os_client.count(index=INDEX.index_name)["count"]
    except Exception as exc:  # noqa: BLE001 — a health check reports, never raises
        log.warning("health: OpenSearch unreachable: %s", type(exc).__name__)
        return False, None
