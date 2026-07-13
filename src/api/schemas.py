"""API contract — the request/response models for the service (W7 D1).

These Pydantic models ARE the contract. Three things fall out of writing them
down instead of passing raw dicts around:

  1. **Validation for free.** FastAPI parses the JSON body into `AskRequest`;
     anything that doesn't fit (missing `question`, `k="five"`, `k=0`) is
     rejected with a 422 and a precise error path — before a single line of our
     code runs. Bad input never reaches the retriever.
  2. **Type safety inward.** Inside the handler `req.question` is a `str`, not a
     `body.get("question")` that might be None.
  3. **Docs for free.** FastAPI turns these into an OpenAPI schema, which is what
     renders the interactive Swagger UI at /docs. The contract can't drift from
     the code, because the contract *is* the code.

The response mirrors the unified dict `src.answer.answer()` already returns (see
its module docstring), plus `latency_ms` measured at the HTTP boundary. Keeping
the two shapes aligned means the API adds a transport layer, not a second data
model to keep in sync.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.config import SERVING


class AskRequest(BaseModel):
    """A question to answer over the vLLM docs."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural-language question about vLLM.",
        examples=["How does vLLM do continuous batching?"],
    )
    # NOTE: the default is None, not True. None means "use the server's configured
    # default" (SERVING.use_agent, currently False = fixed RAG). Why not default
    # the API to the agent: the W6 D6 regression measured the agent *below* fixed
    # RAG (context_precision 0.78->0.53, 22% of questions hitting the step cap),
    # so the agent stays behind a flag until it converges. Hardcoding `True` here
    # would silently route production traffic around that decision — config stays
    # the single place the routing default lives.
    use_agent: bool | None = Field(
        None,
        description=(
            "Route through the multi-step agent (true) or the fixed retrieve->"
            "generate pipeline (false). Omit to use the server default "
            f"(currently use_agent={SERVING.use_agent})."
        ),
    )
    k: int | None = Field(
        None,
        ge=1,
        le=20,
        description=(
            "Number of doc chunks to retrieve. Fixed-RAG path only — on the agent "
            "path the tools own their own k. Omit for the configured default."
        ),
    )


class Citation(BaseModel):
    """One source the answer actually cited via its [n] marker."""

    n: int = Field(..., description="The [n] marker used in the answer text.")
    source: str = Field(..., description="Source doc path/URL of the excerpt.")
    title: str = Field(..., description="Title of the source page.")


class ToolCall(BaseModel):
    """One tool the agent invoked, in order — the 'what did it actually do' trace."""

    name: str
    args: dict[str, Any]


class AskResponse(BaseModel):
    """A grounded, cited answer plus the observability fields around it."""

    answer: str
    citations: list[Citation]
    tool_trace: list[ToolCall]
    steps: int = Field(..., description="Tool-call rounds. Fixed RAG is always 1.")
    mode: Literal["agent", "rag"] = Field(..., description="Which engine answered.")
    degraded: bool = Field(
        ..., description="Agent hit its step cap and returned the fallback answer."
    )
    latency_ms: float = Field(..., description="Server-side time to produce the answer.")


class HealthResponse(BaseModel):
    """Liveness/readiness for container orchestration and deploy smoke tests.

    `status` is "ok" only when every dependency needed to answer a question is
    actually usable — a process that is *running* but can't reach OpenSearch
    cannot serve traffic, and a load balancer has to be told that.
    """

    status: Literal["ok", "degraded"]
    opensearch: bool = Field(..., description="Is the OpenSearch cluster reachable?")
    indexed_chunks: int | None = Field(
        None, description="Docs in the index (None if OpenSearch is unreachable)."
    )
    embedder_loaded: bool
    model: str = Field(..., description="The configured generation model.")
    default_mode: Literal["agent", "rag"]
