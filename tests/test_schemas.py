"""Unit tests for the API request contract (src.api.schemas.AskRequest).

The Pydantic model is the service's first line of defence: FastAPI validates the
JSON body against it and rejects bad input with a 422 *before* our retriever
runs. These tests pin that boundary — the constraints that keep a malformed
request from ever reaching the pipeline:

  - `question` must be a non-trivial string (3..1000 chars),
  - `k`, when given, must be a sane retrieval size (1..20),
  - `use_agent` defaults to None = "use the server's configured default", NOT a
    hardcoded True that would route production traffic around the D6 decision.

pydantic is a light import (no torch), so this stays fully hermetic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.api.schemas import AskRequest


def test_minimal_valid_request_uses_server_defaults():
    req = AskRequest(question="How does vLLM batch requests?")
    # Omitted optionals mean "defer to the server", represented as None — never a
    # baked-in engine/retrieval choice.
    assert req.use_agent is None
    assert req.k is None


def test_question_too_short_is_rejected():
    # min_length=3 — a 1-2 char "question" is noise, not a query.
    with pytest.raises(ValidationError):
        AskRequest(question="hi")


def test_question_too_long_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest(question="x" * 1001)


def test_missing_question_is_rejected():
    with pytest.raises(ValidationError):
        AskRequest()  # type: ignore[call-arg]


@pytest.mark.parametrize("k", [0, -1, 21])
def test_k_out_of_bounds_is_rejected(k):
    # k is the number of chunks fed to the LLM: 0/negative is meaningless, and >20
    # would blow the context/cost budget. Bounds are 1..20.
    with pytest.raises(ValidationError):
        AskRequest(question="valid question", k=k)


@pytest.mark.parametrize("k", [1, 5, 20])
def test_k_within_bounds_is_accepted(k):
    assert AskRequest(question="valid question", k=k).k == k


def test_use_agent_accepts_explicit_bool():
    assert AskRequest(question="valid question", use_agent=True).use_agent is True
    assert AskRequest(question="valid question", use_agent=False).use_agent is False
