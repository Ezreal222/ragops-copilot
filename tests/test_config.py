"""Unit tests for src.config — the pipeline's tunable knobs.

config.py is pure data (frozen dataclasses, no I/O), so these run anywhere with
no external service. They aren't testing arithmetic — they're guarding the
*decisions* baked into the defaults so a careless edit can't silently move them:

  - the eval-derived refusal threshold (0.82) stays inside its justified band,
  - the two retrieval stages stay consistent (top_k <= rerank_top_n),
  - the config objects stay frozen (callers can't mutate shared global state),
  - the production degrade switch defaults to the stable fixed-RAG path.

If a value here needs to change, the test should change WITH it in the same
commit — that's the point: the number and its guard travel together.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import AGENT, EMBED, LLM, RETRIEVAL, SERVING


def test_refusal_threshold_in_eval_derived_band():
    # analyze_score_threshold.py put on-topic top-1 min at 0.834 and off-topic
    # median at 0.793; 0.82 sits between them. Guard that band so a "let's just
    # bump it" edit can't quietly start refusing real questions or grounding on
    # off-topic noise.
    assert 0.793 < AGENT.min_relevance_score < 0.834


def test_retrieval_stages_are_consistent():
    # Two-stage retrieval: k-NN returns rerank_top_n candidates, the reranker
    # trims them to top_k. Feeding the LLM more chunks than stage 1 produced is
    # incoherent, so top_k must never exceed the candidate pool.
    assert RETRIEVAL.top_k <= RETRIEVAL.rerank_top_n
    assert RETRIEVAL.top_k >= 1


def test_embedding_dimension_matches_bge_small():
    # The index mapping's knn_vector dimension is hard-wired to this number;
    # a mismatch makes OpenSearch reject every write. bge-small-en-v1.5 = 384.
    assert EMBED.dimension == 384


def test_serving_defaults_to_fixed_rag():
    # The W6 D6 regression kept the agent behind a flag; the runtime default must
    # stay on the stable fixed pipeline until the agent converges.
    assert SERVING.use_agent is False


def test_prices_are_positive():
    # Cost metrics multiply token counts by these; a zero/negative price would
    # silently zero out the dashboard's dollar figures.
    assert LLM.price_prompt_usd_per_1m > 0
    assert LLM.price_completion_usd_per_1m > 0


def test_config_objects_are_frozen():
    # frozen=True is what lets us share single global instances safely. Prove the
    # immutability actually holds — a plain @dataclass would let this through.
    with pytest.raises(dataclasses.FrozenInstanceError):
        RETRIEVAL.top_k = 99
