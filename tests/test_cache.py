"""Unit tests for src.cache — the W7 D6 response cache.

The cache exists to skip the LLM on a repeat question, so its correctness is
mostly about NOT serving the wrong answer: the right key must hit, a different
route/k must miss, a full cache must evict the least-recently-used entry, and a
stale entry must expire. These run with no external service — the cache is a pure
in-memory data structure, and time is injected so the TTL test is fast and
deterministic (no sleeping on a wall clock).
"""

from __future__ import annotations

import src.cache as cache_mod
from src.cache import ResponseCache, make_key


# --- make_key: what shares an entry, and what must not --------------------------

def test_key_normalizes_case_and_whitespace():
    # Trivial formatting differences should share one entry — that's the cheap,
    # safe half of "near-duplicate" matching.
    a = make_key("  What   is  PagedAttention? ", use_agent=False, k=None)
    b = make_key("what is pagedattention?", use_agent=False, k=None)
    assert a == b


def test_key_separates_route_and_k():
    # use_agent and k change the ANSWER, so they must change the key — otherwise a
    # cached agent answer could be served to a fixed-RAG request.
    base = make_key("q", use_agent=False, k=5)
    assert base != make_key("q", use_agent=True, k=5)
    assert base != make_key("q", use_agent=False, k=3)
    assert base != make_key("q", use_agent=False, k=None)


# --- get/set basics -------------------------------------------------------------

def test_miss_then_hit():
    c = ResponseCache(max_size=8, ttl_s=60)
    assert c.get("k") is None  # cold miss
    c.set("k", {"answer": "hi"})
    assert c.get("k") == {"answer": "hi"}  # warm hit


def test_set_overwrites():
    c = ResponseCache(max_size=8, ttl_s=60)
    c.set("k", 1)
    c.set("k", 2)
    assert c.get("k") == 2
    assert len(c) == 1


# --- LRU eviction ---------------------------------------------------------------

def test_evicts_least_recently_used_when_full():
    c = ResponseCache(max_size=2, ttl_s=60)
    c.set("a", 1)
    c.set("b", 2)
    # Touch "a" so "b" becomes the least-recently-used, then overflow.
    assert c.get("a") == 1
    c.set("c", 3)  # capacity 2 exceeded -> evict the LRU, which is now "b"
    assert c.get("b") is None
    assert c.get("a") == 1
    assert c.get("c") == 3
    assert len(c) == 2


# --- TTL expiry (time injected, no real sleep) ----------------------------------

def test_entry_expires_after_ttl(monkeypatch):
    clock = {"t": 1000.0}
    monkeypatch.setattr(cache_mod.time, "monotonic", lambda: clock["t"])

    c = ResponseCache(max_size=8, ttl_s=10)
    c.set("k", "v")
    clock["t"] += 5  # within ttl
    assert c.get("k") == "v"
    clock["t"] += 6  # now 11s old > 10s ttl
    assert c.get("k") is None
    # Expired entries are dropped on access (lazy eviction), so the map shrinks.
    assert len(c) == 0


def test_clear_empties_the_cache():
    c = ResponseCache(max_size=8, ttl_s=60)
    c.set("a", 1)
    c.set("b", 2)
    c.clear()
    assert len(c) == 0
    assert c.get("a") is None
