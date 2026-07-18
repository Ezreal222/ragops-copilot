"""In-process response cache (W7 D6) — turn a repeat /ask into a dict lookup.

The D6 latency profile (eval/profile_latency.py) settled where the time and money
go: the LLM call is ~99.9% of a request's latency (4-13 s) and ~all of its cost,
while retrieval is ~13 ms. That makes one optimization obvious — if we've already
answered this exact question, don't call the LLM again. A cache hit collapses the
whole pipeline to a hash lookup: sub-millisecond, and $0.

This is a **thread-safe LRU cache with a TTL**, and every one of those three
properties is here for a concrete reason:

  - **LRU + max_size**: the cache holds whole answer payloads, so it must be
    bounded or it's a memory leak. When it's full, the least-recently-USED entry
    (not the oldest-inserted) evicts — a hot FAQ stays warm even as rarer
    questions churn through. `OrderedDict.move_to_end` on read + `popitem(last=
    False)` on overflow is the textbook O(1) LRU.
  - **TTL**: the corpus can be re-ingested underneath a running server, which
    would leave a cached answer citing chunks that no longer exist. A per-entry
    expiry bounds that staleness window without any cache-invalidation plumbing.
  - **Thread-safety**: FastAPI runs the sync /ask handler in a threadpool, so
    several requests touch this map at once. A single lock around the (short,
    non-blocking) get/set critical sections is enough and never wraps an LLM call.

Deliberately NOT here: semantic / near-duplicate matching. The key is the
normalized question text, so "what is paged attention" and "What's PagedAttention?"
miss each other. Matching those needs an embedding-similarity lookup with a
threshold — a real feature with a real false-hit failure mode (returning the wrong
cached answer), so it's a documented next step, not a silent default.

Scope is one process (see the CacheConfig docstring): correct for our single
uvicorn worker; a shared Redis cache is the multi-worker upgrade, and this class
is the seam for it — swap the dict for a Redis client and the call sites don't
change.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any


def make_key(question: str, *, use_agent: bool, k: int | None) -> str:
    """Build the cache key from the question and the knobs that change its answer.

    The question is normalized — lower-cased and whitespace-collapsed — so trivial
    formatting differences ("  What is  X? " vs "what is x?") share one entry;
    that's the cheap, safe half of "near-duplicate" matching (case + spacing),
    without the risk of conflating genuinely different questions.

    `use_agent` and `k` are part of the key because they change the ANSWER: the
    agent and fixed-RAG paths can answer differently, and a different k feeds the
    LLM a different context. Caching on the question alone would serve an agent
    answer to a fixed-RAG request. `k` is normalized via repr so None (meaning
    "server default") and an explicit int don't collide confusingly.
    """
    norm_q = " ".join(question.lower().split())
    return f"{norm_q}|use_agent={use_agent}|k={k!r}"


class ResponseCache:
    """A bounded, expiring, thread-safe question -> answer map.

    Stores whatever value the caller hands `set()` (here: the `answer()` result
    dict). It knows nothing about the answer's shape — that keeps it reusable and
    testable in isolation. Hit/miss accounting is the CALLER's job (the API records
    the Prometheus counter), so this class stays a pure data structure.
    """

    def __init__(self, max_size: int, ttl_s: float) -> None:
        self._max_size = max_size
        self._ttl_s = ttl_s
        # key -> (stored_at_monotonic, value). OrderedDict gives us LRU ordering:
        # the leftmost item is the least-recently-used.
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        """Return the cached value, or None on a miss or an expired entry.

        A None return is BOTH "never seen" and "seen but stale" — the caller
        treats them identically (recompute), so we don't distinguish. An expired
        entry is deleted on access (lazy eviction), which is enough for a cache
        that's read on every request; we don't run a background sweeper.
        """
        now = time.monotonic()
        with self._lock:
            hit = self._store.get(key)
            if hit is None:
                return None
            stored_at, value = hit
            if now - stored_at > self._ttl_s:
                # Expired: drop it and report a miss.
                del self._store[key]
                return None
            # Fresh hit: mark it most-recently-used so it survives eviction.
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        """Insert/refresh an entry, evicting the LRU one if we're over capacity."""
        now = time.monotonic()
        with self._lock:
            self._store[key] = (now, value)
            self._store.move_to_end(key)
            # Evict from the left (least-recently-used) until we're within bounds.
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop everything — used by tests and the natural manual-flush hook."""
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)
