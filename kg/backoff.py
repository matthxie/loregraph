"""Retry-with-backoff for live LLM calls.

Rate limits (429) and transient 5xx errors are infrastructure noise, not signal: an
unretried 429 either crashes a query or silently degrades it to the extractive fallback,
which then scores as a (fake) memory failure. Every live call site (extract / answer /
judge / L3) wraps its API call in `call_with_backoff` so a transient error costs seconds,
not a corrupted sample.

Exponential backoff with full jitter; non-retryable errors re-raise immediately.
"""
from __future__ import annotations

import random
import time

_RETRYABLE_NAMES = {"RateLimitError", "APITimeoutError", "APIConnectionError",
                    "InternalServerError", "ServiceUnavailableError"}
_RETRYABLE_STATUS = {429, 500, 502, 503, 529}


def _retryable(e: Exception) -> bool:
    if type(e).__name__ in _RETRYABLE_NAMES:
        return True
    return getattr(e, "status_code", None) in _RETRYABLE_STATUS


def call_with_backoff(fn, *, tries: int = 5, base: float = 2.0, max_sleep: float = 30.0):
    """Call fn(); on a retryable API error sleep (exponential, jittered) and retry.
    Re-raises immediately on non-retryable errors and after the last attempt."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 — filtered by _retryable below
            if attempt == tries - 1 or not _retryable(e):
                raise
            time.sleep(min(max_sleep, base * (2 ** attempt)) * (0.5 + random.random()))
