"""Token + dollar accounting for the live LLM paths (extractor / agent / L3).

The pipeline has *no* cost model of its own (the rest of `kg` is offline-by-default
and never reads `msg.usage`). The test-run dashboard needs real per-document and
per-query cost, so this module adds a tiny, **offline-safe** accounting layer:

  * `PRICING` — per-model USD rates (from the Anthropic pricing reference; see the
    table below). Keyed by the model ids `Config` actually uses, plus the bare
    aliases an `--model` override might pass.
  * `UsageMeter` — a thread-safe accumulator. Every live `client.messages.create(...)`
    is recorded by reading `msg.usage`; the read is `getattr`-guarded so the scripted
    fake clients in the test-suite (which return turns with no `.usage`) never break,
    and the offline backends — which never call the API at all — simply carry an empty
    meter that reports `$0 / 0 tokens`. That is what keeps the offline path byte-for-byte
    unchanged while the live path is fully costed.

Rates are USD per **million** tokens (input, output, cache-read ≈ 0.1×input,
cache-write/5-min ≈ 1.25×input):

    claude-haiku-4-5(-20251001)   1.00 / 5.00  / 0.10 / 1.25   (extractor + L3 + agent default)
    claude-sonnet-4-6             3.00 / 15.00 / 0.30 / 3.75   (--model override)
    claude-opus-4-8               5.00 / 25.00 / 0.50 / 6.25   (--model override)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

_M = 1_000_000.0

# (input, output, cache_read, cache_write) USD per token.
PRICING: dict[str, tuple[float, float, float, float]] = {
    "claude-haiku-4-5-20251001": (1.00 / _M, 5.00 / _M, 0.10 / _M, 1.25 / _M),
    "claude-haiku-4-5":          (1.00 / _M, 5.00 / _M, 0.10 / _M, 1.25 / _M),
    "claude-sonnet-4-6":         (3.00 / _M, 15.00 / _M, 0.30 / _M, 3.75 / _M),
    "claude-opus-4-8":           (5.00 / _M, 25.00 / _M, 0.50 / _M, 6.25 / _M),
}
# Unknown / future model ids fall back to the cheap Haiku rate so a run is never
# free by accident; the model id is still recorded so the miss is visible.
_DEFAULT = PRICING["claude-haiku-4-5-20251001"]


def price(model: str, input_tokens: int, output_tokens: int,
          cache_read: int = 0, cache_write: int = 0) -> float:
    pin, pout, pcr, pcw = PRICING.get(model, _DEFAULT)
    return (input_tokens * pin + output_tokens * pout
            + cache_read * pcr + cache_write * pcw)


@dataclass
class CallRecord:
    """One billed `messages.create` call."""
    site: str          # "extract" | "agent" | "l3"
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read: int = 0
    cache_write: int = 0
    usd: float = 0.0
    label: str = ""    # doc id / query id, for per-item attribution


def _usage_fields(usage) -> tuple[int, int, int, int]:
    def g(k: str) -> int:
        return int(getattr(usage, k, 0) or 0)
    return (g("input_tokens"), g("output_tokens"),
            g("cache_read_input_tokens"), g("cache_creation_input_tokens"))


def totals_of(records: list[CallRecord]) -> dict:
    """Roll a list of CallRecords into the flat dict the dashboard consumes."""
    return {
        "llm_calls": len(records),
        "input_tokens": sum(r.input_tokens for r in records),
        "output_tokens": sum(r.output_tokens for r in records),
        "cache_read": sum(r.cache_read for r in records),
        "cache_write": sum(r.cache_write for r in records),
        "tokens": sum(r.input_tokens + r.output_tokens for r in records),
        "cost_usd": round(sum(r.usd for r in records), 6),
    }


def empty_totals() -> dict:
    return totals_of([])


class UsageMeter:
    """Thread-safe accumulator of CallRecords. Shared by an extractor across the
    ingest thread-pool, or owned by a single agent run — a lock keeps concurrent
    `record` calls safe either way."""

    def __init__(self):
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()

    def record(self, site: str, model: str, msg, label: str = "") -> CallRecord | None:
        """Record a live `messages.create` response. Returns None (recording nothing)
        when `msg` carries no `.usage` — i.e. a scripted fake or an offline stub — so
        callers can wire this in unconditionally without breaking the no-key path."""
        usage = getattr(msg, "usage", None)
        if usage is None:
            return None
        i, o, cr, cw = _usage_fields(usage)
        rec = CallRecord(site=site, model=model, input_tokens=i, output_tokens=o,
                         cache_read=cr, cache_write=cw,
                         usd=price(model, i, o, cr, cw), label=label)
        with self._lock:
            self.records.append(rec)
        return rec

    def totals(self) -> dict:
        with self._lock:
            return totals_of(list(self.records))

    def drain(self) -> list[CallRecord]:
        """Pop and return all records so far (used to attribute one document's worth
        of extractor calls before moving to the next document)."""
        with self._lock:
            recs, self.records = self.records, []
        return recs
