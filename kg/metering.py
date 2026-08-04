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

    gpt-4o-mini                   0.15 / 0.60  / 0.075 / 0.15   (app-key ingest pin)
    gpt-5.4-nano                  0.20 / 1.25  / 0.02  / 0.25   (former app-key extractor pin)
    gpt-5.4-mini                  0.75 / 4.50  / 0.075 / 0.9375 (app-key chat + Ask pin)
    gpt-5.6-luna                  1.00 / 6.00  / 0.10  / 1.25   (former app-key RAG answerer pin)
    gpt-5.6-terra                 2.50 / 15.00 / 0.25  / 3.125  (former app-key agentic tier)
    claude-haiku-4-5(-20251001)   1.00 / 5.00  / 0.10 / 1.25   (legacy reference)
    claude-sonnet-4-6             3.00 / 15.00 / 0.30 / 3.75   (legacy reference)
    claude-opus-4-8               5.00 / 25.00 / 0.50 / 6.25   (legacy reference)
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

_M = 1_000_000.0

# (input, output, cache_read, cache_write) USD per token.
#
# The cache_write rate must never be 0 for a model that can report writes: `_usage_fields`
# carves written tokens OUT of input_tokens, so a zero fourth rate silently bills those
# tokens at nothing. "No cache-write surcharge" means writes bill as ORDINARY INPUT —
# for those models the fourth rate equals the input rate.
PRICING: dict[str, tuple[float, float, float, float]] = {
    # OpenAI automatic prefix caching bills cached input at 50% ($0.075/M for 4o-mini);
    # no cache-write surcharge (writes = ordinary input).
    "gpt-4o-mini":               (0.15 / _M, 0.60 / _M, 0.075 / _M, 0.15 / _M),
    "gpt-4o-mini-2024-07-18":    (0.15 / _M, 0.60 / _M, 0.075 / _M, 0.15 / _M),
    # gpt-5 family: cached input bills at 10% of input; no cache-write surcharge.
    "gpt-5":                     (1.25 / _M, 10.00 / _M, 0.125 / _M, 1.25 / _M),
    "gpt-5-mini":                (0.25 / _M, 2.00 / _M, 0.025 / _M, 0.25 / _M),
    "gpt-5-nano":                (0.05 / _M, 0.40 / _M, 0.005 / _M, 0.05 / _M),
    # gpt-5.4/5.6 families (July 2026 list): cached input at 10% of input; cache WRITES bill
    # at 1.25x input — a real third rate, not ordinary input (nano's and mini's are derived
    # from that family rule; luna/terra are published figures). These rows mirror brainbrain's
    # electron/src/main/pricing.ts MODEL_RATES; the app trusts this table for engine-side
    # calls, so keep the two in lockstep when either changes.
    "gpt-5.4-nano":              (0.20 / _M, 1.25 / _M, 0.02 / _M, 0.25 / _M),
    "gpt-5.4-nano-2026-03-17":   (0.20 / _M, 1.25 / _M, 0.02 / _M, 0.25 / _M),
    "gpt-5.4-mini":              (0.75 / _M, 4.50 / _M, 0.075 / _M, 0.9375 / _M),
    "gpt-5.4-mini-2026-03-17":   (0.75 / _M, 4.50 / _M, 0.075 / _M, 0.9375 / _M),
    "gpt-5.6-luna":              (1.00 / _M, 6.00 / _M, 0.10 / _M, 1.25 / _M),
    "gpt-5.6-terra":             (2.50 / _M, 15.00 / _M, 0.25 / _M, 3.125 / _M),
    "claude-haiku-4-5-20251001": (1.00 / _M, 5.00 / _M, 0.10 / _M, 1.25 / _M),
    "claude-haiku-4-5":          (1.00 / _M, 5.00 / _M, 0.10 / _M, 1.25 / _M),
    "claude-sonnet-4-6":         (3.00 / _M, 15.00 / _M, 0.30 / _M, 3.75 / _M),
    "claude-opus-4-8":           (5.00 / _M, 25.00 / _M, 0.50 / _M, 6.25 / _M),
}
# Unknown / future model ids fall back to the cheap gpt-4o-mini rate so a run is never
# free by accident; the model id is still recorded so the miss is visible.
_DEFAULT = PRICING["gpt-4o-mini"]


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
    truncated: bool = False   # response hit max_tokens (output ceiling) → payload cut short


def _usage_fields(usage) -> tuple[int, int, int, int]:
    def g(*keys: str) -> int:
        for k in keys:
            v = getattr(usage, k, None)
            if v is not None:
                return int(v)
        return 0
    # OpenAI uses prompt_tokens/completion_tokens; Anthropic uses input_tokens/output_tokens.
    i = g("input_tokens", "prompt_tokens")
    o = g("output_tokens", "completion_tokens")
    cr = g("cache_read_input_tokens")            # Anthropic name
    cw = g("cache_creation_input_tokens")
    if cr == 0:
        # OpenAI reports automatic prefix-cache hits under prompt_tokens_details.cached_tokens,
        # and (unlike Anthropic) its prompt_tokens INCLUDES the cached portion — split it out
        # so cached tokens are priced at the discounted cache_read rate, not double-counted.
        ptd = getattr(usage, "prompt_tokens_details", None)
        cached = int(getattr(ptd, "cached_tokens", 0) or 0) if ptd is not None else 0
        if cached:
            cr = cached
            i = max(0, i - cached)
        # Cache WRITES are a third rate (1.25x input on the gpt-5 families), reported in the
        # same details object and — like cached — already counted inside prompt_tokens.
        # Missing it under-bills every chained call, and the app now consumes these dollars
        # as a real spend cap, so under-billing is the direction that actually hurts.
        written = int(getattr(ptd, "cache_write_tokens", 0) or 0) if ptd is not None else 0
        if written:
            cw = written
            i = max(0, i - written)
    return (i, o, cr, cw)


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
        "truncated": sum(1 for r in records if r.truncated),
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
        # OpenAI: finish_reason on choices[0]; Anthropic: stop_reason on the message.
        choices = getattr(msg, "choices", None)
        if choices:
            truncated = getattr(choices[0], "finish_reason", None) == "length"
        else:
            truncated = getattr(msg, "stop_reason", None) == "max_tokens"
        rec = CallRecord(site=site, model=model, input_tokens=i, output_tokens=o,
                         cache_read=cr, cache_write=cw,
                         usd=price(model, i, o, cr, cw), label=label,
                         truncated=truncated)
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
