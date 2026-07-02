"""Lightweight stage profiler for the ingest/query pipeline.

Answers two questions the dashboard couldn't before: WHERE is wall-clock time going
(extraction LLM calls? embedding? PageRank? the cross-encoder?) and WHAT is incurring
cost (which call site). Timing comes from `span(label)` context managers placed at each
pipeline stage; cost attribution comes from the existing UsageMeter CallRecords grouped
by site (kg/metering.py) — this module only adds the time half.

Design constraints:
  * Zero overhead when off: `span()` is a no-op unless a Profiler has been activated
    (module-level ambient, so instrumentation never threads a profiler through the
    many pipeline signatures).
  * Thread-safe: extraction fans out across a ThreadPoolExecutor, so concurrent spans
    record into the same profiler under a lock. NB: summed thread-time for concurrent
    stages (e.g. `extract.llm`) can exceed wall-clock — that is standard profiler
    semantics and exactly what you want for "where is the work".
  * Labels are dotted and low-cardinality (stage names, never item ids) so aggregation
    stays bounded: `ingest.*`, `extract.*`, `canon.*`, `query.*`, `judge.*`.

Usage (testrun does this):
    prof = Profiler()
    activate(prof)
    try:
        ...pipeline...
        stage_times = prof.drain()   # {label: {"seconds": s, "calls": n}}
    finally:
        deactivate()
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

_lock = threading.Lock()
_active: "Profiler | None" = None


class Profiler:
    """Thread-safe {label -> (seconds, calls)} accumulator."""

    def __init__(self):
        self._lock = threading.Lock()
        self._agg: dict[str, list] = {}   # label -> [seconds, calls]

    def add(self, label: str, seconds: float) -> None:
        with self._lock:
            e = self._agg.setdefault(label, [0.0, 0])
            e[0] += seconds
            e[1] += 1

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {k: {"seconds": round(v[0], 4), "calls": v[1]}
                    for k, v in sorted(self._agg.items())}

    def drain(self) -> dict[str, dict]:
        """Snapshot and reset — used per-item so each step/query gets its own breakdown."""
        with self._lock:
            out = {k: {"seconds": round(v[0], 4), "calls": v[1]}
                   for k, v in sorted(self._agg.items())}
            self._agg.clear()
            return out


def activate(p: Profiler) -> None:
    global _active
    with _lock:
        _active = p


def deactivate() -> None:
    global _active
    with _lock:
        _active = None


@contextmanager
def span(label: str):
    """Time a pipeline stage into the active profiler; no-op when none is active."""
    p = _active
    if p is None:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        p.add(label, time.perf_counter() - t0)


def merge_profiles(into: dict, part: dict) -> dict:
    """Accumulate one drained profile into a running total (label-wise sum)."""
    for label, v in part.items():
        e = into.setdefault(label, {"seconds": 0.0, "calls": 0})
        e["seconds"] = round(e["seconds"] + v["seconds"], 4)
        e["calls"] += v["calls"]
    return into


def compact(profile: dict) -> dict:
    """{label: {"seconds":, "calls":}} -> {label: seconds} for per-item payloads
    (keeps run.json small; calls only matter on the run-level totals)."""
    return {k: v["seconds"] for k, v in profile.items()}
