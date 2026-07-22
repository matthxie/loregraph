"""Time normalization shared by the mappers + normalize.

Every source stamps time differently — ChatGPT uses float Unix seconds, Claude and Gemini
use ISO-8601 strings (with varying sub-second/offset shapes). We funnel them all to one
ISO seconds-resolution UTC string matching kg.store.now_iso(), so the bi-temporal layer
orders imported facts by real chat time exactly like a live capture.
"""
from __future__ import annotations

from datetime import datetime, timezone


def to_iso(value: object) -> str | None:
    """A source timestamp → 'YYYY-MM-DDTHH:MM:SS+00:00' (UTC, seconds), or None when it is
    absent/unparseable (the caller inherits a fallback time rather than crashing)."""
    dt = to_datetime(value)
    return dt.isoformat(timespec="seconds") if dt else None


def to_datetime(value: object) -> datetime | None:
    """Parse a Unix-epoch number OR an ISO-8601 string to an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        # numeric string epoch (some Gemini/ChatGPT variants)
        try:
            return datetime.fromtimestamp(float(s), tz=timezone.utc)
        except (ValueError, OverflowError, OSError):
            pass
        s = s.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
        except ValueError:
            return None
        return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    return None


def header_date(dt: datetime) -> str:
    """A datetime → the '[chat session — …]' header date 'YYYY/MM/DD (Day) HH:MM' the
    turns chunker + LongMemEval renderer use (see scripts/build_longmemeval.render_session)."""
    return dt.strftime("%Y/%m/%d (%a) %H:%M")
