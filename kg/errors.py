"""Engine error taxonomy — the ONE exception family allowed to escape a public
Engine method (Engine Interface Contract §7). The app-side router maps these onto
its JSON-RPC error codes; anything else escaping a public method is a bug, so the
Engine facade wraps unexpected exceptions into the base EngineError.
"""
from __future__ import annotations


class EngineError(Exception):
    """Base — router maps to INTERNAL_ERROR."""


class InvalidInput(EngineError):
    """Bad parameter shape/value from the caller."""


class NotFound(EngineError):
    """Unknown episode / entity id."""


class StoreError(EngineError):
    """Persistence layer failed (corrupt/unwritable store)."""


class ProviderUnavailable(EngineError):
    """No LLM provider configured/authenticated for an LLM-dependent call."""


class ProviderError(EngineError):
    """The provider ran but failed (API error, bad response)."""


class ModelUnavailable(EngineError):
    """Local model weights missing or corrupt."""
