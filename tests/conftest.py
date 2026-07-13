"""Per-test isolation of the env-backed LLM provider (kg/llm_client.py).

The provider is selected from the process env (``KG_LLM`` + the matching ``*_API_KEY``)
so scattered call sites all agree without threading a handle through every function; the
Engine persists a UI-driven switch by writing those vars via ``set_active_provider``.
That write is a raw ``os.environ`` mutation, not a monkeypatch, so a test that opens an
Engine (mock/none) leaks its ``KG_LLM`` into every test that runs after it — turning the
"no provider → must raise" contract tests green-then-red purely on collection order.

This autouse fixture snapshots and restores the provider vars around each test, so the
default (``KG_LLM`` unset → openai, keyed off ``OPENAI_API_KEY``) is what every test sees
unless it sets its own.
"""
from __future__ import annotations

import os

import pytest

_PROVIDER_ENV = ("KG_LLM", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY")


@pytest.fixture(autouse=True)
def _isolate_provider_env():
    saved = {k: os.environ.get(k) for k in _PROVIDER_ENV}
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
