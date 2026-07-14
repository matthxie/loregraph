"""Per-test isolation of the env-backed LLM provider (kg/llm_client.py).

The provider is selected from the process env (``KG_LLM`` + the matching ``*_API_KEY``)
so scattered call sites all agree without threading a handle through every function; the
Engine persists a UI-driven switch by writing those vars via ``set_active_provider``.
That write is a raw ``os.environ`` mutation, not a monkeypatch, so a test that opens an
Engine (mock/none) leaks its ``KG_LLM`` into every test that runs after it — turning the
"no provider → must raise" contract tests green-then-red purely on collection order.

This autouse fixture snapshots and restores the provider vars around each test, so the
default is what every test sees unless it sets its own. It also pins the CLI binary
overrides (``KG_CODEX_BIN`` / ``KG_CLAUDE_BIN``) to "" — "override set but not runnable"
→ the binary probe returns None — so ``detect_provider()`` (KG_LLM unset → probe codex,
claude, then API keys; fork-parity spec A1) can never find a REAL logged-in CLI on the
dev machine and flip the suite onto live subscription calls. With the CLIs masked and no
key set, detection lands on "none" and the suite stays offline/deterministic; a test
exercising the probes overrides the vars (or monkeypatches the status functions) itself.
"""
from __future__ import annotations

import os

import pytest

_PROVIDER_ENV = ("KG_LLM", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "CODEX_API_KEY",
                 "CLAUDE_API_KEY", "KG_CODEX_BIN", "KG_CLAUDE_BIN", "KG_LINK_TITLES")


@pytest.fixture(autouse=True)
def _isolate_provider_env():
    saved = {k: os.environ.get(k) for k in _PROVIDER_ENV}
    os.environ["KG_CODEX_BIN"] = ""    # mask the real CLIs (see module docstring)
    os.environ["KG_CLAUDE_BIN"] = ""
    os.environ["KG_LINK_TITLES"] = "0"  # never fetch link titles in tests (hermetic)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
