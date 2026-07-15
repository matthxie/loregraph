"""kg.llm_client — the multi-provider LLM client factory.

Every LLM-dependent call site in this package (extraction in kg/extractors.py, the
grounded answerer in kg/rag.py, the mock paths in kg/engine.py) speaks ONE dialect:
the OpenAI SDK's ``client.chat.completions.create(**kw)`` shape, reading back
``resp.choices[0].message.content`` / ``.tool_calls`` and ``resp.usage``. This module is
the single place that decides which concrete client backs that call — so the rest of the
package never imports openai/anthropic and never learns which provider is live.

The provider is selected from the process env (``KG_LLM``), because the call sites are
scattered across threads and modules that already read the env directly; persisting the
choice into the env (``set_active_provider``) is how a UI-driven switch reaches all of
them without threading a handle through every function. When ``KG_LLM`` is unset/empty,
``detect_provider()`` probes for the first live credential — subscription CLIs first
(codex, then claude), API keys second (anthropic, then openai) — so a signed-in
subscription is always preferred over metered billing without any configuration. The
first detection is pinned back into ``KG_LLM`` so the CLI auth probes (subprocesses)
run once per process, not once per LLM call.

Providers:
  * ``openai``    — the real ``openai.OpenAI()`` client, billed against ``OPENAI_API_KEY``.
  * ``codex``     — the user's ChatGPT-subscription ``codex`` CLI, driven one ``codex exec``
    per call. The billing guard is the whole point: ``codex`` bills the ChatGPT
    subscription only while NO API key is visible to it, so the child env STRIPS
    ``OPENAI_API_KEY`` / ``CODEX_API_KEY`` — a stray key must never silently flip a call
    onto metered API billing. The daemon stores no credential; ``codex login`` (interactive,
    owned by the desktop app) writes ~/.codex/auth.json and that is the only credential.
  * ``claude``    — the user's Claude-subscription ``claude`` CLI (Claude Code), driven one
    ``claude -p … --output-format json`` per call. Same billing guard, same rationale:
    the API-key vars, the ``ANTHROPIC_AUTH_TOKEN`` bearer credential, and the Bedrock/
    Vertex/Foundry reroute switches are stripped from the child env so the call can only
    bill the claude.ai login (``claude auth login``'s credential), never a stray metered
    credential or cloud endpoint.
  * ``anthropic`` — a small shim over the ``anthropic`` Messages API, when that SDK is
    installed; otherwise ``make_client`` raises ``ProviderUnavailable``.
  * ``mock``      — a deterministic, offline, network-free shim for tests and the dev loop.
  * ``none``      — no provider; ``make_client`` returns ``None`` and ``llm_available`` is False.

The codex/claude/anthropic/mock shims all duck-type the OpenAI SDK surface via
``types.SimpleNamespace`` so the call sites cannot tell them apart from the real thing.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import types
import uuid

from .errors import ProviderError, ProviderUnavailable

SUPPORTED_KINDS = ("mock", "none", "openai", "codex", "anthropic", "claude")

# Billing guard: either of these in codex's child env can route the call off the ChatGPT
# subscription onto metered API billing.
_CODEX_STRIP_ENV = ("OPENAI_API_KEY", "CODEX_API_KEY")
# GUI apps don't inherit the shell PATH; probe the usual install spots (official installer
# target first, then brew). PATH lookup covers the dev shell.
_CODEX_CANDIDATES = ("~/.local/bin/codex", "/opt/homebrew/bin/codex", "/usr/local/bin/codex")

# Billing guard for the claude CLI, same rationale as _CODEX_STRIP_ENV: any of these in
# its env routes the call off the claude.ai subscription login — the API-key vars (and the
# AUTH_TOKEN bearer credential) flip it onto metered API billing, and the USE_BEDROCK /
# USE_VERTEX / USE_FOUNDRY switches reroute it to cloud-metered endpoints entirely.
_CLAUDE_STRIP_ENV = ("ANTHROPIC_API_KEY", "CLAUDE_API_KEY", "ANTHROPIC_AUTH_TOKEN",
                     "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX",
                     "CLAUDE_CODE_USE_FOUNDRY")
_CLAUDE_CANDIDATES = ("~/.claude/local/claude", "~/.local/bin/claude",
                      "/opt/homebrew/bin/claude", "/usr/local/bin/claude")

# Per-provider default models (see default_model). The CLI kinds deliberately map to None:
# the subscription CLIs pick their own default and a foreign model id would be rejected.
_DEFAULT_MODELS = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5-20251001"}

# the answerer's OpenAI default (extraction/L3 use the provider default above)
RAG_OPENAI_DEFAULT = "gpt-5-mini"


# --------------------------------------------------------------------------- #
# Provider selection (env-backed, so scattered call sites all agree)
# --------------------------------------------------------------------------- #
def detect_provider() -> str:
    """The first live provider, probed subscription-first: a logged-in ``codex`` CLI →
    ``"codex"``; a logged-in ``claude`` CLI → ``"claude"``; ``ANTHROPIC_API_KEY`` set →
    ``"anthropic"``; ``OPENAI_API_KEY`` set → ``"openai"``; else ``"none"``. Used by
    ``current_provider()`` when ``KG_LLM`` is unset — an explicit ``KG_LLM`` always wins.
    The CLI probes are free auth checks (no model call, no tokens)."""
    if _codex_login_status()[0]:
        return "codex"
    if _claude_login_status()[0]:
        return "claude"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "none"


def default_model(kind: str | None = None) -> str | None:
    """The model a call site should default to for a provider kind: openai →
    ``gpt-4o-mini``; anthropic → ``claude-haiku-4-5-20251001``; codex/claude (and mock/none)
    → None, meaning "don't pass a model — let the CLI use its own default". ``kind=None`` →
    the active provider's kind."""
    if kind is None:
        kind = current_provider()["kind"]
    return _DEFAULT_MODELS.get((kind or "").strip().lower())


def resolve_model(configured: str | None, openai_default: str | None = None,
                  provider: dict | None = None) -> str | None:
    """The model a call site should actually request: an explicit ``configured`` value
    always wins (Config model fields default to None = "no preference"). Otherwise the
    active provider's ``default_model``, except ``openai_default`` lets a call site pick
    its own OpenAI model (the answerer wants gpt-5-mini, extraction gpt-4o-mini)."""
    if configured:
        return configured
    provider = provider or current_provider()
    kind = (provider.get("kind") or "").strip().lower()
    if kind == "openai" and openai_default:
        return openai_default
    return default_model(kind)


def current_provider() -> dict:
    """The active provider read from the process env: ``{"kind", "api_key"}``. ``kind`` from
    ``KG_LLM`` when set, else auto-detected (``detect_provider``) ONCE and pinned back into
    ``KG_LLM`` — the probe shells out to the codex/claude CLIs, and this is called on every
    LLM call site, so an unpinned detection would spawn 1-2 auth subprocesses per call (and
    a transient probe failure could flip the provider mid-run). ``set_active_provider``
    overwrites the pin. ``api_key`` from ``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` per kind
    (None for kinds that don't take one)."""
    kind = (os.environ.get("KG_LLM") or "").strip().lower()
    if not kind:
        kind = detect_provider()
        os.environ["KG_LLM"] = kind  # pin: probe once per process, not once per LLM call
    if kind == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    elif kind == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
    else:
        api_key = None
    return {"kind": kind, "api_key": api_key}


def set_active_provider(provider: dict) -> None:
    """Persist ``provider`` into the env so every downstream call site (which reads the env)
    picks it up. Sets ``KG_LLM``; for openai/anthropic a supplied ``api_key`` is written to the
    matching ``*_API_KEY`` var so the SDK finds it. Raises ValueError on an unknown kind."""
    provider = dict(provider or {})
    kind = (provider.get("kind") or "").strip().lower()
    if kind not in SUPPORTED_KINDS:
        raise ValueError(f"unknown provider kind {kind!r} (supported: {SUPPORTED_KINDS})")
    os.environ["KG_LLM"] = kind
    api_key = provider.get("api_key")
    if kind == "openai" and api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    elif kind == "anthropic" and api_key:
        os.environ["ANTHROPIC_API_KEY"] = api_key


def make_client(provider: dict | None = None):
    """An OpenAI-SDK-shaped client (exposes ``.chat.completions.create(**kw)``) for the active
    provider, or ``None`` when kind == ``none``. ``provider=None`` → ``current_provider()``. The
    openai/anthropic SDKs are imported lazily here so ``import kg`` stays light and offline."""
    provider = provider or current_provider()
    kind = (provider.get("kind") or "").strip().lower()
    if kind not in SUPPORTED_KINDS:
        raise ProviderUnavailable(
            f"provider kind {kind!r} not supported (supported: {SUPPORTED_KINDS})")
    if kind == "none":
        return None
    if kind == "mock":
        return MockChatClient()
    if kind == "codex":
        return CodexClient()
    if kind == "claude":
        return ClaudeClient()
    if kind == "openai":
        try:
            import openai
        except ImportError as e:
            raise ProviderUnavailable(f"openai sdk not installed: {e}")
        return openai.OpenAI()  # reads OPENAI_API_KEY
    # anthropic
    try:
        import anthropic
    except ImportError:
        raise ProviderUnavailable("anthropic sdk not installed")
    return AnthropicClient(anthropic.Anthropic())


def llm_available(provider: dict | None = None) -> bool:
    """Whether the active provider can actually serve a call right now: mock → always; none →
    never; openai/anthropic → the matching API key is present; codex/claude → the CLI reports
    a connected subscription login (a free auth probe, no model call)."""
    provider = provider or current_provider()
    kind = (provider.get("kind") or "").strip().lower()
    if kind == "mock":
        return True
    if kind == "none":
        return False
    if kind == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    if kind == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    if kind == "codex":
        return _codex_login_status()[0]
    if kind == "claude":
        return _claude_login_status()[0]
    return False


def provider_status(provider: dict | None = None) -> dict:
    """``{"kind", "connected", "detail"}`` for the active provider. codex probes
    ``codex login status``, claude probes ``claude auth status``; the key-based providers
    report whether their key is set."""
    provider = provider or current_provider()
    kind = (provider.get("kind") or "").strip().lower()
    if kind == "mock":
        return {"kind": kind, "connected": True, "detail": "mock provider (offline)"}
    if kind == "none":
        return {"kind": kind, "connected": False, "detail": "no credentials"}
    if kind == "openai":
        ok = bool(os.environ.get("OPENAI_API_KEY"))
        return {"kind": kind, "connected": ok,
                "detail": "OPENAI_API_KEY set" if ok else "OPENAI_API_KEY missing"}
    if kind == "anthropic":
        ok = bool(os.environ.get("ANTHROPIC_API_KEY"))
        return {"kind": kind, "connected": ok,
                "detail": "ANTHROPIC_API_KEY set" if ok else "ANTHROPIC_API_KEY missing"}
    if kind == "codex":
        ok, detail = _codex_login_status()
        return {"kind": kind, "connected": ok, "detail": detail}
    if kind == "claude":
        ok, detail = _claude_login_status()
        return {"kind": kind, "connected": ok, "detail": detail}
    return {"kind": kind, "connected": False, "detail": f"unknown kind {kind!r}"}


def provider_signout(kind: str) -> dict:
    """Sign the provider out. codex → ``codex logout``, claude → ``claude auth logout``
    (both best-effort); every other kind is stateless here → ``{"ok": True}``."""
    kind = (kind or "").strip().lower()
    if kind == "codex":
        return _codex_logout()
    if kind == "claude":
        return _claude_logout()
    return {"ok": True}


def provider_usage(kind: str) -> dict:
    """A plan-usage snapshot. codex → its rate-limit windows read for free (no model call);
    every other kind reports no meter here → ``{}``."""
    kind = (kind or "").strip().lower()
    if kind == "codex":
        return _codex_usage()
    return {}


# --------------------------------------------------------------------------- #
# SDK-shaped response builders — what every shim returns so call sites can't tell
# a shim from the real openai client (see _MockAnswerClient in kg/engine.py).
# --------------------------------------------------------------------------- #
def _sdk_response(*, content: str | None, tool_calls: list | None,
                  finish_reason: str, prompt_tokens: int = 0,
                  completion_tokens: int = 0):
    """One ``chat.completions.create`` return value, duck-typed to the OpenAI SDK: readable as
    ``resp.choices[0].message.content`` / ``.tool_calls`` / ``.finish_reason`` and
    ``resp.usage.prompt_tokens`` / ``.completion_tokens``."""
    message = types.SimpleNamespace(content=content, tool_calls=tool_calls or None)
    choice = types.SimpleNamespace(message=message, finish_reason=finish_reason)
    usage = types.SimpleNamespace(prompt_tokens=int(prompt_tokens),
                                  completion_tokens=int(completion_tokens))
    return types.SimpleNamespace(choices=[choice], usage=usage)


def _tool_call(name: str, arguments: str, call_id: str = "call_0"):
    """One ``message.tool_calls[0]`` entry: ``tc.id`` and ``tc.function.name`` /
    ``tc.function.arguments`` (a JSON string), exactly as the call sites read them."""
    return types.SimpleNamespace(
        id=call_id,
        function=types.SimpleNamespace(name=name, arguments=arguments))


def _forced_tool_name(kw: dict) -> str | None:
    """The function name a ``tool_choice`` forces, or the sole tool's name when tools are
    present without ANY explicit choice; None when the call wants free-form content. An
    explicit non-forcing choice ("auto"/"none", or {"type": "auto"}) means the model picks
    freely — required for the multi-tool agent loop, where forcing tools[0] every turn
    would wedge the run on the first tool."""
    choice = kw.get("tool_choice")
    if isinstance(choice, dict):
        if choice.get("type") == "function":
            name = choice.get("function", {}).get("name")
            if name:
                return name
        return None
    if isinstance(choice, str):
        return None
    tools = kw.get("tools") or []
    if tools:
        try:
            return tools[0]["function"]["name"]
        except (KeyError, TypeError, IndexError):
            return None
    return None


# --------------------------------------------------------------------------- #
# Message flattening / JSON salvage — shared by the text-only shims (codex, claude)
# --------------------------------------------------------------------------- #
def _flatten_messages(messages: list) -> str:
    """Join an OpenAI ``messages`` array into one prompt string: system+user text in order,
    with each ``image_url`` block replaced by a note (the CLI shims are text-only, an
    accepted v0 limitation)."""
    parts: list[str] = []
    for m in messages or []:
        content = m.get("content") if isinstance(m, dict) else None
        if isinstance(content, str):
            if content.strip():
                parts.append(content)
            continue
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text" and block.get("text", "").strip():
                    parts.append(block["text"])
                elif block.get("type") == "image_url":
                    parts.append("[image omitted: CLI provider is text-only]")
    return "\n\n".join(parts)


def _strip_fences(s: str) -> str:
    """Some models wrap JSON in a ```json fence; peel it so the balanced-brace scan lands
    on the object."""
    t = (s or "").strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t[3:]
        t = t.strip()
        if t.endswith("```"):
            t = t[:-3].strip()
    return t


def _first_json_object(raw: str) -> str | None:
    """The first balanced ``{...}`` span in ``raw`` that parses as a JSON object, tolerating
    code fences and surrounding prose — the reply from a tool-less CLI that was merely *asked*
    to answer in JSON. Returns the JSON substring, or None."""
    t = _strip_fences(raw)
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, ch in enumerate(t):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    span = t[start:i + 1]
                    try:
                        json.loads(span)
                        return span
                    except json.JSONDecodeError:
                        start = -1
    return None


# --------------------------------------------------------------------------- #
# MockChatClient — deterministic offline shim (reusable lift of _MockAnswerClient)
# --------------------------------------------------------------------------- #
class MockChatClient:
    """OpenAI-SDK-shaped, deterministic, network-free. When a tool is forced it returns one
    canned tool call under that tool's name with an empty payload (so RagAnswerer /
    the extractor run their tool-call path end-to-end); otherwise it returns a canned content
    string. Lifted from ``_MockAnswerClient`` in kg/engine.py so engine.py can import it here."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        name = _forced_tool_name(kw)
        if name:
            return _sdk_response(
                content=None,
                tool_calls=[_tool_call(name, json.dumps({}))],
                finish_reason="tool_calls")
        return _sdk_response(
            content="(mock provider) canned answer over the retrieved context.",
            tool_calls=None, finish_reason="stop")


# --------------------------------------------------------------------------- #
# CodexClient — one guarded `codex exec` per call, billed to the ChatGPT subscription
# --------------------------------------------------------------------------- #
def _codex_binary() -> str | None:
    """Locate the codex CLI. ``KG_CODEX_BIN`` set ⇒ use it verbatim (or None if it isn't a
    runnable file — never fall through to the probe). Else PATH ``codex``, else the usual
    install spots (GUI apps don't inherit the shell PATH)."""
    override = os.environ.get("KG_CODEX_BIN")
    if override is not None:
        return override if (override and os.path.isfile(override)
                            and os.access(override, os.X_OK)) else None
    hit = shutil.which("codex")
    if hit:
        return hit
    for c in _CODEX_CANDIDATES:
        p = os.path.expanduser(c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _codex_child_env() -> dict:
    """codex's child env with the API-key vars stripped, forcing subscription auth
    (~/.codex/auth.json is then the only credential in play)."""
    env = dict(os.environ)
    for k in _CODEX_STRIP_ENV:
        env.pop(k, None)
    return env


def _codex_login_status() -> tuple[bool, str]:
    """Free auth check (``codex login status``): no model call, no tokens burned. Returns
    ``(connected, detail)``."""
    bin_ = _codex_binary()
    if not bin_:
        return False, "codex CLI not installed"
    try:
        p = subprocess.run([bin_, "login", "status"], capture_output=True, text=True,
                           env=_codex_child_env(), timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"codex login status failed: {e}"
    text = (p.stdout + p.stderr).strip()
    low = text.lower()
    ok = p.returncode == 0 and "logged in" in low and "not logged in" not in low
    return ok, text or ("connected" if ok else "not logged in")


def _codex_logout() -> dict:
    """``codex logout`` — clears ~/.codex/auth.json, best-effort (exit status isn't surfaced).
    ``{"ok": False, "detail": ...}`` only when the binary is absent."""
    bin_ = _codex_binary()
    if not bin_:
        return {"ok": False, "detail": "codex CLI not installed"}
    try:
        subprocess.run([bin_, "logout"], capture_output=True, text=True,
                       env=_codex_child_env(), timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"ok": True}


def _codex_usage() -> dict:
    """Best-effort plan-usage snapshot from the session cache written after real codex calls.
    Fails soft to ``{"available": False, "plan": None, "windows": []}`` — the daemon never
    surfaces an error envelope for usage."""
    unavailable = {"available": False, "plan": None, "windows": []}
    root = os.path.join(os.path.expanduser(os.environ.get("CODEX_HOME", "~/.codex")),
                        "sessions")
    if not os.path.isdir(root):
        return unavailable
    files: list[str] = []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            if n.endswith(".jsonl"):
                files.append(os.path.join(dirpath, n))
    try:
        files.sort(key=os.path.getmtime, reverse=True)
    except OSError:
        return unavailable

    def window(d):
        if not isinstance(d, dict):
            return None
        used = d.get("used_percent")
        if not isinstance(used, (int, float)) or isinstance(used, bool):
            return None
        return {"used_percent": int(used), "resets_at": d.get("resets_at"),
                "minutes": d.get("window_minutes")}

    for path in files[:50]:
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - (2 << 20)))
                lines = f.read().decode("utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for line in reversed(lines):
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rate_limits = obj.get("rate_limits") if isinstance(obj, dict) else None
            if not isinstance(rate_limits, dict):
                continue
            windows = [w for w in (window(rate_limits.get("primary")),
                                   window(rate_limits.get("secondary"))) if w]
            if windows:
                return {"available": True, "plan": None, "windows": windows}
    return unavailable


class CodexClient:
    """OpenAI-SDK-shaped client backed by ``codex exec`` — one guarded child per call, billed
    to the user's ChatGPT subscription (the API-key vars are stripped from the child env). Text
    only: images in the messages become placeholders. A forced tool becomes an instruction to
    reply with ONLY the JSON matching the tool's schema, and the first balanced JSON object in
    the reply is re-wrapped as a tool call so the call sites' tool-call path runs unchanged."""

    EXEC_TIMEOUT = 300

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        prompt = _flatten_messages(kw.get("messages") or [])
        name = _forced_tool_name(kw)
        schema = None
        if kw.get("tools"):
            try:
                schema = kw["tools"][0]["function"]["parameters"]
            except (KeyError, TypeError, IndexError):
                schema = None
        elif kw.get("response_format"):
            schema = kw.get("response_format")
        if name or schema is not None:
            instruction = ("Respond with ONLY a single JSON object matching this schema, "
                           "no prose, no code fences:\n"
                           + json.dumps(schema if schema is not None else {}))
            prompt = f"{prompt}\n\n{instruction}" if prompt else instruction
        text = self._exec(prompt, timeout=kw.get("timeout"))
        if name:
            span = _first_json_object(text)
            arguments = span if span is not None else "{}"
            return _sdk_response(
                content=None,
                tool_calls=[_tool_call(name, arguments)],
                finish_reason="tool_calls")
        return _sdk_response(content=text, tool_calls=None, finish_reason="stop")

    def _exec(self, prompt: str, timeout: float | None = None) -> str:
        """Run one ``codex exec``, capturing the final assistant message. Raises
        ``ProviderUnavailable`` when the CLI is missing / not logged in and ``ProviderError`` on
        a non-zero exit or empty output. ``timeout`` tightens (never widens) the per-exec
        cap — the agent loop passes its remaining budget so one call can't blow the run's
        deadline (§9.2)."""
        cap = min(float(timeout), self.EXEC_TIMEOUT) if timeout else self.EXEC_TIMEOUT
        bin_ = _codex_binary()
        if not bin_:
            raise ProviderUnavailable(
                "the Codex CLI isn't installed — connect ChatGPT in Settings first")
        call_dir = os.path.join(tempfile.gettempdir(), "kg-codex-cwd", uuid.uuid4().hex)
        os.makedirs(call_dir, exist_ok=True)
        out_file = os.path.join(call_dir, "last-message.txt")
        cmd = [bin_, "exec", "--skip-git-repo-check", "--ephemeral", "--ignore-user-config",
               "-s", "read-only", "--color", "never", "-C", call_dir, "-o", out_file, "-"]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  env=_codex_child_env(), cwd=call_dir,
                                  timeout=cap)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"codex exec timed out after {cap:.0f}s")
        except OSError as e:
            raise ProviderUnavailable(f"could not launch codex: {e}")
        text = ""
        try:
            if os.path.isfile(out_file):
                with open(out_file, encoding="utf-8") as f:
                    text = f.read().strip()
            if not text:
                text = (proc.stdout or "").strip()
        finally:
            shutil.rmtree(call_dir, ignore_errors=True)
        if text:
            return text
        tail = (proc.stderr or "").strip()[-300:]
        low = tail.lower()
        if "not logged in" in low or "codex login" in low:
            raise ProviderUnavailable(
                "ChatGPT isn't connected — run `codex login` (or connect in Settings), "
                "then retry")
        raise ProviderError(
            f"codex exec exited {proc.returncode} with no result"
            + (f": {tail}" if tail else ""))


# --------------------------------------------------------------------------- #
# ClaudeClient — one guarded `claude -p` per call, billed to the Claude subscription
# --------------------------------------------------------------------------- #
def _claude_binary() -> str | None:
    """Locate the claude CLI (Claude Code). ``KG_CLAUDE_BIN`` set ⇒ use it verbatim (or None
    if it isn't a runnable file — never fall through to the probe). Else PATH ``claude``, else
    the usual install spots (GUI apps don't inherit the shell PATH)."""
    override = os.environ.get("KG_CLAUDE_BIN")
    if override is not None:
        return override if (override and os.path.isfile(override)
                            and os.access(override, os.X_OK)) else None
    hit = shutil.which("claude")
    if hit:
        return hit
    for c in _CLAUDE_CANDIDATES:
        p = os.path.expanduser(c)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    return None


def _claude_child_env() -> dict:
    """claude's child env with the API-key vars stripped, forcing subscription auth (the
    claude.ai login from ``claude auth login`` is then the only credential in play)."""
    env = dict(os.environ)
    for k in _CLAUDE_STRIP_ENV:
        env.pop(k, None)
    return env


def _claude_login_status() -> tuple[bool, str]:
    """Free auth check (``claude auth status``, JSON by default): no model call, no tokens
    burned. With the API keys stripped, ``loggedIn: true`` can only mean the claude.ai
    subscription login. Returns ``(connected, detail)``."""
    bin_ = _claude_binary()
    if not bin_:
        return False, "claude CLI not installed"
    try:
        p = subprocess.run([bin_, "auth", "status", "--json"], capture_output=True,
                           text=True, env=_claude_child_env(), timeout=20)
    except (OSError, subprocess.TimeoutExpired) as e:
        return False, f"claude auth status failed: {e}"
    text = (p.stdout or "").strip()
    try:
        status = json.loads(text)
    except json.JSONDecodeError:
        status = {}
    ok = p.returncode == 0 and isinstance(status, dict) and bool(status.get("loggedIn"))
    if ok:
        detail = " ".join(filter(None, ["logged in", status.get("email"),
                                        status.get("subscriptionType")]))
    else:
        detail = text or (p.stderr or "").strip() or "not logged in"
    return ok, detail


def _claude_logout() -> dict:
    """``claude auth logout`` — clears the stored claude.ai credential, best-effort (exit
    status isn't surfaced). ``{"ok": False, "detail": ...}`` only when the binary is absent."""
    bin_ = _claude_binary()
    if not bin_:
        return {"ok": False, "detail": "claude CLI not installed"}
    try:
        subprocess.run([bin_, "auth", "logout"], capture_output=True, text=True,
                       env=_claude_child_env(), timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return {"ok": True}


class ClaudeClient:
    """OpenAI-SDK-shaped client backed by ``claude -p`` (Claude Code, non-interactive) — one
    guarded child per call, billed to the user's Claude subscription (the API-key vars are
    stripped from the child env). Text only: images in the messages become placeholders. A
    forced tool becomes an instruction to reply with ONLY the JSON matching the tool's
    schema, and the first balanced JSON object in the reply is re-wrapped as a tool call so
    the call sites' tool-call path runs unchanged (the CodexClient recipe verbatim).

    Flags (verified against ``claude --help``): ``-p`` prints one response and exits, reading
    the prompt from stdin; ``--output-format json`` wraps the run in a machine-readable
    envelope whose ``result`` field is the final assistant text; ``--tools ""`` disables all
    agent tools; ``--no-session-persistence`` keeps the session off disk; ``--safe-mode``
    (the ``--ignore-user-config`` analog) skips the user's CLAUDE.md/plugins/hooks/MCP
    servers while leaving auth untouched."""

    EXEC_TIMEOUT = 300

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        prompt = _flatten_messages(kw.get("messages") or [])
        name = _forced_tool_name(kw)
        schema = None
        if kw.get("tools"):
            try:
                schema = kw["tools"][0]["function"]["parameters"]
            except (KeyError, TypeError, IndexError):
                schema = None
        elif kw.get("response_format"):
            schema = kw.get("response_format")
        if name or schema is not None:
            instruction = ("Respond with ONLY a single JSON object matching this schema, "
                           "no prose, no code fences:\n"
                           + json.dumps(schema if schema is not None else {}))
            prompt = f"{prompt}\n\n{instruction}" if prompt else instruction
        text = self._exec(prompt, model=kw.get("model"), timeout=kw.get("timeout"))
        if name:
            span = _first_json_object(text)
            arguments = span if span is not None else "{}"
            return _sdk_response(
                content=None,
                tool_calls=[_tool_call(name, arguments)],
                finish_reason="tool_calls")
        return _sdk_response(content=text, tool_calls=None, finish_reason="stop")

    def _exec(self, prompt: str, model: str | None = None,
              timeout: float | None = None) -> str:
        """Run one ``claude -p``, reading the final assistant message out of the JSON
        envelope. Raises ``ProviderUnavailable`` when the CLI is missing / not logged in and
        ``ProviderError`` on a non-zero exit, an error result, or empty output. ``timeout``
        tightens (never widens) the per-exec cap — the agent loop passes its remaining
        budget so one call can't blow the run's deadline (§9.2)."""
        cap = min(float(timeout), self.EXEC_TIMEOUT) if timeout else self.EXEC_TIMEOUT
        bin_ = _claude_binary()
        if not bin_:
            raise ProviderUnavailable(
                "the Claude CLI isn't installed — connect Claude in Settings first")
        call_dir = os.path.join(tempfile.gettempdir(), "kg-claude-cwd", uuid.uuid4().hex)
        os.makedirs(call_dir, exist_ok=True)
        cmd = [bin_, "-p", "--output-format", "json", "--tools", "",
               "--no-session-persistence", "--safe-mode"]
        # Only forward a model the claude CLI can serve: full ids and the CLI's own aliases
        # (sonnet/opus/haiku). A foreign id (a Config still holding an openai default) would
        # be rejected, so anything else falls to the CLI's default.
        if model and (model in ("sonnet", "opus", "haiku") or model.startswith("claude")):
            cmd += ["--model", model]
        try:
            proc = subprocess.run(cmd, input=prompt, capture_output=True, text=True,
                                  env=_claude_child_env(), cwd=call_dir,
                                  timeout=cap)
        except subprocess.TimeoutExpired:
            raise ProviderError(f"claude -p timed out after {cap:.0f}s")
        except OSError as e:
            raise ProviderUnavailable(f"could not launch claude: {e}")
        finally:
            shutil.rmtree(call_dir, ignore_errors=True)
        text = self._result_text((proc.stdout or "").strip())
        if text:
            return text
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        low = tail.lower()
        if "not logged in" in low or "auth login" in low or "/login" in low:
            raise ProviderUnavailable(
                "Claude isn't connected — run `claude auth login` (or connect in Settings), "
                "then retry")
        raise ProviderError(
            f"claude -p exited {proc.returncode} with no result"
            + (f": {tail}" if tail else ""))

    @staticmethod
    def _result_text(stdout: str) -> str:
        """The final assistant text from ``--output-format json``: either one result object
        or (newer CLIs) an array of run events whose last ``type == "result"`` entry carries
        it. Unparseable output degrades to the raw stdout so a plain-text reply still lands."""
        if not stdout:
            return ""
        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return stdout
        result = None
        if isinstance(payload, dict) and payload.get("type") == "result":
            result = payload
        elif isinstance(payload, list):
            for entry in reversed(payload):
                if isinstance(entry, dict) and entry.get("type") == "result":
                    result = entry
                    break
        if result is None:
            return stdout
        if result.get("is_error"):
            raise ProviderError(
                f"claude -p returned an error result: {str(result.get('result'))[:300]}")
        return (result.get("result") or "").strip()


# --------------------------------------------------------------------------- #
# AnthropicClient — small translation to the anthropic Messages API
# --------------------------------------------------------------------------- #
class AnthropicClient:
    """OpenAI-SDK-shaped client translating ``chat.completions.create`` to the anthropic
    Messages API and back. Supports both paths: a forced ``tool_choice`` maps to an anthropic
    ``tool``-choice and the returned ``tool_use`` block becomes a ``message.tool_calls`` entry;
    otherwise the text blocks become ``message.content``."""

    def __init__(self, client):
        self._client = client
        self.chat = self
        self.completions = self

    def create(self, **kw):
        system, messages = self._split_messages(kw.get("messages") or [])
        params: dict = {
            "model": kw.get("model"),
            "max_tokens": kw.get("max_tokens") or kw.get("max_completion_tokens") or 1024,
            "messages": messages,
        }
        if system:
            params["system"] = system
        if kw.get("temperature") is not None:
            params["temperature"] = kw["temperature"]
        if kw.get("timeout"):
            params["timeout"] = kw["timeout"]
        tools = self._translate_tools(kw.get("tools"))
        if tools:
            params["tools"] = tools
        forced = _forced_tool_name(kw)
        if forced:
            params["tool_choice"] = {"type": "tool", "name": forced}
        resp = self._client.messages.create(**params)
        return self._translate_response(resp)

    @staticmethod
    def _split_messages(messages: list) -> tuple[str, list]:
        """Anthropic takes the system prompt as a top-level string, not a message; pull system
        turns out and translate the rest, mapping ``image_url`` data-URL blocks to anthropic
        image blocks (other image_urls become a text note — anthropic wants inline/base64).
        Multi-turn tool history round-trips too: an assistant message's ``tool_calls`` become
        ``tool_use`` blocks and a ``role:"tool"`` result becomes a user-turn ``tool_result``
        block referencing its ``tool_call_id`` — without these the agent loop's transcript
        would silently desync. Consecutive same-role turns merge (the API wants alternation)."""
        system_parts: list[str] = []
        out: list[dict] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                continue
            if role == "tool":
                result = content if isinstance(content, str) else json.dumps(content or "")
                out.append({"role": "user", "content": [
                    {"type": "tool_result",
                     "tool_use_id": m.get("tool_call_id") or "call_0",
                     "content": result}]})
                continue
            blocks: list[dict] = []
            if isinstance(content, str):
                if content:
                    blocks.append({"type": "text", "text": content})
            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "text":
                        blocks.append({"type": "text", "text": block.get("text", "")})
                    elif block.get("type") == "image_url":
                        url = block.get("image_url", {}).get("url", "")
                        img = AnthropicClient._image_block(url)
                        blocks.append(img)
            if role == "assistant":
                for tc in m.get("tool_calls") or []:
                    fn = (tc.get("function", {}) if isinstance(tc, dict)
                          else getattr(tc, "function", None))
                    name = (fn.get("name") if isinstance(fn, dict)
                            else getattr(fn, "name", "")) or ""
                    raw_args = (fn.get("arguments") if isinstance(fn, dict)
                                else getattr(fn, "arguments", "")) or "{}"
                    try:
                        args = json.loads(raw_args)
                    except (TypeError, json.JSONDecodeError):
                        args = {}
                    tc_id = (tc.get("id") if isinstance(tc, dict)
                             else getattr(tc, "id", None)) or "call_0"
                    blocks.append({"type": "tool_use", "id": tc_id,
                                   "name": name, "input": args})
            if not blocks:
                continue
            out.append({"role": "assistant" if role == "assistant" else "user",
                        "content": blocks})
        merged: list[dict] = []
        for m in out:
            if merged and merged[-1]["role"] == m["role"]:
                merged[-1]["content"].extend(m["content"])
            else:
                merged.append(m)
        return "\n\n".join(system_parts), merged

    @staticmethod
    def _image_block(url: str) -> dict:
        """A data: URL becomes an anthropic base64 image block; anything else (a remote URL)
        degrades to a text note, since this shim doesn't fetch."""
        if url.startswith("data:") and "," in url and ";base64," in url:
            header, data = url.split(",", 1)
            media_type = header[len("data:"):].split(";", 1)[0] or "image/png"
            return {"type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": data}}
        return {"type": "text", "text": "[image omitted: unsupported image url]"}

    @staticmethod
    def _translate_tools(tools) -> list:
        """OpenAI function tools → anthropic tools (name/description/input_schema)."""
        out = []
        for t in tools or []:
            fn = t.get("function", {}) if isinstance(t, dict) else {}
            if not fn.get("name"):
                continue
            out.append({"name": fn["name"], "description": fn.get("description", ""),
                        "input_schema": fn.get("parameters", {"type": "object"})})
        return out

    @staticmethod
    def _translate_response(resp):
        """Anthropic Messages reply → OpenAI-SDK shape: text blocks joined into
        ``message.content``, ``tool_use`` blocks into ``message.tool_calls``; ``stop_reason``
        mapped to ``finish_reason``."""
        text_parts: list[str] = []
        tool_calls: list = []
        for block in getattr(resp, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                text_parts.append(getattr(block, "text", ""))
            elif btype == "tool_use":
                tool_calls.append(_tool_call(
                    getattr(block, "name", ""),
                    json.dumps(getattr(block, "input", {}) or {}),
                    call_id=getattr(block, "id", "call_0")))
        stop = getattr(resp, "stop_reason", None)
        finish = {"tool_use": "tool_calls", "max_tokens": "length",
                  "end_turn": "stop", "stop_sequence": "stop"}.get(stop, "stop")
        usage = getattr(resp, "usage", None)
        prompt_tokens = int(getattr(usage, "input_tokens", 0) or 0) if usage else 0
        completion_tokens = int(getattr(usage, "output_tokens", 0) or 0) if usage else 0
        return _sdk_response(
            content="".join(text_parts) or None,
            tool_calls=tool_calls or None,
            finish_reason=finish,
            prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
