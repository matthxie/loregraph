"""Agentic answering (PROTOCOL §9.2) — a bounded tool loop over the engine facade.

The provider LLM traverses the graph by *calling* the same read verbs the daemon
serves (search, retrieve, facts, graph_preview, episode) — one implementation per
verb, no second retrieval path (§8.10 parity-firewall spirit). Two drivers:

  * native   — openai/anthropic (and the mock client): real multi-turn tool calling
               over the OpenAI-SDK surface every kg.llm_client shim duck-types.
  * reduced  — the codex/claude CLI shims are one-shot subprocesses with no tool
               support, so the loop replays the transcript as text each turn and asks
               for ONE JSON object: a tool call or the final answer (§9.4).

Budget: the client applies a hard 600 s ceiling to chat.agent (§9.2), so the loop
keeps its own deadline and forces the final answer while there is still time for one
more provider call. `agent.progress` narration is emitted through the `progress`
callback; it never extends the budget.

Citations are validated against the WIDENED universe (§9.2): every episode id that
visibly appeared in any tool result during the run — not just the final context.
Invalid ids are dropped from `citations` and stripped from the answer text (§3.12).

Web tools are provider-native only; no provider shim exposes them yet, so the loop
runs graph-only — per §9.2 that is silent degradation, never an error.
"""
from __future__ import annotations

import json
import time

from .errors import EngineError, ProviderError
from .rag import _EP_ID, _validate, strip_citations

MAX_STEPS = 16                  # daemon-side clamp on the client's max_steps (§9.2)
MAX_STEPS_REDUCED = 6           # CLI shims: each turn is a full subprocess re-send
_DETAIL_CHARS = 200             # §9.3: progress detail / trace input_summary cap
_TOOL_RESULT_CHARS = 4000       # keep tool results prompt-sized

_SYSTEM = """You are a memory assistant answering ONE question over a personal \
knowledge graph of the user's notes ("episodes"), entities, and bi-temporal facts.

Use the tools to gather evidence, then call submit_answer exactly once. Rules:
- Cite evidence inline as bracketed episode ids, e.g. [ep_note_0003]. Cite ONLY \
episode ids you saw in tool results this run — never invent ids.
- If two pieces of evidence conflict, surface BOTH sides with their episode ids; \
never silently pick one.
- Prefer graph_search for exact names/phrases/file types, graph_retrieve for \
semantic questions, graph_facts for one entity's relationships over time, \
graph_neighbors to walk from a node, graph_episode to read one note verbatim.
- Be concise. When the graph has no evidence, say so plainly."""


def _tool(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function",
            "function": {"name": name, "description": description,
                         "parameters": {"type": "object", "properties": props,
                                        "required": required}}}


_GRAPH_TOOLS = [
    _tool("graph_search", "Keyword/BM25 lookup: exact names, phrases, file types.",
          {"terms": {"type": "string"}, "k": {"type": "integer"}}, ["terms"]),
    _tool("graph_retrieve", "Semantic retrieval (PPR) with assembled evidence.",
          {"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
    _tool("graph_facts", "One entity's bi-temporal facts (name or node id).",
          {"entity": {"type": "string"}, "include_closed": {"type": "boolean"}},
          ["entity"]),
    _tool("graph_neighbors", "One-hop graph around an episode/entity/concept id.",
          {"id": {"type": "string"}}, ["id"]),
    _tool("graph_episode", "Read one note verbatim by episode id.",
          {"id": {"type": "string"}}, ["id"]),
]

_SUBMIT_TOOL = _tool(
    "submit_answer", "Submit the final answer with its episode-id citations.",
    {"answer": {"type": "string"},
     "citations": {"type": "array", "items": {"type": "string"}}},
    ["answer", "citations"])


class _Evidence:
    """The widened citation universe (§9.2): every episode id that visibly appeared
    in any tool result, in first-appearance order, plus the rendered fact lines the
    loop touched (they become the response's context.facts).

    Ids scraped from rendered fact TEXT are admitted only if `is_episode` confirms
    they resolve to a real, live episode node: fact lines embed extractor-derived
    entity names, so a hostile note can mint an entity that LOOKS like an episode id
    ("ep_fake_evil") — without the check, that string would enter the universe and
    the gate would bless a citation of nonexistent evidence."""

    def __init__(self, is_episode):
        self._is_episode = is_episode
        self.episode_ids: list[str] = []
        self._seen: set[str] = set()
        self.facts: list[str] = []
        self._fact_seen: set[str] = set()

    def add_ids(self, *ids: str) -> None:
        for eid in ids:
            if eid and isinstance(eid, str) and eid not in self._seen:
                self._seen.add(eid)
                self.episode_ids.append(eid)

    def add_fact(self, rendered: str) -> None:
        if rendered and rendered not in self._fact_seen:
            self._fact_seen.add(rendered)
            self.facts.append(rendered)
            self.add_ids(*(eid for eid in _EP_ID.findall(rendered)
                           if self._is_episode(eid)))

    def add_row(self, row: dict) -> None:
        """A structured §3 Fact row from an engine verb: kept whole (the daemon serves
        it verbatim as a wire Fact), deduped on its rendered line; its grounding
        episode id joins the universe when it resolves to a live episode."""
        rendered = row.get("rendered") or ""
        if not rendered or rendered in self._fact_seen:
            return
        self._fact_seen.add(rendered)
        self.facts.append(row)
        eid = row.get("episode_id")
        if eid and isinstance(eid, str) and self._is_episode(eid):
            self.add_ids(eid)
        self.add_ids(*(i for i in _EP_ID.findall(rendered)
                       if self._is_episode(i)))

    @property
    def universe(self) -> list[str]:
        return self.episode_ids


def _clip(text: str, n: int) -> str:
    flat = " ".join((text or "").split())
    return flat[:n]


def _int_or(v, default: int, lo: int = 1, hi: int = 50) -> int:
    """A model-supplied integer argument, defensively coerced and clamped — the
    provider may emit "a few", 3.7, or a list, and a bad scalar must not kill the run."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, n))


def _execute_tool(engine, name: str, args: dict, *, k: int, as_of: str | None,
                  ev: _Evidence) -> tuple[dict, str]:
    """Dispatch one tool call to the engine facade and register every episode id the
    result shows. Returns (result-for-the-model, output_summary). Engine errors AND
    malformed model-supplied arguments come back as tool-visible errors, never
    exceptions — a bad input must not kill the run (the model can correct on retry)."""
    try:
        if name == "graph_search":
            res = engine.search(str(args.get("terms") or ""),
                                k=_int_or(args.get("k"), k))
            eps = [{"id": h["id"], "title": h.get("title"),
                    "created_at": h.get("when") or None,
                    "snippet": _clip(h.get("text") or h.get("description") or "", 200),
                    "score": h.get("score")} for h in res.get("episodes", [])]
            ev.add_ids(*(e["id"] for e in eps))
            return {"episodes": eps}, f"{len(eps)} episode(s)"
        if name == "graph_retrieve":
            res = engine.retrieve(str(args.get("query") or ""),
                                  k=_int_or(args.get("k"), k), as_of=as_of)
            eps = [{"id": h["id"], "title": h.get("title"),
                    "created_at": h.get("when") or None,
                    "snippet": _clip(h.get("text") or h.get("description") or "", 300),
                    "score": h.get("score")} for h in res.get("episodes", [])]
            ev.add_ids(*(e["id"] for e in eps))
            rendered = []
            for f in res.get("facts") or []:
                if isinstance(f, dict):
                    ev.add_row(f)
                    if f.get("rendered"):
                        rendered.append(f["rendered"])
                elif isinstance(f, str):
                    ev.add_fact(f)
                    rendered.append(f)
            return ({"episodes": eps, "facts": rendered},
                    f"{len(eps)} episode(s), {len(rendered)} fact(s)")
        if name == "graph_facts":
            res = engine.facts(str(args.get("entity") or ""), as_of=as_of,
                               include_closed=bool(args.get("include_closed", True)))
            rows = [{"rendered": f["rendered"], "status": f["status"],
                     "episode_id": f["episode_id"]} for f in res.get("facts", [])]
            for f in res.get("facts", []):
                ev.add_row(f)
            return ({"entity": res.get("entity"), "resolved": res.get("resolved"),
                     "facts": rows}, f"{len(rows)} fact(s)")
        if name == "graph_neighbors":
            res = engine.graph_preview(str(args.get("id") or ""))
            nodes = [{"id": n["id"], "label": _clip(n.get("name") or "", 80),
                      "kind": n.get("kind")} for n in res.get("nodes", [])]
            edges = [{"source": e.get("src"), "target": e.get("dst"),
                      "label": e.get("label") or e.get("etype") or ""}
                     for e in res.get("edges", [])]
            ev.add_ids(*(n["id"] for n in nodes if n.get("kind") == "episode"))
            return ({"nodes": nodes, "edges": edges},
                    f"{len(nodes)} node(s), {len(edges)} edge(s)")
        if name == "graph_episode":
            det = engine.episode(str(args.get("id") or ""))
            if det is None:
                return {"error": f"no episode {args.get('id')!r}"}, "not found"
            ev.add_ids(det["id"])
            return ({"id": det["id"], "title": det.get("title"),
                     "created_at": det.get("created_at"),
                     "text": _clip(det.get("text") or "", 2000),
                     "description": det.get("description"),
                     "entities": det.get("entities", []),
                     "concepts": det.get("concepts", [])}, "episode read")
        return {"error": f"unknown tool {name!r}"}, "unknown tool"
    except (EngineError, ValueError, TypeError, KeyError) as e:
        return {"error": str(e)}, "error"


def _notify(progress, state: str, *, tool: str | None = None,
            detail: str | None = None) -> None:
    if progress is None:
        return
    try:
        note = {"state": state}
        if tool:
            note["tool"] = tool
        if detail is not None:
            note["detail"] = _clip(detail, _DETAIL_CHARS)
        progress(note)
    except Exception:  # noqa: BLE001 — narration must never kill the run
        pass


def run_agent(engine, question: str, *, client, provider_kind: str, k: int = 8,
              as_of: str | None = None, max_steps: int | None = None,
              progress=None, budget_s: float = 540.0) -> dict:
    """Run the bounded tool loop and return the §9.2 result: the chat.answer shape
    plus `trace` and `steps`. `progress` (optional) receives {state[, tool, detail]}
    dicts per §9.3 — the caller adds `seq` and emits the terminal notification."""
    reduced = provider_kind in ("codex", "claude")
    ceiling = MAX_STEPS_REDUCED if reduced else MAX_STEPS
    steps = min(int(max_steps), ceiling) if max_steps is not None else \
        min(12, ceiling)
    steps = max(0, steps)               # max_steps=0 → force an immediate answer
    deadline = time.monotonic() + budget_s

    from .models import NodeType
    store = engine._g.store

    def _is_episode(eid: str) -> bool:
        n = store.get_node(eid)
        return n is not None and n.valid and n.ntype is NodeType.EPISODE

    ev = _Evidence(_is_episode)
    trace: list[dict] = []
    driver = _run_reduced if reduced else _run_native
    ans_text, raw_citations = driver(engine, question, client=client, k=k,
                                     as_of=as_of, max_steps=steps, ev=ev,
                                     trace=trace, deadline=deadline,
                                     progress=progress)

    raw = [c for c in (raw_citations or []) if isinstance(c, str)]
    if not raw:
        raw = _EP_ID.findall(ans_text or "")
    kept, dropped = _validate(raw, ev.universe)
    answer = strip_citations(ans_text or "", dropped) or "(no answer produced)"
    return {"answer": answer, "citations": kept, "invalid_citations": dropped,
            "context": {"episodes": ev.episode_ids, "facts": ev.facts,
                        "as_of": as_of},
            "trace": trace, "steps": len(trace)}


def _time_for_another_call(deadline: float, longest_call: float) -> bool:
    """Only start another tool round if a realistically-sized round PLUS the forced
    final call still fit before the deadline. History can under-estimate (a fast
    first call, then a slow one), so this is a heuristic — the hard guarantee is
    _call_timeout, which caps EVERY provider call to the remaining budget."""
    expect = max(longest_call, 15.0)
    return time.monotonic() + 2.0 * expect + 10.0 < deadline


def _call_timeout(deadline: float) -> float:
    """Per-call hard cap: no single provider call may run past the run's deadline
    (§9.2 — the daemon must finish inside the client's 600 s ceiling). The CLI shims
    tighten their 300 s exec timeout to this; SDK clients get it as request timeout."""
    return max(5.0, deadline - time.monotonic() - 5.0)


# ------------------------------------------------------------------ native driver
def _run_native(engine, question, *, client, k, as_of, max_steps, ev, trace,
                deadline, progress) -> tuple[str, list]:
    messages = [{"role": "system", "content": _SYSTEM},
                {"role": "user", "content": f"QUESTION: {question}"
                 + (f"\nAS-OF: {as_of}" if as_of else "")}]
    tools = _GRAPH_TOOLS + [_SUBMIT_TOOL]
    longest = 0.0
    used = 0
    while True:
        forcing = used >= max_steps or not _time_for_another_call(deadline, longest)
        kw = {"messages": messages, "max_tokens": 1024,
              "timeout": _call_timeout(deadline)}
        if forcing:
            kw["tools"] = [_SUBMIT_TOOL]
            kw["tool_choice"] = {"type": "function",
                                 "function": {"name": "submit_answer"}}
        else:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(**kw)
            msg = resp.choices[0].message
        except EngineError:
            raise
        except Exception as e:  # noqa: BLE001 — taxonomy boundary
            if forcing:
                # the final forced call died (deadline squeeze, provider hiccup):
                # return the run's evidence with no answer rather than losing it all
                return "", []
            raise ProviderError(f"agent provider call failed: {e}") from e
        longest = max(longest, time.monotonic() - t0)
        calls = list(getattr(msg, "tool_calls", None) or [])
        if not calls:
            # free-form answer without the submit tool — accept it as final
            return (getattr(msg, "content", None) or ""), []
        submit = next((c for c in calls
                       if getattr(c.function, "name", "") == "submit_answer"), None)
        if submit is not None:
            try:
                payload = json.loads(submit.function.arguments or "{}")
            except json.JSONDecodeError:
                payload = {}
            return (str(payload.get("answer") or ""),
                    list(payload.get("citations") or []))
        if forcing:
            # forced submit_answer but got something else (mock/misbehaving model)
            return (getattr(msg, "content", None) or ""), []
        messages.append({"role": "assistant",
                         "content": getattr(msg, "content", None),
                         "tool_calls": [
                             {"id": getattr(c, "id", f"call_{i}"),
                              "type": "function",
                              "function": {"name": c.function.name,
                                           "arguments": c.function.arguments}}
                             for i, c in enumerate(calls)]})
        for c in calls:
            name = getattr(c.function, "name", "")
            try:
                args = json.loads(c.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            in_summary = _clip(json.dumps(args, ensure_ascii=False), _DETAIL_CHARS)
            _notify(progress, "tool", tool=name, detail=in_summary)
            result, out_summary = _execute_tool(engine, name, args, k=k,
                                                as_of=as_of, ev=ev)
            trace.append({"seq": len(trace) + 1, "tool": name,
                          "input_summary": in_summary,
                          "output_summary": out_summary})
            used += 1
            messages.append({"role": "tool",
                             "tool_call_id": getattr(c, "id", "call_0"),
                             "content": _clip(json.dumps(result,
                                                         ensure_ascii=False),
                                              _TOOL_RESULT_CHARS)})


# ----------------------------------------------------------------- reduced driver
_REDUCED_RULES = """\
Respond with ONLY one JSON object, nothing else. Either call a tool:
  {"tool": "<name>", "arguments": {...}}
or submit the final answer:
  {"answer": "<answer text citing episode ids like [ep_...]>", "citations": ["ep_..."]}"""


def _run_reduced(engine, question, *, client, k, as_of, max_steps, ev, trace,
                 deadline, progress) -> tuple[str, list]:
    """Structured re-prompting for the tool-less CLI shims (§9.4): each turn re-sends
    the whole transcript as text and salvages the first JSON object of the reply."""
    catalog = "\n".join(
        f"- {t['function']['name']}: {t['function']['description']} "
        f"arguments schema: {json.dumps(t['function']['parameters']['properties'])}"
        for t in _GRAPH_TOOLS)
    base = (f"{_SYSTEM}\n\nTOOLS you can call (max {max_steps} calls):\n{catalog}"
            f"\n\n{_REDUCED_RULES}\n\nQUESTION: {question}"
            + (f"\nAS-OF: {as_of}" if as_of else ""))
    transcript: list[str] = []
    longest = 0.0
    salvage_retry = True
    for used in range(max_steps + 1):
        forcing = used >= max_steps or not _time_for_another_call(deadline, longest)
        prompt = base + ("".join(transcript) or "")
        if forcing:
            prompt += ("\n\nYou have no tool calls left. Submit the final answer "
                       "JSON object now.")
        t0 = time.monotonic()
        try:
            resp = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                timeout=_call_timeout(deadline))
            text = resp.choices[0].message.content or ""
        except EngineError:
            raise
        except Exception as e:  # noqa: BLE001 — taxonomy boundary
            if forcing:
                return "", []    # keep the evidence; the answer is just missing
            raise ProviderError(f"agent provider call failed: {e}") from e
        longest = max(longest, time.monotonic() - t0)
        from .llm_client import _first_json_object
        span = _first_json_object(text)
        obj = json.loads(span) if span is not None else None
        if isinstance(obj, dict) and "tool" in obj and not forcing:
            name = str(obj.get("tool") or "")
            args = obj.get("arguments") if isinstance(obj.get("arguments"), dict) \
                else {}
            in_summary = _clip(json.dumps(args, ensure_ascii=False), _DETAIL_CHARS)
            _notify(progress, "tool", tool=name, detail=in_summary)
            result, out_summary = _execute_tool(engine, name, args, k=k,
                                                as_of=as_of, ev=ev)
            trace.append({"seq": len(trace) + 1, "tool": name,
                          "input_summary": in_summary,
                          "output_summary": out_summary})
            transcript.append(
                f"\n\nTOOL CALL {len(trace)}: {name}({in_summary})\nRESULT: "
                + _clip(json.dumps(result, ensure_ascii=False),
                        _TOOL_RESULT_CHARS))
            continue
        if isinstance(obj, dict) and "answer" in obj:
            return str(obj.get("answer") or ""), list(obj.get("citations") or [])
        # Off-protocol reply: no JSON at all, an unexpected object shape, or a tool
        # call on the forced last turn. One structured re-ask, then plain prose is
        # accepted as the answer — but never a raw JSON blob (a tool call is not an
        # answer a user should read).
        if salvage_retry and not forcing:
            salvage_retry = False
            transcript.append("\n\nYour last reply was not one of the two JSON "
                              f"shapes. {_REDUCED_RULES}")
            continue
        return ("" if obj is not None else text), []
    return "", []
