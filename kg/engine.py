"""Engine facade — the app-facing surface of this package (Engine Interface Contract v0).

The app's router imports this ONE class and treats the package as a black box:

    from kg.engine import Engine
    eng = Engine.open(data_dir, provider={"kind": "mock"}, log=my_log)
    res = eng.ingest(NoteInput(text="…", created_at="2026-07-12T09:00:00Z"))
    ans = eng.answer("where does Becky live?")
    eng.close()

v0 scope — CONNECTABLE, not complete. Real graph behind ingest/retrieve/answer/
episode(s)/stats/delete_episode; contract methods the engine can't honestly serve yet
raise EngineError("not implemented"). Known deltas from the contract, worked out later:

  * Providers: "mock" and "none" are contract-complete; "openai", "codex" and
    "anthropic" are live via kg.llm_client — set_active_provider() persists the chosen
    kind (and any injected api_key) into the process env so every scattered call site
    picks it up, and make_client() builds the concrete SDK-shaped client on demand.
  * Durability: ingest() saves the store before returning, but does not fsync yet.
    Idempotency IS honored (content-hash dedup is native to the ingest pipeline).
  * tasks in IngestResult is always [] — task/intent extraction is not in the schema yet.
  * Embeddings use the local bge model: deterministic and offline once cached, but the
    first ever run downloads weights (ensure_model() is one of the unimplemented stubs).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import types
from dataclasses import dataclass, field

from .config import Config
from .corpus import CorpusItem
from .errors import (EngineError, InvalidInput, NotFound, ProviderError,
                     ProviderUnavailable, StoreError)
from .extractors import Extraction, ExtractedEntity, UsageMeter
from .llm_client import SUPPORTED_KINDS
from .models import EdgeType, NodeType, entity_category_for_type

_STUB = ("search", "facts", "profile", "rebuild", "reingest", "maintain",
         "ensure_model")


# --------------------------------------------------------------------------- #
# Contract types (§2)
# --------------------------------------------------------------------------- #
@dataclass
class NoteInput:
    text: str                       # raw note, byte-verbatim; never rewritten
    created_at: str                 # ISO-8601
    attachments: list[str] = field(default_factory=list)
    source: str = "app"


@dataclass
class Task:
    text: str                       # extracted intent, verbatim-ish
    due: str | None = None


@dataclass
class IngestResult:
    episode_id: str
    tasks: list[Task] = field(default_factory=list)
    entities: int = 0               # mention writes this ingest
    relations: int = 0              # fact-edge actions this ingest
    concepts: int = 0               # tag links this ingest (not separately counted yet)
    skipped: bool = False           # duplicate note (idempotent re-run)


# --------------------------------------------------------------------------- #
# Mock provider (§5): deterministic, offline, no LLM
# --------------------------------------------------------------------------- #
class _MockExtractor:
    """Canned extraction: capitalized words become OTHER entities, frequent lowercase
    words become tags. Deterministic, instant, model-free — exists so the app's smoke
    test can drive the full ingest pipeline without a provider or local NLP models."""
    name = "mock"

    def __init__(self):
        self.meter = UsageMeter()

    def extract_text(self, text: str, title: str = "") -> Extraction:
        ents = []
        seen = set()
        for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text):
            if w.lower() not in seen:
                seen.add(w.lower())
                ents.append(ExtractedEntity(name=w))
        tags = []
        for w in re.findall(r"\b[a-z]{5,}\b", text.lower()):
            if w not in tags:
                tags.append(w)
            if len(tags) == 3:
                break
        return Extraction(entities=ents[:8], tags=tags)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return Extraction(tags=[label_hint] if label_hint else [])


class _MockAnswerClient:
    """OpenAI-SDK-shaped client returning one canned submit_answer tool call, so the
    real RagAnswerer (context assembly, citation validation) runs end-to-end offline."""

    def __init__(self):
        self.chat = self
        self.completions = self

    def create(self, **kw):
        tc = types.SimpleNamespace(
            id="call_0",
            function=types.SimpleNamespace(
                name="submit_answer",
                arguments=json.dumps({"answer": "(mock provider) canned answer over the "
                                                "retrieved context.",
                                      "citations": []})))
        message = types.SimpleNamespace(content=None, tool_calls=[tc])
        choice = types.SimpleNamespace(message=message, finish_reason="tool_calls")
        usage = types.SimpleNamespace(prompt_tokens=0, completion_tokens=0)
        return types.SimpleNamespace(choices=[choice], usage=usage)


# --------------------------------------------------------------------------- #
# Engine (§1)
# --------------------------------------------------------------------------- #
class Engine:
    """Facade over KnowledgeGraph shaped to the Engine Interface Contract."""

    def __init__(self, data_dir: str, provider: dict, log):
        self._log = log or (lambda level, msg: None)
        self._data_dir = os.path.abspath(data_dir)
        self._closed = False
        self._provider = dict(provider or {})
        kind = self._provider.get("kind")
        if kind not in SUPPORTED_KINDS:
            raise ProviderUnavailable(
                f"provider kind {kind!r} not supported yet (supported: {SUPPORTED_KINDS})")
        # One env-backed switch reaches every scattered LLM call site (extraction, rag),
        # replacing the old hand-rolled OPENAI_API_KEY bridge; an injected api_key rides along.
        from .llm_client import set_active_provider
        set_active_provider(self._provider)

        os.makedirs(os.path.join(self._data_dir, "store"), exist_ok=True)
        cfg = Config.default()
        cfg.embedder = "st"
        from .graph import KnowledgeGraph
        store_path = os.path.join(self._data_dir, "store", "kg.db")
        if kind in ("mock", "none"):
            # model-free extraction: mock is the contract; none = no LLM escalation either
            # way, and skipping the local NLP stack keeps open() light. Revisit for none.
            from unittest import mock as _m
            with _m.patch("kg.graph.get_extractor", return_value=_MockExtractor()):
                self._g = KnowledgeGraph.open(store_path, cfg)
            self._g.extractor = _MockExtractor()
        else:
            self._g = KnowledgeGraph.open(store_path, cfg)
        self._log("info", f"engine open: data_dir={self._data_dir} provider={kind}")

    # -------------------------------------------------------------- lifecycle
    @classmethod
    def open(cls, data_dir: str, provider: dict, log=None) -> "Engine":
        return cls(data_dir, provider, log)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._g.save()
        finally:
            self._closed = True
            self._log("info", "engine closed")

    def _check(self):
        if self._closed:
            raise EngineError("engine is closed")

    # -------------------------------------------------------------- ingestion
    def ingest(self, note: NoteInput) -> IngestResult:
        self._check()
        if not isinstance(note.text, str) or not note.text.strip():
            raise InvalidInput("note.text must be a non-empty string")
        if not note.created_at:
            raise InvalidInput("note.created_at is required (ISO-8601)")
        nid = hashlib.sha256(f"{note.created_at}\n{note.text}".encode("utf-8")).hexdigest()[:16]
        item = CorpusItem(id=nid, modality="text", source_ref=f"{note.source}/{nid}",
                          text=note.text, created_at=note.created_at)
        try:
            report = self._g.ingest([item])
            self._g.save()                      # durability: on disk before we return
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise StoreError(f"ingest failed: {e}") from e
        if report.extraction_failures:
            raise ProviderError(f"extraction failed: {report.notes[:1]}")
        return IngestResult(episode_id=f"ep_{nid}", tasks=[],
                            entities=report.mentions, relations=report.facts,
                            skipped=bool(report.skipped))

    def delete_episode(self, episode_id: str) -> None:
        self._check()
        if not self._g.store.has_node(episode_id):
            raise NotFound(f"unknown episode: {episode_id}")
        from .forget import forget as _forget
        _forget(self._g.store, episode_ids=[episode_id])
        self._g.save()

    # ------------------------------------------------------------------ query
    def retrieve(self, query: str, k: int = 8, as_of: str | None = None) -> dict:
        """Full retrieval pipeline (route → PPR → augment → rerank), no LLM: the same
        evidence answer() would hand its model, structured for direct display.
        `rendered_text` is the exact prompt blob, for callers running their own LLM."""
        self._check()
        if not query or not query.strip():
            raise InvalidInput("query must be non-empty")
        res = self._g.search(query, k=k, as_of=as_of)
        return {"query": query, "as_of": as_of, "lane": res.lane,
                "episodes": [{"id": h.episode_id, "score": h.score,
                              "when": h.when, "text": h.text}
                             for h in res.hits],
                "facts": res.facts,
                "rendered_text": res.context}

    def answer(self, question: str, k: int = 8, as_of: str | None = None) -> dict:
        self._check()
        if not question or not question.strip():
            raise InvalidInput("question must be non-empty")
        kind = self._provider.get("kind")
        if kind == "none":
            raise ProviderUnavailable("no LLM provider configured")
        if kind != "mock":
            # The env-selected provider must be able to serve a call now (key present /
            # codex logged in); the migrated rag path builds the client via make_client().
            from .llm_client import llm_available
            if not llm_available(self._provider):
                raise ProviderUnavailable(
                    f"provider {kind!r} is not ready — connect it before asking")
        client = _MockAnswerClient() if kind == "mock" else None
        try:
            ans = self._g.ask(question, k=k, as_of=as_of, client=client)
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise ProviderError(f"answer failed: {e}") from e
        return {"answer": ans.answer, "citations": ans.citations,
                "invalid_citations": ans.dropped_citations,
                "context": {"episodes": ans.context_episodes, "facts": ans.facts,
                            "as_of": ans.as_of}}

    def episode(self, episode_id: str) -> dict | None:
        self._check()
        n = self._g.store.get_node(episode_id)
        if n is None or n.ntype is not NodeType.EPISODE or not n.valid:
            return None                        # tombstoned episodes are gone from this view
        entities, categories = self._episode_entities(episode_id)
        return {"id": n.id, "text": n.raw_text or "", "created_at": n.created_at,
                "ingested_at": n.ingested_at, "source": n.name,
                "entities": entities, "entity_categories": categories}

    def _episode_entities(self, episode_id: str) -> tuple[list[str], dict[str, str]]:
        """The entities this episode mentions, walking the star episode ← MENTIONED_IN ←
        mention → RESOLVES_TO → entity. Categories derive on read from the stored
        entity_category, falling back to entity_category_for_type(entity_type) (ingest does
        not persist the glyph category yet — see BUILD_REPORT followups)."""
        store = self._g.store
        names: list[str] = []
        categories: dict[str, str] = {}
        for mid, _d in store.neighbors(episode_id, etypes={EdgeType.MENTIONED_IN},
                                       direction="in"):
            for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                            direction="out"):
                node = store.get_node(eid)
                if node and node.name not in categories:
                    names.append(node.name)
                    categories[node.name] = (
                        node.entity_category
                        or entity_category_for_type(node.entity_type).value)
        return names, categories

    def graph_preview(self, episode_id: str) -> dict:
        """A minimal graph neighbourhood for one episode, shaped for the graph renderer:
        {nodes:[{id,name,category}], edges:[{src,dst,label}]}. v0 returns the episode's
        direct entities as nodes with a MENTIONS edge from the episode to each; a deeper
        two-hop expansion is deferred (see BUILD_REPORT followups)."""
        self._check()
        store = self._g.store
        n = store.get_node(episode_id)
        if n is None or n.ntype is not NodeType.EPISODE or not n.valid:
            raise NotFound(f"unknown episode: {episode_id}")
        nodes = [{"id": episode_id, "name": n.name, "category": "episode"}]
        edges = []
        for mid, _d in store.neighbors(episode_id, etypes={EdgeType.MENTIONED_IN},
                                       direction="in"):
            for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                            direction="out"):
                node = store.get_node(eid)
                if node is None or any(x["id"] == eid for x in nodes):
                    continue
                category = (node.entity_category
                            or entity_category_for_type(node.entity_type).value)
                nodes.append({"id": eid, "name": node.name, "category": category})
                edges.append({"src": episode_id, "dst": eid, "label": "MENTIONS"})
        return {"nodes": nodes, "edges": edges}

    def episodes_list(self, offset: int = 0, limit: int = 100) -> dict:
        self._check()
        eps = sorted(self._g.store.nodes_of_type(NodeType.EPISODE),
                     key=lambda n: (n.created_at, n.id))
        page = eps[offset:offset + limit]
        return {"total": len(eps), "offset": offset,
                "episodes": [{"id": n.id, "created_at": n.created_at,
                              "preview": (n.raw_text or "")[:160]} for n in page]}

    def stats(self) -> dict:
        self._check()
        return self._g.stats()

    # --------------------------------------------------------------- provider
    def set_provider(self, provider: dict) -> None:
        """Runtime provider switch (§5). v0: swaps between the supported kinds by
        re-opening internals is unnecessary — only the answer path and the openai key
        bridge depend on it."""
        self._check()
        kind = (provider or {}).get("kind")
        if kind not in SUPPORTED_KINDS:
            raise ProviderUnavailable(
                f"provider kind {kind!r} not supported yet (supported: {SUPPORTED_KINDS})")
        from .llm_client import set_active_provider
        self._provider = dict(provider)
        set_active_provider(self._provider)
        self._log("info", f"provider set: {kind}")

    def provider_status(self) -> dict:
        self._check()
        from .llm_client import provider_status as _status
        return _status(self._provider)

    def provider_signout(self) -> dict:
        self._check()
        from .llm_client import provider_signout as _signout
        return _signout(self._provider.get("kind"))

    def provider_usage(self) -> dict:
        self._check()
        from .llm_client import provider_usage as _usage
        return _usage(self._provider.get("kind"))


def _not_implemented(name: str):
    def _stub(self, *a, **kw):
        raise EngineError(f"not implemented in the v0 facade: {name}()")
    _stub.__name__ = name
    return _stub


for _name in _STUB:
    setattr(Engine, _name, _not_implemented(_name))
