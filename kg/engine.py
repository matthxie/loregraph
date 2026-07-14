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
from .models import (Belief, EdgeType, EntityType, NodeType,
                     entity_category_for_type)

_STUB = ("profile", "rebuild", "reingest", "maintain", "ensure_model")

_DATE10 = re.compile(r"^\d{4}-\d{2}-\d{2}")
_BARE_YEAR = re.compile(r"^\d{4}$")


def _norm_event_date(value: str | None, name: str) -> str | None:
    """Normalize a §7.3 since/until bound: a bare year behaves as its Jan-1 start
    (matching as_of semantics); dates/datetimes compare on their 10-char prefix."""
    if value is None:
        return None
    v = value.strip()
    if not v:
        return None
    if _BARE_YEAR.match(v):
        return f"{v}-01-01"
    if _DATE10.match(v):
        return v[:10]
    raise InvalidInput(f"{name} must be an ISO date/datetime or a bare year")


def _norm_mmr_lambda(value) -> float | None:
    """§7.3: clamp to [0, 1]; a non-finite / unparseable value falls back to the
    engine default (None), never an error."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return max(0.0, min(1.0, f))


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
        # Personal-memory mode: the Engine facade backs a personal knowledge graph, where
        # essentially every note is "I did X with Y". The narrator must therefore be a real
        # anchor — with self_entity off the extractor drops 'me' and every first-person
        # relation, gutting the graph's central hub (the deleted fork always extracted 'me').
        # The CLI turns this on via --self; the daemon builds the Engine, so it must be on here.
        cfg.self_entity = True
        cfg.self_name = "me"
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
    def retrieve(self, query: str, k: int = 8, as_of: str | None = None,
                 rerank: bool = False, mmr_lambda: float | None = None,
                 since: str | None = None, until: str | None = None) -> dict:
        """Full retrieval pipeline (route → PPR → augment → rerank), no LLM: the same
        evidence answer() would hand its model, structured for direct display.
        `rendered_text` is the exact prompt blob, for callers running their own LLM.
        `facts` are structured §3 Fact objects (rendered line included on each row).
        Per-call knobs per PROTOCOL §3.3/§7.3: rerank blends the cross-encoder into
        every lane; mmr_lambda dials the MMR stage; since/until bound episodes to an
        event-time window (inputs only — the result shape is unchanged)."""
        self._check()
        if not query or not query.strip():
            raise InvalidInput("query must be non-empty")
        res = self._g.search(query, k=k, as_of=as_of,
                             rerank=True if rerank else None,
                             mmr_lambda=_norm_mmr_lambda(mmr_lambda),
                             since=_norm_event_date(since, "since"),
                             until=_norm_event_date(until, "until"))
        return {"query": query, "as_of": as_of, "lane": res.lane,
                "episodes": [self._episode_ref(h.episode_id, score=h.score,
                                               when=h.when, text=h.text)
                             for h in res.hits],
                "facts": res.fact_rows,
                "rendered_text": res.context}

    def _episode_ref(self, episode_id: str, *, score: float, when: str = "",
                     text: str = "") -> dict:
        """One ranked hit (retrieve/search), joined with the episode node's projection
        fields so the wire layer can serve EpisodeRef.title and fall back to the media
        description for the snippet (PROTOCOL §3/§7.2)."""
        n = self._g.store.get_node(episode_id)
        return {"id": episode_id, "score": score,
                "when": when or (n.created_at if n else ""),
                "text": text or ((n.raw_text or "") if n else ""),
                "title": (n.title if n else None) or None,
                "description": (n.description if n else None) or None}

    def search(self, terms: str, k: int = 10) -> dict:
        """Keyword/BM25 lookup (PROTOCOL §3.4): exact phrases, names, file types over
        the composite corpus (raw text, title, analyzed description, entity/concept
        surfaces, media file-type tokens). No embedder, no graph walk; scores are raw
        BM25 (higher = better, unnormalized)."""
        self._check()
        if not terms or not terms.strip():
            raise InvalidInput("terms must be non-empty")
        k = max(1, min(int(k), 100))    # k=0/-1 would hit `or`-defaults / negative slices
        hits = self._g.keyword_search(terms, k=k)
        return {"terms": terms,
                "episodes": [self._episode_ref(eid, score=score)
                             for eid, score in hits]}

    def answer(self, question: str, k: int = 8, as_of: str | None = None,
               rerank: bool = False, mmr_lambda: float | None = None,
               since: str | None = None, until: str | None = None) -> dict:
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
            ans = self._g.ask(question, k=k, as_of=as_of, client=client,
                              rerank=True if rerank else None,
                              mmr_lambda=_norm_mmr_lambda(mmr_lambda),
                              since=_norm_event_date(since, "since"),
                              until=_norm_event_date(until, "until"))
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise ProviderError(f"answer failed: {e}") from e
        return {"answer": ans.answer, "citations": ans.citations,
                "invalid_citations": ans.dropped_citations,
                "context": {"episodes": ans.context_episodes,
                            "facts": ans.fact_rows or ans.facts,
                            "rendered_text": ans.context_text,
                            "as_of": ans.as_of}}

    def agent(self, question: str, k: int = 8, as_of: str | None = None,
              max_steps: int | None = None, progress=None) -> dict:
        """Agentic answering (PROTOCOL §9.2): the provider LLM runs a bounded tool
        loop over this facade's own read verbs (kg/agent.py), then submits the same
        cited answer shape as answer(), plus `trace` and `steps`. Same provider
        taxonomy as answer(); `progress` receives §9.3 step dicts."""
        self._check()
        if not question or not question.strip():
            raise InvalidInput("question must be non-empty")
        kind = self._provider.get("kind")
        if kind == "none":
            raise ProviderUnavailable("no LLM provider configured")
        if kind != "mock":
            from .llm_client import llm_available
            if not llm_available(self._provider):
                raise ProviderUnavailable(
                    f"provider {kind!r} is not ready — connect it before asking")
        if kind == "mock":
            client = _MockAnswerClient()
        else:
            from .llm_client import make_client
            client = make_client(self._provider)
        from .agent import run_agent
        try:
            return run_agent(self, question, client=client, provider_kind=kind,
                             k=k, as_of=as_of, max_steps=max_steps,
                             progress=progress)
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise ProviderError(f"agent failed: {e}") from e

    def facts(self, entity: str, as_of: str | None = None,
              include_closed: bool = True) -> dict:
        """One entity's bi-temporal relationships, point-in-time capable (PROTOCOL
        §3.5). `entity` is a name or node id, resolved case-insensitively against
        entity/concept/tag surfaces (and aliases). An unknown entity is NOT an error:
        resolved=False with no facts. Retracted facts are never returned; `as_of`
        keeps facts valid at that time; include_closed=False drops ended windows."""
        self._check()
        if not entity or not entity.strip():
            raise InvalidInput("entity must be non-empty")
        node = self._resolve_surface(entity.strip())
        if node is None:
            return {"entity": entity, "resolved": False, "as_of": as_of, "facts": []}
        return {"entity": entity, "resolved": True, "as_of": as_of,
                "facts": self._entity_fact_rows(node.id, as_of=as_of,
                                                include_closed=include_closed)}

    def _resolve_surface(self, surface: str):
        """A fact endpoint by node id or case-insensitive surface name/alias. Relation
        endpoints can be entity anchors OR tag nodes (kg/ingest.py _resolve_endpoint),
        so both types resolve here."""
        store = self._g.store
        n = store.get_node(surface)
        if n is not None and n.valid and n.ntype in (NodeType.ENTITY, NodeType.TAG):
            return n
        want = surface.lower()
        for ntype in (NodeType.ENTITY, NodeType.TAG):
            for cand in store.nodes_of_type(ntype):
                if (cand.name or "").lower() == want or any(
                        (a or "").lower() == want for a in (cand.aliases or [])):
                    return cand
        return None

    def _fact_row(self, src_id: str, dst_id: str, data: dict) -> dict:
        """One structured §3 Fact object from a RELATED_TO edge's data dict."""
        from .facts import FactLine
        store = self._g.store
        rel = data.get("rel_tag")
        rel_node = store.get_node(rel) if rel else None
        sn, tn = store.get_node(src_id), store.get_node(dst_id)
        line = FactLine(src=sn.name if sn else src_id,
                        rel=rel_node.name if rel_node else "related_to",
                        dst=tn.name if tn else dst_id,
                        valid_at=data.get("valid_at", ""),
                        invalid_at=data.get("invalid_at", ""),
                        episode_id=data.get("episode_id", ""))
        return {"source": line.src, "predicate": line.rel, "target": line.dst,
                "status": "ended" if data.get("invalid_at") else "asserted",
                "valid_from": data.get("valid_at") or None,
                "valid_to": data.get("invalid_at") or None,
                "recorded_at": data.get("created_at") or None,
                "episode_id": data.get("episode_id") or None,
                "confidence": data.get("confidence"),
                "provenance": (data.get("provenance") or "").lower() or None,
                "functional": bool(rel_node.functional) if rel_node else False,
                "disputed_by": data.get("disputed_by") or [],
                "rendered": line.render()}

    def _entity_fact_rows(self, entity_id: str, *, as_of: str | None,
                          include_closed: bool) -> list[dict]:
        """Structured §3.5 fact rows for one endpoint, walking RELATED_TO both ways
        (same view rules as kg/facts.py: believed only, closed windows are history,
        retracted never served), ordered by valid-time then transaction time."""
        from .facts import _believed
        from .store import fact_active
        store = self._g.store
        rows: list[dict] = []
        seen: set[tuple] = set()
        for direction in ("out", "in"):
            for nbr, data in store.neighbors(entity_id, etypes={EdgeType.RELATED_TO},
                                             direction=direction):
                if not _believed(data):
                    continue
                if as_of is not None:
                    if not fact_active(data, as_of):
                        continue
                elif not include_closed and data.get("invalid_at"):
                    continue
                src_id, dst_id = ((entity_id, nbr) if direction == "out"
                                  else (nbr, entity_id))
                key = (src_id, data.get("rel_tag"), dst_id,
                       data.get("valid_at", ""), data.get("invalid_at", ""))
                if key in seen:
                    continue
                seen.add(key)
                rows.append(self._fact_row(src_id, dst_id, data))
        rows.sort(key=lambda r: (r["valid_from"] or "", r["recorded_at"] or ""))
        return rows

    def episode(self, episode_id: str) -> dict | None:
        self._check()
        n = self._g.store.get_node(episode_id)
        if n is None or n.ntype is not NodeType.EPISODE or not n.valid:
            return None                        # tombstoned episodes are gone from this view
        entities, categories, concepts = self._episode_entities(episode_id)
        return {"id": n.id, "text": n.raw_text or "", "created_at": n.created_at,
                "ingested_at": n.ingested_at, "source": n.name,
                "title": n.title or None, "description": n.description or None,
                "media_paths": list(n.media_paths or []),
                "modality": n.modality.value if n.modality else "text",
                "entities": entities, "entity_categories": categories,
                "concepts": concepts,
                "facts": self._episode_grounded_facts(episode_id)}

    def _episode_grounded_facts(self, episode_id: str) -> list[dict]:
        """The §3.6 fact rows this episode grounds (asserted or ended by this note;
        retracted facts stay excluded — they were never true). Walks the RELATED_TO
        edges of the episode's mentioned entities and keeps the ones whose provenance
        `episode_id` is this note. A relation endpoint minted as a fallback 'other'
        entity has no mention edge, so as a backstop the mention star is widened by
        the fact edges' own endpoints — per detail request only, never per list row."""
        from .facts import _believed
        from .models import SELF_ENTITY_ID
        store = self._g.store
        seeds = set(self._episode_entity_ids(episode_id))
        for tid, _d in store.neighbors(episode_id, etypes={EdgeType.TAGGED_AS},
                                       direction="out"):
            seeds.add(tid)                      # relation endpoints can be tag nodes
        if store.get_node(SELF_ENTITY_ID) is not None:
            seeds.add(SELF_ENTITY_ID)           # 'me' grounds most personal facts
        rows: list[dict] = []
        seen_edges: set[tuple] = set()
        for eid in seeds:
            for direction in ("out", "in"):
                for nbr, data in store.neighbors(eid, etypes={EdgeType.RELATED_TO},
                                                 direction=direction):
                    if data.get("episode_id") != episode_id or not _believed(data):
                        continue
                    src_id, dst_id = ((eid, nbr) if direction == "out"
                                      else (nbr, eid))
                    key = (src_id, data.get("rel_tag"), dst_id,
                           data.get("valid_at", ""), data.get("seq", 0))
                    if key in seen_edges:
                        continue
                    seen_edges.add(key)
                    rows.append(self._fact_row(src_id, dst_id, data))
        rows.sort(key=lambda r: (r["valid_from"] or "", r["recorded_at"] or ""))
        return rows

    def _episode_entities(
        self, episode_id: str
    ) -> tuple[list[str], dict[str, str], list[str]]:
        """The entities this episode mentions, walking the star episode ← MENTIONED_IN ←
        mention → RESOLVES_TO → entity. Named entities (person/place/thing) come back with
        their glyph category (persisted entity_category, else entity_category_for_type, which
        folds org/work/event into thing). CONCEPT-type nodes are split into a separate
        `concepts` list (topical strings) rather than folded into thing, so clients can count
        and render them as their own category. Each surface name is reported once."""
        store = self._g.store
        names: list[str] = []
        categories: dict[str, str] = {}
        concepts: list[str] = []
        seen: set[str] = set()
        for mid, _d in store.neighbors(episode_id, etypes={EdgeType.MENTIONED_IN},
                                       direction="in"):
            for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                            direction="out"):
                node = store.get_node(eid)
                if not node or node.name in seen:
                    continue
                seen.add(node.name)
                if node.entity_type is EntityType.CONCEPT:
                    concepts.append(node.name)
                else:
                    names.append(node.name)
                    categories[node.name] = (
                        node.entity_category
                        or entity_category_for_type(node.entity_type).value)
        return names, categories, concepts

    _PREVIEW_MAX_NODES = 22

    def graph_preview(self, node_id: str) -> dict:
        """The complete one-hop display graph rooted at an episode, entity, or concept
        (PROTOCOL §3.6a), shaped for the wire layer:
        {nodes:[{id,name,kind,category,hop,external_connections}],
         edges:[{src,dst,etype,label}]}.

        Mention stars collapse to direct episode→entity MENTIONS edges (label "");
        asserted RELATED_TO facts between two drawn nodes ride with their predicate
        name in `label`. At most 22 nodes: the root, then hop-1 neighbours by
        descending display connectivity. `external_connections` counts each drawn
        node's unique display neighbours that did NOT make it on screen, so clients
        can draw dashed continuation stubs."""
        self._check()
        store = self._g.store
        root = store.get_node(node_id)
        if (root is None or not root.valid
                or root.ntype not in (NodeType.EPISODE, NodeType.ENTITY)):
            raise NotFound(f"unknown graph node: {node_id}")
        ranked = sorted(self._display_neighbors(node_id),
                        key=lambda i: (-len(self._display_neighbors(i)), i))
        drawn_ids = [node_id] + ranked[:self._PREVIEW_MAX_NODES - 1]
        drawn = set(drawn_ids)
        nodes = []
        for hop_pos, nid in enumerate(drawn_ids):
            n = store.get_node(nid)
            is_ep = n.ntype is NodeType.EPISODE
            # An episode's `name` is its source_ref ("app"/"capture") — useless as a
            # graph label, so episodes display their text (PROTOCOL §3.6a example).
            label = (" ".join((n.raw_text or n.description or n.name or "").split())[:80]
                     if is_ep else n.name)
            nodes.append({
                "id": nid, "name": label,
                "kind": ("episode" if is_ep
                         else "concept" if n.entity_type is EntityType.CONCEPT
                         else "entity"),
                "category": None if is_ep else (
                    n.entity_category
                    or entity_category_for_type(n.entity_type).value),
                "hop": 0 if hop_pos == 0 else 1,
                "external_connections": len(self._display_neighbors(nid) - drawn),
            })
        edges, seen = [], set()
        for nid in drawn_ids:
            n = store.get_node(nid)
            if n.ntype is NodeType.EPISODE:
                pairs = ((nid, eid, "MENTIONS", "")
                         for eid in self._episode_entity_ids(nid))
            else:
                pairs = ((src, dst, "RELATED_TO", pred)
                         for src, dst, pred in self._entity_fact_edges(nid))
            for src, dst, etype, label in pairs:
                key = (src, dst, etype, label)
                if src in drawn and dst in drawn and key not in seen:
                    seen.add(key)
                    edges.append({"src": src, "dst": dst,
                                  "etype": etype, "label": label})
        return {"nodes": nodes, "edges": edges}

    def _episode_entity_ids(self, episode_id: str) -> list[str]:
        """Distinct valid entity ids this episode mentions (via the mention star)."""
        store = self._g.store
        out: list[str] = []
        seen: set[str] = set()
        for mid, _d in store.neighbors(episode_id, etypes={EdgeType.MENTIONED_IN},
                                       direction="in"):
            for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                            direction="out"):
                node = store.get_node(eid)
                if node is not None and node.valid and eid not in seen:
                    seen.add(eid)
                    out.append(eid)
        return out

    def _entity_fact_edges(self, entity_id: str) -> list[tuple[str, str, str]]:
        """(src, dst, predicate_name) for asserted RELATED_TO facts touching this
        entity, in stored orientation. Retracted facts (never actually true) are
        excluded; closed facts stay — they are real history the graph should show."""
        store = self._g.store
        out: list[tuple[str, str, str]] = []
        for direction in ("out", "in"):
            for nbr, data in store.neighbors(entity_id,
                                             etypes={EdgeType.RELATED_TO},
                                             direction=direction):
                if data.get("belief") == Belief.RETRACTED.value:
                    continue
                rel = data.get("rel_tag") or ""
                rel_node = store.get_node(rel) if rel else None
                pred = rel_node.name if rel_node is not None else rel
                src, dst = ((entity_id, nbr) if direction == "out"
                            else (nbr, entity_id))
                out.append((src, dst, pred))
        return out

    def _display_neighbors(self, node_id: str) -> set[str]:
        """One-hop neighbours in DISPLAY-graph terms: an episode connects to the
        entities it mentions; an entity connects to its mentioning episodes and its
        fact partners."""
        store = self._g.store
        n = store.get_node(node_id)
        if n is None or not n.valid:
            return set()
        if n.ntype is NodeType.EPISODE:
            return set(self._episode_entity_ids(node_id))
        out = {ep for ep in store.entity_episodes(node_id)
               if (epn := store.get_node(ep)) is not None and epn.valid}
        for src, dst, _pred in self._entity_fact_edges(node_id):
            other = dst if src == node_id else src
            partner = store.get_node(other)
            if partner is not None and partner.valid:
                out.add(other)
        return out

    def episodes_list(self, offset: int = 0, limit: int = 100) -> dict:
        """Every episode as a full §7.2 list row, newest-first (created_at desc, id desc)
        — the same projection episode() serves, so the wire layer never needs a per-row
        detail round-trip."""
        self._check()
        eps = sorted(self._g.store.nodes_of_type(NodeType.EPISODE),
                     key=lambda n: (n.created_at, n.id), reverse=True)
        rows = []
        for n in eps[offset:offset + limit]:
            entities, categories, concepts = self._episode_entities(n.id)
            try:
                gp = self.graph_preview(n.id)
            except EngineError:
                gp = {"nodes": [], "edges": []}
            rows.append({"id": n.id, "text": n.raw_text or "",
                         "created_at": n.created_at, "ingested_at": n.ingested_at,
                         "source": n.name, "title": n.title or None,
                         "description": n.description or None,
                         "media_paths": list(n.media_paths or []),
                         "modality": n.modality.value if n.modality else "text",
                         "entities": entities, "entity_categories": categories,
                         "concepts": concepts, "graph_preview": gp})
        return {"total": len(eps), "offset": offset, "episodes": rows}

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
