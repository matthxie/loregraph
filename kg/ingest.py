"""Ingestion pipeline (docs/ARCHITECTURE.md §6) — the episodic / semantic split.

Per entry: intake → SHA256 cache check → normalize → extract (directly from raw content)
→ write an immutable **Episode** + one immutable **Mention** per entity occurrence
(MENTIONED_IN → episode, RESOLVES_TO → a lean canonical **Entity** anchor) → canonicalize
tags/entities/relations → resolve each relationship through the bi-temporal fact logic
(kg/temporal.py: open / confirm / close / supersede). Then an incremental pass derives
Episode↔Episode edges for the new episodes (shared tags/entities + embedding kNN).

Embeddings are written ONLY to immutable nodes (episode text, mention surfaces) — never to
the canonical entity anchors — so embedding cost is strictly proportional to new data and
nothing is ever re-embedded on an update (the re-embedding problem is designed out).

LLM extraction fans out under a bounded-concurrency semaphore; all graph mutation happens
sequentially in the main thread so shared state stays consistent.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

from .canonicalize import Canonicalizer, normalize_key
from .chunkers import chunk_for
from .config import Config
from .corpus import CorpusItem
from .embedders import Embedder
from .extractors import Extraction, Extractor, extract_text_sectioned
from .models import (Edge, EdgeType, EntityCategory, EntityType, Modality, NodeType,
                     Provenance, entity_category_for_type, episode_node, mention_node,
                     quantity_node, source_node)
from .profiler import span as prof_span
from .store import GraphStore, now_iso
from .temporal import apply_fact


@dataclass
class IngestReport:
    ingested: int = 0           # episodes written
    skipped: int = 0
    failed: int = 0
    mentions: int = 0
    facts: int = 0              # fact-edge actions (open/close/supersede/confirm)
    quantity_facts: int = 0     # typed amount/count/measurement occurrences written
    extraction_failures: int = 0
    seconds: float = 0.0
    extractor: str = ""
    embedder: str = ""
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        warn = (f"  ⚠ {self.extraction_failures} extraction failures"
                if self.extraction_failures else "")
        return (f"episodes={self.ingested} mentions={self.mentions} facts={self.facts} "
                f"quantities={self.quantity_facts} "
                f"skipped={self.skipped} failed={self.failed} in {self.seconds:.1f}s  "
                f"(extractor={self.extractor}, embedder={self.embedder}){warn}")


def _sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
    return h.hexdigest()


def _modality_of(label: str | None) -> Modality:
    """Map a free-form modality label to the enum. Unknown labels fall back to FILE so a new
    medium never crashes ingest — the surface choice keys off raw_text presence, not this."""
    try:
        return Modality((label or "text").lower())
    except ValueError:
        return Modality.FILE


class Ingestor:
    def __init__(self, store: GraphStore, extractor: Extractor, embedder: Embedder,
                 canon: Canonicalizer, config: Config):
        self.store = store
        self.extractor = extractor
        self.embedder = embedder
        self.canon = canon
        self.config = config
        # mention ids are globally unique + stable across ingests
        self._mention_seq = sum(1 for n in store.nodes.values()
                                if n.ntype == NodeType.MENTION)
        # quantity-fact node ids: globally unique, never reused/merged (see quantity_node)
        self._fact_seq = sum(1 for n in store.nodes.values()
                             if n.ntype == NodeType.ENTITY and n.entity_type == EntityType.QUANTITY)

    # ------------------------------------------------------------------ public
    def ingest(self, items: list[CorpusItem]) -> IngestReport:
        t0 = time.time()
        report = IngestReport(extractor=self.extractor.name, embedder=self.embedder.name)

        # 0. optional natural-boundary chunking: one big text entry → several small
        #    chunk episodes + an un-rankable SOURCE parent (edges wired in step 4b)
        items, parents = self._expand_chunks(items)

        # 1. intake + cache decisions. Episodes are APPEND-ONLY: identical content is
        #    skipped; changed content under a known id appends a new immutable episode
        #    version (the old one stays — both are real historical entries).
        #    Intra-batch siblings must see each other's claims: nothing is written until
        #    step 4, so this batch's claimed hashes + ep_ids are tracked locally — keeping
        #    the content-hash skip and append-only versioning identical to a per-item path
        #    (two same-content items dedup; two same-id, different-content items version)
        #    instead of silently colliding on add_node.
        pending: list[tuple[CorpusItem, str, str]] = []  # item, hash, episode_id
        batch_hashes: set[str] = set()
        batch_ep_ids: set[str] = set()
        for item in items:
            # TEXT-surfaced items hash on their text; described media (no raw_text) hash on
            # the artifact path so two distinct uploads never collide. The event-date
            # (created_at) and stable raw id are SALTED into the key: two deliberate
            # captures may have byte-identical text and timestamps but are still distinct
            # episodes, while a true re-ingest reuses the same id and still skips.
            content = item.text if item.text is not None else (item.image_path or "")
            h = _sha256(item.modality, content, item.created_at or "", item.id)
            if h in self.store.hash_cache or h in batch_hashes:
                report.skipped += 1
                continue
            base_id = f"ep_{item.id}"
            ep_id = base_id
            if self.store.has_node(base_id) or base_id in batch_ep_ids:
                existing = self.store.get_node(base_id)
                if existing is not None and existing.content_hash == h:
                    report.skipped += 1          # identical episode already in the store
                    continue
                v = 1                            # same id, new content → append a version
                while self.store.has_node(f"{base_id}_v{v}") or f"{base_id}_v{v}" in batch_ep_ids:
                    v += 1
                ep_id = f"{base_id}_v{v}"
            batch_hashes.add(h)
            batch_ep_ids.add(ep_id)
            pending.append((item, h, ep_id))

        if not pending:
            report.seconds = time.time() - t0
            return report

        # 2. extract concurrently (bounded semaphore on the LLM calls)
        with prof_span("ingest.extract"):
            extractions, errors = self._extract_all([p[0] for p in pending])
        report.extraction_failures = len(errors)
        if errors:
            report.notes.append(f"extraction failed for {len(errors)} item(s); "
                                f"first error: {errors[0]}")

        # 3. embed episode surfaces (raw text / image description) in one batch
        surfaces = [self._embed_surface(item, ext)
                    for (item, *_), ext in zip(pending, extractions)]
        with prof_span("ingest.embed_episodes"):
            ep_vecs = self.embedder.embed(surfaces)

        # 3b. batch-embed every tag/entity/relation surface for canonicalization
        canon_surfaces: list[str] = []
        ment_surfaces: list[str] = []
        for ext in extractions:
            canon_surfaces.extend(ext.tags)
            for e in ext.entities:
                canon_surfaces.append(e.name)
                ment_surfaces.append(e.name)
            for r in ext.relations:
                canon_surfaces.extend([r.source, r.target, *r.labels])
            for qf in ext.facts:
                canon_surfaces.extend([qf.subject, qf.predicate])
        with prof_span("ingest.embed_canon_prime"):
            self.canon.prime_embeddings(canon_surfaces)

        # 3c. batch-embed mention surfaces (the immutable mention layer's embeddings)
        with prof_span("ingest.embed_mentions"):
            uniq_ments = sorted({s for s in ment_surfaces if s.strip()})
            ment_vecs = dict(zip(uniq_ments, self.embedder.embed(uniq_ments))) if uniq_ments else {}

        # 4. write each episode (sequential — canonicalize + bi-temporal fact logic inside),
        #    checkpointing to SQLite as we go so a crash loses at most one flush window
        new_eps: list[str] = []
        flush_every = getattr(self.config, "ingest_flush_every", 0)
        with prof_span("ingest.write_episodes"):
            for i, ((item, h, ep_id), ext, vec) in enumerate(
                    zip(pending, extractions, ep_vecs)):
                try:
                    m, f, qf = self._write_episode(item, h, ep_id, ext, vec, ment_vecs,
                                                   report)
                    new_eps.append(ep_id)
                    report.ingested += 1
                    report.mentions += m
                    report.facts += f
                    report.quantity_facts += qf
                except Exception as e:  # noqa: BLE001
                    report.failed += 1
                    report.notes.append(f"{item.id}: {e!r}")
                if flush_every and self.store.path and (i + 1) % flush_every == 0:
                    self.store.flush()

        # 4b. parent SOURCE nodes + PART_OF / NEXT structure edges for chunked entries
        if parents:
            with prof_span("ingest.write_parents"):
                self._write_parents(parents)

        # 5. derive Episode↔Episode edges for the NEW episodes only (incremental)
        with prof_span("ingest.derive_edges"):
            self._derive_episode_edges(new_eps)
        if self.store.path:
            self.store.flush()

        report.seconds = time.time() - t0
        return report

    # ------------------------------------------------------------------ chunking
    def _expand_chunks(self, items: list[CorpusItem]) \
            -> tuple[list[CorpusItem], list[tuple[str, CorpusItem, list[str]]]]:
        """With config.chunking on, replace each big text entry by its natural-boundary
        chunks (kg/chunkers.py). Chunk ids are `<source_id>#cNNN` — deterministic, and
        the `#` suffix collapses in eval's session-level matching. Returns the expanded
        item list plus (source_node_id, original item, [chunk episode ids]) per parent."""
        mode = getattr(self.config, "chunking", "none")
        if mode == "none":
            return items, []
        out: list[CorpusItem] = []
        parents: list[tuple[str, CorpusItem, list[str]]] = []
        for item in items:
            if item.modality != "text" or not item.text:
                out.append(item)
                continue
            chunks = chunk_for(item.text, mode=mode,
                               target=int(self.config.chunk_target_chars),
                               max_chars=int(self.config.chunk_max_chars))
            if not chunks:
                out.append(item)
                continue
            kids: list[str] = []
            for c in chunks:
                cid = f"{item.id}#c{c.ordinal:03d}"
                out.append(CorpusItem(id=cid, modality="text",
                                      source_ref=item.source_ref, title=item.title,
                                      text=c.text, created_at=item.created_at))
                kids.append(f"ep_{cid}")
            parents.append((f"src_{item.id}", item, kids))
        return out, parents

    def _write_parents(self, parents: list[tuple[str, CorpusItem, list[str]]]) -> None:
        """One SOURCE node per chunked entry (full original text, provenance) +
        PART_OF (chunk→parent, low weight) and NEXT (chunk→next sibling) edges."""
        pw = float(getattr(self.config, "part_of_weight", 0.3))
        nw = float(getattr(self.config, "next_weight", 0.5))
        for src_id, item, kids in parents:
            present = [k for k in kids if self.store.has_node(k)]
            if not present:
                continue                      # every chunk skipped/failed → no parent
            if not self.store.has_node(src_id):
                ts = item.created_at or now_iso()
                self.store.add_node(source_node(
                    src_id, source_ref=item.source_ref or "", raw_text=item.text,
                    content_hash=_sha256("source", item.text or ""), ts=ts,
                    title=item.title, ingested_at=now_iso()))
            for k in present:
                self.store.add_edge(Edge(src=k, dst=src_id, etype=EdgeType.PART_OF,
                                         provenance=Provenance.DERIVED,
                                         confidence=1.0, weight=pw))
            for a, b in zip(present, present[1:]):
                self.store.add_edge(Edge(src=a, dst=b, etype=EdgeType.NEXT,
                                         provenance=Provenance.DERIVED,
                                         confidence=1.0, weight=nw))

    # ----------------------------------------------------------------- extract
    def _extract_all(self, items: list[CorpusItem]) -> tuple[list[Extraction], list[str]]:
        def work(item: CorpusItem) -> tuple[Extraction, str | None]:
            try:
                if item.text is None:  # described media (image/audio/pdf/…): perceive it
                    return self.extractor.extract_image(item.image_path, item.label_hint), None
                return self._extract_text(item.text, item.title), None
            except Exception as e:  # noqa: BLE001 — keep the batch alive, record the error
                return Extraction(), f"{item.id}: {e!r}"
        with ThreadPoolExecutor(max_workers=self.config.semaphore_limit) as pool:
            pairs = list(pool.map(work, items))
        extractions = [p[0] for p in pairs]
        errors = [p[1] for p in pairs if p[1]]
        return extractions, errors

    def _extract_text(self, text: str, title: str) -> Extraction:
        return extract_text_sectioned(self.extractor, text, title, self.config.long_doc_chars)

    def _embed_surface(self, item: CorpusItem, ext: Extraction) -> str:
        if item.text is None:  # described media — the extractor's description is the surface
            return ext.description or (item.label_hint or "media")
        text = item.text or ""
        if len(text) > self.config.long_doc_chars:
            return text[:self.config.lead_chars]
        return text

    def _resolve_endpoint(self, surface: str, ent_map: dict[str, str],
                          tag_map: dict[str, str]) -> tuple[str | None, bool]:
        """Resolve a relation endpoint to a graph node. Prefer an entity named in THIS episode
        (mention star), then one of this episode's tags, then an EXISTING tag anywhere in the
        graph (canon's global tag-key map) — so a relation that names a tag the current note
        didn't re-tag still lands on the real tag node instead of minting a shadow 'other'
        entity. Only when nothing matches do we fall back to an 'other' entity anchor.
        Returns (node_id, minted_fallback) so the caller can report the fallback mint."""
        key = surface.lower()
        hit = ent_map.get(key) or tag_map.get(key)
        if hit:
            return hit, False
        existing_tag = self.canon._tag_keys.get(normalize_key(surface))
        if existing_tag:
            return existing_tag, False
        return self.canon.resolve_entity(surface, EntityType.OTHER), True

    # ------------------------------------------------------------------- write
    def _write_episode(self, item: CorpusItem, h: str, ep_id: str, ext: Extraction, vec,
                       ment_vecs: dict, report: IngestReport | None = None) \
            -> tuple[int, int, int]:
        ts = item.created_at or now_iso()
        modality = _modality_of(item.modality)
        raw = item.text  # None for described media → the description becomes the surface
        node = episode_node(ep_id, modality=modality, source_ref=item.source_ref,
                            raw_text=raw, content_hash=h, ts=ts,
                            description=ext.description, ingested_at=now_iso(),
                            media_paths=[item.image_path] if item.image_path else [])
        node.name = item.title or item.id
        self.store.add_node(node)
        self.store.vectors.add("episode", ep_id, vec)
        self.store.add_hash(h, ep_id)

        def _note(msg: str) -> None:
            if report is not None:
                report.notes.append(f"{ep_id}: {msg}")

        # tags → TAGGED_AS (Episode → Tag)
        seen_tags: set[str] = set()
        tag_map: dict[str, str] = {}      # surface(lower) -> tag id, for relation endpoints
        for t in ext.tags:
            tid = self.canon.resolve_tag(t)
            if not tid:
                # normalize_key() reduced the surface to an empty key (all punctuation/
                # stopword-stripped) → it can never become a tag node; surface the drop.
                _note(f"tag {t!r} dropped (normalizes to an empty key)")
                continue
            tag_map[t.lower()] = tid
            self.store.add_edge(Edge(src=ep_id, dst=tid, etype=EdgeType.TAGGED_AS,
                                    provenance=Provenance.EXTRACTED, confidence=1.0))
            if tid not in seen_tags:          # df = # episodes referencing the tag (dedup
                self.canon.bump_doc_frequency(tid)   # duplicate surfaces / re-canonicalized
                seen_tags.add(tid)                   # variants within one episode)
            cname = self.store.get_node(tid).name
            tag_map.setdefault(cname.lower(), tid)
            if cname not in node.tags:
                node.tags.append(cname)

        # entities → an immutable Mention (MENTIONED_IN → episode, RESOLVES_TO → entity)
        ent_map: dict[str, str] = {}
        seen_ents: set[str] = set()
        n_mentions = 0
        for e in ext.entities:
            eid = self.canon.resolve_entity(e.name, e.type)
            if not eid:
                continue
            # stamp the broad glyph category on the anchor; a later, more specific mention
            # upgrades a generic THING (e.g. "Jordan" first seen as a thing, later as a
            # person). getattr keeps the extractor-supplied category optional; the type-
            # derived category is the fallback either way.
            cat = getattr(e, "category", None) or entity_category_for_type(e.type)
            anchor = self.store.get_node(eid)
            if anchor and (anchor.entity_category is None
                           or (anchor.entity_category == EntityCategory.THING
                               and cat != EntityCategory.THING)):
                anchor.entity_category = cat
                anchor.last_modified = ts
                self.store.touch_node(eid)
            ent_map[e.name.lower()] = eid
            mid = f"men_{self._mention_seq:05d}"
            self._mention_seq += 1
            span = self._char_span(item.text, e.name)
            self.store.add_node(mention_node(mid, surface=e.name, etype=e.type,
                                            episode_id=ep_id, ts=ts, char_span=span))
            mv = ment_vecs.get(e.name)
            if mv is not None:
                self.store.vectors.add("mention", mid, mv)
            self.store.add_edge(Edge(src=mid, dst=ep_id, etype=EdgeType.MENTIONED_IN,
                                    provenance=Provenance.EXTRACTED, confidence=1.0))
            self.store.add_edge(Edge(src=mid, dst=eid, etype=EdgeType.RESOLVES_TO,
                                    provenance=Provenance.EXTRACTED, confidence=1.0))
            n_mentions += 1
            if eid not in seen_ents:          # df = # episodes referencing the entity
                self.canon.bump_doc_frequency(eid)
                seen_ents.add(eid)

        # relations → bi-temporal facts (open / confirm / close / supersede). An endpoint
        # resolves to an ENTITY if one was named (mention star), else to a TAG if it names
        # one of this episode's tags OR an existing tag anywhere in the graph (so a
        # tag→entity / tag→tag relation lands on the real tag node instead of a shadow
        # 'other' entity), else falls back to a new 'other' entity anchor.
        n_facts = 0
        for r in ext.relations:
            s, s_minted = self._resolve_endpoint(r.source, ent_map, tag_map)
            t, t_minted = self._resolve_endpoint(r.target, ent_map, tag_map)
            if not s or not t or r.confidence < 0.1:
                continue
            if s == t:
                # self-loop after canonicalization (both endpoints resolved to the same
                # node) — never a real fact; drop it visibly rather than silently.
                _note(f"relation {r.source!r}→{r.target!r} dropped (endpoints resolve to "
                      f"the same node)")
                continue
            for role, surface, minted in (("source", r.source, s_minted),
                                          ("target", r.target, t_minted)):
                if minted:
                    _note(f"relation endpoint {surface!r} ({role}) minted as a new 'other' "
                          f"entity (not a named entity/tag — possible typo)")
            already = set(self.store.edge_rel_tags(s, t))
            seen_here: set[str] = set()
            for label in r.labels[:self.config.max_relation_labels]:
                rid = self.canon.resolve_relation(label)
                if not rid or rid in seen_here:
                    continue
                seen_here.add(rid)
                if rid not in already:
                    self.canon.bump_doc_frequency(rid)
                action = apply_fact(self.store, src=s, dst=t, rel_tag=rid,
                                    status=r.status, at=ts, valid_from=r.valid_from,
                                    valid_to=r.valid_to, provenance=r.provenance,
                                    confidence=r.confidence, episode_id=ep_id)
                # "dispute" mutated an existing edge's disputed_by (a gated close) rather
                # than writing a fact, and "skip" did nothing — neither is a new fact.
                if action not in ("skip", "dispute"):
                    n_facts += 1

        # quantities → a fresh QUANTITY node per OCCURRENCE + one edge from its subject.
        # Deliberately bypasses canon.resolve_entity/apply_fact: every occurrence is a new
        # node/edge, never merged or confirm-collapsed, so distinct amounts and repeated
        # occurrences both stay distinct and summable without regex-parsing names.
        n_quantity = 0
        for qf in ext.facts:
            sid, s_minted = self._resolve_endpoint(qf.subject, ent_map, tag_map)
            if not sid:
                continue
            if s_minted:
                _note(f"quantity subject {qf.subject!r} minted as a new 'other' entity "
                      f"(not a named entity/tag — possible typo)")
            rid = self.canon.resolve_relation(qf.predicate)
            if not rid:
                continue
            qid = f"qty_{self._fact_seq:06d}"
            self._fact_seq += 1
            display = f"{qf.value:g}{(' ' + qf.unit) if qf.unit else ''}"
            self.store.add_node(quantity_node(qid, display=display, value=qf.value,
                                              unit=qf.unit, ts=qf.date or ts))
            self.store.add_edge(Edge(src=sid, dst=qid, etype=EdgeType.RELATED_TO,
                                    provenance=Provenance.EXTRACTED, confidence=0.8,
                                    weight=0.8, rel_tag=rid, valid_at=qf.date or ts,
                                    episode_id=ep_id, created_at=ts))
            n_quantity += 1
        return n_mentions, n_facts, n_quantity

    @staticmethod
    def _char_span(text: str | None, surface: str) -> list[int] | None:
        if not text or not surface:
            return None
        i = text.lower().find(surface.lower())
        return [i, i + len(surface)] if i >= 0 else None

    # --------------------------------------------------------- derived edges
    def _derive_episode_edges(self, new_eps: list[str]) -> None:
        """SHARED_TAG / SHARED_ENTITY (overlap × IDF) + episode-embedding kNN,
        **incrementally**: only pairs involving one of this batch's NEW episodes are
        (re)computed, so derive cost is proportional to new data, not corpus size.

        For a pair touching a new episode the weight is exact — full shared-hub overlap
        × current IDF, identical to what a full recompute would produce. Two accepted
        drifts vs the old full-corpus pass: an old-old pair keeps the (IDF-stale) weight
        it got when the later of the two arrived, and an old episode whose kNN would
        newly include a fresh episode only gets that SIMILAR_TO edge if the fresh
        episode's own top-k catches it (standard incremental-kNN asymmetry)."""
        if not new_eps:
            return
        if getattr(self.config, "shared_edges", True):
            self._shared_edges_for(new_eps, EdgeType.SHARED_TAG)
            self._shared_edges_for(new_eps, EdgeType.SHARED_ENTITY)

        # episode embedding kNN → SIMILAR_TO (new episodes only)
        for eid in new_eps:
            qv = self.store.vectors.get("episode", eid)
            if qv is None:
                continue
            for other, cos in self.store.vectors.search(
                    "episode", qv, k=self.config.episode_knn_k + 1,
                    floor=self.config.episode_knn_floor, exclude={eid}):
                self.store.add_edge(Edge(src=eid, dst=other, etype=EdgeType.SIMILAR_TO,
                                        provenance=Provenance.SIMILAR,
                                        confidence=round(cos, 3), weight=round(cos, 3)))

    def _episode_hubs(self, ep_id: str, etype: EdgeType) -> set[str]:
        """The hub nodes an episode shares through: its tags, or (via the mention star)
        its resolved entities."""
        if etype == EdgeType.SHARED_TAG:
            return {nbr for nbr, _d in self.store.neighbors(
                ep_id, etypes={EdgeType.TAGGED_AS}, direction="out")}
        ents: set[str] = set()
        for mid, _d in self.store.neighbors(ep_id, etypes={EdgeType.MENTIONED_IN},
                                            direction="in"):
            for ent, _d2 in self.store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                                 direction="out"):
                ents.add(ent)
        return ents

    def _hub_members(self, hub: str, etype: EdgeType) -> list[str]:
        """Valid episodes attached to a hub, capped at the `shared_hub_cap` most recent
        (a hub big enough to hit the cap has high df, hence near-zero IDF weight)."""
        if etype == EdgeType.SHARED_TAG:
            eps = {ep for ep, _d in self.store.neighbors(
                hub, etypes={EdgeType.TAGGED_AS}, direction="in")}
        else:
            eps = self.store.entity_episodes(hub)
        members = [e for e in eps
                   if (n := self.store.get_node(e)) and n.valid]
        cap = int(getattr(self.config, "shared_hub_cap", 0))
        if cap and len(members) > cap:
            members.sort(key=lambda e: (self.store.get_node(e).created_at, e))
            return members[-cap:]
        return members

    def _shared_edges_for(self, new_eps: list[str], etype: EdgeType) -> None:
        hubs_of: dict[str, set[str]] = {}

        def hubs(ep: str) -> set[str]:
            got = hubs_of.get(ep)
            if got is None:
                got = hubs_of[ep] = self._episode_hubs(ep, etype)
            return got

        done: set[tuple[str, str]] = set()
        for ep in new_eps:
            for hub in hubs(ep):
                for other in self._hub_members(hub, etype):
                    if other == ep:
                        continue
                    pair = (ep, other) if ep <= other else (other, ep)
                    if pair in done:
                        continue
                    done.add(pair)
                    shared = hubs(ep) & hubs(other)
                    if len(shared) < self.config.shared_min_overlap:
                        continue
                    w = sum(self.canon.idf_weight(h) for h in sorted(shared))
                    self.store.add_edge(Edge(src=pair[0], dst=pair[1], etype=etype,
                                            provenance=Provenance.DERIVED,
                                            confidence=min(1.0, w / 3.0),
                                            weight=round(w, 3)))
