"""Ingestion pipeline (docs/ARCHITECTURE.md §6).

Per object: intake → SHA256 cache check (skip/supersede) → normalize → extract
(directly from raw content, no summary) → canonicalize tags/entities → embed →
write ObjectNode + TagNodes + EntityNodes + edges (with provenance + confidence).
Then a global pass derives ObjectNode↔ObjectNode edges (shared tags/entities +
embedding kNN), since the corpus has no free hyperlinks.

LLM extraction fans out under a bounded-concurrency semaphore; all graph mutation
happens sequentially in the main thread so shared state stays consistent.
"""
from __future__ import annotations

import hashlib
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .canonicalize import Canonicalizer
from .config import Config
from .corpus import CorpusItem
from .embedders import Embedder
from .extractors import Extraction, Extractor
from .models import (Edge, EdgeType, EntityType, Modality, Provenance, object_node)
from .store import GraphStore, now_iso


@dataclass
class IngestReport:
    ingested: int = 0
    skipped: int = 0
    superseded: int = 0
    failed: int = 0
    extraction_failures: int = 0
    seconds: float = 0.0
    extractor: str = ""
    embedder: str = ""
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        warn = (f"  ⚠ {self.extraction_failures} extraction failures"
                if self.extraction_failures else "")
        return (f"ingested={self.ingested} skipped={self.skipped} "
                f"superseded={self.superseded} failed={self.failed} "
                f"in {self.seconds:.1f}s  (extractor={self.extractor}, "
                f"embedder={self.embedder}){warn}")


def _sha256(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
    return h.hexdigest()


class Ingestor:
    def __init__(self, store: GraphStore, extractor: Extractor, embedder: Embedder,
                 canon: Canonicalizer, config: Config):
        self.store = store
        self.extractor = extractor
        self.embedder = embedder
        self.canon = canon
        self.config = config

    # ------------------------------------------------------------------ public
    def ingest(self, items: list[CorpusItem]) -> IngestReport:
        t0 = time.time()
        report = IngestReport(extractor=self.extractor.name, embedder=self.embedder.name)

        # 1. intake + cache/supersede decisions (sequential, cheap)
        pending: list[tuple[CorpusItem, str, str, str | None]] = []  # item, hash, obj_id, supersedes
        for item in items:
            content = item.text if item.modality == "text" else (item.image_path or "")
            h = _sha256(item.modality, content)
            if h in self.store.hash_cache:
                report.skipped += 1
                continue
            base_id = f"obj_{item.id}"
            supersedes = None
            obj_id = base_id
            if self.store.has_node(base_id):
                existing = self.store.get_node(base_id)
                if existing.content_hash == h:
                    report.skipped += 1
                    continue
                # changed content → new versioned node, soft-invalidate the old
                v = 1
                while self.store.has_node(f"{base_id}_v{v}"):
                    v += 1
                obj_id = f"{base_id}_v{v}"
                supersedes = base_id
            pending.append((item, h, obj_id, supersedes))

        if not pending:
            report.seconds = time.time() - t0
            return report

        # 2. extract concurrently (bounded semaphore on the LLM calls)
        extractions, errors = self._extract_all([p[0] for p in pending])
        report.extraction_failures = len(errors)
        if errors:
            # surface systemic failures (e.g. a bad/absent API key) instead of
            # silently building an empty graph
            report.notes.append(f"extraction failed for {len(errors)} item(s); "
                                f"first error: {errors[0]}")

        # 3. embed object surfaces in one batch (raw text / image description)
        surfaces = [self._embed_surface(item, ext)
                    for (item, *_), ext in zip(pending, extractions)]
        obj_vecs = self.embedder.embed(surfaces)

        # 3b. batch-embed every tag/entity surface up front (huge speedup vs. per-resolve)
        tag_ent_surfaces: list[str] = []
        for ext in extractions:
            tag_ent_surfaces.extend(ext.tags)
            tag_ent_surfaces.extend(e.name for e in ext.entities)
            for r in ext.relations:
                tag_ent_surfaces.append(r.source)
                tag_ent_surfaces.append(r.target)
                tag_ent_surfaces.extend(r.labels)   # relationship-label embeddings too
        self.canon.prime_embeddings(tag_ent_surfaces)

        # 4. write each object (sequential)
        for (item, h, obj_id, supersedes), ext, vec in zip(pending, extractions, obj_vecs):
            try:
                self._write_object(item, h, obj_id, supersedes, ext, vec)
                report.ingested += 1
                if supersedes:
                    report.superseded += 1
            except Exception as e:  # noqa: BLE001
                report.failed += 1
                report.notes.append(f"{item.id}: {e!r}")

        # 5. derive ObjectNode↔ObjectNode edges across the whole graph
        self._derive_object_edges()

        report.seconds = time.time() - t0
        return report

    # ----------------------------------------------------------------- extract
    def _extract_all(self, items: list[CorpusItem]) -> tuple[list[Extraction], list[str]]:
        """Returns (extractions, error_messages). A failed item degrades to an empty
        Extraction so the batch survives, but the error is recorded (not swallowed)."""
        def work(item: CorpusItem) -> tuple[Extraction, str | None]:
            try:
                if item.modality == "image":
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
        """Section-by-section for very long docs (§9 risk 4), else one shot."""
        if len(text) <= self.config.long_doc_chars:
            return self.extractor.extract_text(text, title)
        chunk = self.config.long_doc_chars
        merged = Extraction()
        for i in range(0, min(len(text), chunk * 6), chunk):  # cap at ~6 sections
            part = self.extractor.extract_text(text[i:i + chunk], title if i == 0 else "")
            merged.merge(part)
        return merged

    def _embed_surface(self, item: CorpusItem, ext: Extraction) -> str:
        if item.modality == "image":
            return ext.description or (item.label_hint or "an image")
        text = item.text or ""
        if len(text) > self.config.long_doc_chars:
            return text[:self.config.lead_chars]   # lead section for long docs
        return text

    # ------------------------------------------------------------------- write
    def _write_object(self, item: CorpusItem, h: str, obj_id: str,
                      supersedes: str | None, ext: Extraction, vec) -> None:
        ts = now_iso()
        modality = Modality.IMAGE if item.modality == "image" else Modality.TEXT
        raw = None if item.modality == "image" else item.text
        node = object_node(obj_id, modality=modality, source_ref=item.source_ref,
                           raw_text=raw, content_hash=h, ts=ts,
                           description=ext.description)
        node.name = item.title or item.id
        self.store.add_node(node)
        self.store.vectors.add("object", obj_id, vec)
        self.store.hash_cache[h] = obj_id
        if supersedes:
            self._retract(supersedes)              # undo the old version's df/hash
            self.store.supersede_node(supersedes, obj_id)

        # tags → TAGGED_AS
        for t in ext.tags:
            tid = self.canon.resolve_tag(t)
            if not tid:
                continue
            self.store.add_edge(Edge(src=obj_id, dst=tid, etype=EdgeType.TAGGED_AS,
                                    provenance=Provenance.EXTRACTED, confidence=1.0))
            self.canon.bump_doc_frequency(tid)
            cname = self.store.get_node(tid).name
            if cname not in node.tags:
                node.tags.append(cname)

        # entities → MENTIONS (+ provenance back-pointer)
        ent_map: dict[str, str] = {}
        for e in ext.entities:
            eid = self.canon.resolve_entity(e.name, e.type)
            if not eid:
                continue
            ent_map[e.name.lower()] = eid
            self.store.add_edge(Edge(src=obj_id, dst=eid, etype=EdgeType.MENTIONS,
                                    provenance=Provenance.EXTRACTED, confidence=0.9))
            self.canon.bump_doc_frequency(eid)
            en = self.store.get_node(eid)
            if obj_id not in en.provenance_objs:
                en.provenance_objs.append(obj_id)

        # relations → directed RELATED_TO between entities, labelled with the
        # consolidated relationship-tag set (gated by confidence)
        for r in ext.relations:
            s = ent_map.get(r.source.lower()) or self.canon.resolve_entity(r.source, EntityType.OTHER)
            t = ent_map.get(r.target.lower()) or self.canon.resolve_entity(r.target, EntityType.OTHER)
            if not s or not t or s == t:
                continue
            if r.confidence < 0.1:   # link gate: drop near-zero-confidence inferences
                continue
            # consolidate each free-form label into a canonical relationship-tag node,
            # and emit ONE directed edge per relation (rev 4 — parallel typed edges),
            # so each carries its own provenance/confidence/timestamp
            already = set(self.store.edge_rel_tags(s, t))
            seen_here: set[str] = set()
            for label in r.labels[:self.config.max_relation_labels]:
                rid = self.canon.resolve_relation(label)
                if not rid or rid in seen_here:
                    continue
                seen_here.add(rid)
                # idempotent df: bump only when this (s→t, rid) edge is genuinely new,
                # so re-ingesting identical content can't double-count relation frequency
                if rid not in already:
                    self.canon.bump_doc_frequency(rid)
                self.store.add_edge(Edge(src=s, dst=t, etype=EdgeType.RELATED_TO,
                                        provenance=r.provenance, confidence=r.confidence,
                                        weight=r.confidence, rel_tag=rid))

    def _retract(self, old_id: str) -> None:
        """Undo a soon-to-be-superseded object's side effects: decrement the
        doc_frequency it contributed to each tag/entity, and free its content hash
        so identical content can be re-ingested later."""
        old = self.store.get_node(old_id)
        if not old:
            return
        for nbr, data in self.store.neighbors(old_id, valid_only=False):
            if data["etype"] in (EdgeType.TAGGED_AS.value, EdgeType.MENTIONS.value):
                n = self.store.get_node(nbr)
                if n and n.doc_frequency > 0:
                    n.doc_frequency -= 1
        if old.content_hash and old.content_hash in self.store.hash_cache:
            self.store.hash_cache.pop(old.content_hash, None)

    # --------------------------------------------------------- derived edges
    def _derive_object_edges(self) -> None:
        """SHARED_TAG / SHARED_ENTITY (overlap × IDF) + object embedding kNN.

        Runs over the full valid-object set each ingest. At MVP scale (one-shot
        ingest of ~200 objects) this is paid once; incremental scoping to only the
        newly-written objects is the noted optimization if ingest goes online.
        """
        from .models import NodeType
        objects = [n.id for n in self.store.nodes_of_type(NodeType.OBJECT)]  # valid only
        invalid = {n.id for n in self.store.nodes_of_type(NodeType.OBJECT, valid_only=False)
                   if not n.valid}

        # inverted indexes tag/entity -> objects (via TAGGED_AS / MENTIONS)
        tag_objs: dict[str, list[str]] = defaultdict(list)
        ent_objs: dict[str, list[str]] = defaultdict(list)
        for oid in objects:
            for nbr, data in self.store.neighbors(oid):
                if data["etype"] == EdgeType.TAGGED_AS.value:
                    tag_objs[nbr].append(oid)
                elif data["etype"] == EdgeType.MENTIONS.value:
                    ent_objs[nbr].append(oid)

        self._add_shared_edges(tag_objs, EdgeType.SHARED_TAG)
        self._add_shared_edges(ent_objs, EdgeType.SHARED_ENTITY)

        # object embedding kNN → SIMILAR_TO (skip superseded objects)
        for oid in objects:
            qv = self.store.vectors.get("object", oid)
            if qv is None:
                continue
            for other, cos in self.store.vectors.search(
                    "object", qv, k=self.config.object_knn_k + 1,
                    floor=self.config.object_knn_floor, exclude={oid} | invalid):
                self.store.add_edge(Edge(src=oid, dst=other, etype=EdgeType.SIMILAR_TO,
                                        provenance=Provenance.SIMILAR,
                                        confidence=round(cos, 3), weight=round(cos, 3)))

    def _add_shared_edges(self, hub_objs: dict[str, list[str]], etype: EdgeType) -> None:
        pair_weight: dict[tuple[str, str], float] = defaultdict(float)
        pair_count: dict[tuple[str, str], int] = defaultdict(int)  # # shared hubs
        for hub_id, objs in hub_objs.items():
            if len(objs) < 2:
                continue
            w = self.canon.idf_weight(hub_id)   # specificity: rare shared tags weigh more
            uniq = sorted(set(objs))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    pair_weight[(uniq[i], uniq[j])] += w
                    pair_count[(uniq[i], uniq[j])] += 1
        for (a, b), w in pair_weight.items():
            if pair_count[(a, b)] < self.config.shared_min_overlap:
                continue  # require at least this many shared tags/entities
            self.store.add_edge(Edge(src=a, dst=b, etype=etype,
                                    provenance=Provenance.DERIVED,
                                    confidence=min(1.0, w / 3.0), weight=round(w, 3)))
