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
                     ProviderUnavailable, StoreError, UnsupportedMedia)
from .ingest import _BARE_URL
from .extractors import (SUPPORTED_IMAGE_EXTS, Extraction, ExtractedEntity,
                         UsageMeter)
from .llm_client import SUPPORTED_KINDS
from .models import (Belief, Edge, EdgeType, EntityType, Modality, NodeType,
                     Provenance, entity_category_for_type)

_STUB = ("profile", "rebuild", "reingest", "maintain", "ensure_model")

_DATE10 = re.compile(r"^\d{4}-\d{2}-\d{2}")
_BARE_YEAR = re.compile(r"^\d{4}$")

# Attachment extensions that perceive as images. Anything else is out of scope for
# perception; it defaults to FILE so a non-image attachment is not mislabeled IMAGE.
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".heic", ".heif", ".tif",
               ".tiff"}


def _repo_marker_key(path: str) -> str:
    """Sync-marker key for a repo: its absolute on-disk path (so two repos with the same
    basename don't share a last-SHA)."""
    return os.path.abspath(os.path.expanduser(path.rstrip("/")))


def _attachment_modality(path: str | None) -> str:
    """Sniff an attachment's extension to a CorpusItem modality label. Image types →
    'image' (perceived by the VLM); everything else → 'file' (out of scope, stored not
    perceived) so it lands as FILE via _modality_of rather than being mislabeled IMAGE."""
    ext = os.path.splitext(path or "")[1].lower()
    return "image" if ext in _IMAGE_EXTS else "file"


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
    media_paths: list[str] = field(default_factory=list)


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


@dataclass
class ImportReport:
    """Result of a cold-start chat-history import (Engine.import_conversations)."""
    source: str = ""                # resolved source ("chatgpt"|"claude"|"gemini")
    conversations: int = 0          # canonical conversations mapped from the export
    episodes_ingested: int = 0      # NEW session episodes written this run
    skipped: int = 0                # session episodes already present (idempotent re-run)
    images_perceived: int = 0       # inline images given a real vision description
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


@dataclass
class VaultImportReport:
    """Result of an Obsidian vault import (Engine.import_vault). A vault is human-curated, so
    the interesting counts are the STRUCTURE we wired from that curation: how many wikilinks
    resolved to a real note (`links_resolved`) vs pointed at a note that doesn't exist
    (`links_unresolved`, skipped silently), and how many `TAGGED_AS` links we laid down."""
    notes: int = 0                  # notes parsed + pushed through ingest
    episodes_ingested: int = 0      # NEW note episodes written this run
    skipped: int = 0                # note episodes already present (idempotent re-run)
    links_resolved: int = 0         # wikilinks wired to a real target Episode (HYPERLINKS_TO)
    links_unresolved: int = 0       # wikilinks whose target note is absent (skipped, no stub)
    images_perceived: int = 0       # inline images given a real vision description
    tags: int = 0                   # TAGGED_AS links wired deterministically
    errors: list[str] = field(default_factory=list)
    seconds: float = 0.0


class _StructureOnlyExtractor:
    """A no-op extractor for structure-only vault import (extract=False): every pass returns
    an empty Extraction, so an Episode is still created + embedded (local bge, not an LLM) and
    the deterministic wikilink/tag structure carries the graph, but ZERO LLM calls are made.
    Viable only for Obsidian, where the explicit links keep the graph connected without any
    inferred concepts (a chat/URL import would fall apart structure-only)."""
    name = "structure_only"

    def __init__(self):
        self.meter = UsageMeter()

    def extract_text(self, text: str, title: str = "") -> Extraction:
        return Extraction()

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return Extraction()

    def extract_url(self, url: str) -> Extraction:
        return Extraction()


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

    def extract_url(self, url: str) -> Extraction:
        domain = re.sub(r"^https?://", "", url).split("/")[0]
        tags = [t for t in re.split(r"[.\-/]", domain) if len(t) >= 3][:3]
        return Extraction(entities=[ExtractedEntity(name=domain)], tags=tags,
                          description=f"A saved web page at {domain}.", page_title=domain)

    def extract_commit(self, message: str, diff: str) -> Extraction:
        subject = (message or "").strip().splitlines()[0].strip() if message else ""
        ents = []
        seen = set()
        for w in re.findall(r"\b[A-Z][a-z]{2,}\b", message or ""):
            if w.lower() not in seen:
                seen.add(w.lower())
                ents.append(ExtractedEntity(name=w))
        return Extraction(entities=ents[:5], tags=["commit"],
                          description=(subject or "A git commit."), source_text=diff or None)

    def extract_repo(self, signals: dict) -> Extraction:
        name = (signals or {}).get("name") or "project"
        libs = (signals or {}).get("libraries") or []
        ents = [ExtractedEntity(name=name)] + [ExtractedEntity(name=l) for l in libs[:5]]
        return Extraction(entities=ents, tags=["project"],
                          description=f"A code project named {name}.")


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
        # Open-boundary compatibility repair: link episodes written before the URL was
        # preserved as raw_text recover it from source_ref (idempotent, no-op on new stores).
        self.repair_link_raw_text()
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
        if not isinstance(note.text, str):
            raise InvalidInput("note.text must be a string")
        attachments = list(note.attachments or [])
        media_paths = list(note.media_paths or attachments)
        readable_media = attachments or media_paths
        has_text = bool(note.text.strip())
        # a media-only note (empty text) is valid when it carries attachments; otherwise
        # empty text is nothing to ingest.
        if not has_text and not readable_media:
            raise InvalidInput("note.text must be a non-empty string")
        if not note.created_at:
            raise InvalidInput("note.created_at is required (ISO-8601)")
        # salt the id with attachments so two media-only notes (empty text) at the same
        # timestamp don't collide on one nid and dedup each other away.
        nid = hashlib.sha256(
            f"{note.created_at}\n{note.text}\n{chr(0).join(media_paths)}"
            .encode("utf-8")).hexdigest()[:16]
        # A note whose entire text is a bare URL (and carries no attachments) is modality
        # LINK: the modality routes through the extractor's extract_url path (fetch +
        # subject-scoped extraction) and source_ref is the stripped URL so the fetch and
        # the SOURCE-node provenance both have it. The byte-exact capture stays in text →
        # raw_text: when title resolution fails-soft, clients still have the URL to render
        # instead of an untitled empty card.
        stripped = note.text.strip()
        is_link = has_text and not readable_media and bool(_BARE_URL.match(stripped))
        if is_link:
            item = CorpusItem(id=nid, modality="link", source_ref=stripped,
                              text=note.text, created_at=note.created_at)
        else:
            # Sniff the attachment's real type so an image attachment is labeled IMAGE — not
            # the blanket "image" for every attachment (which mislabels PDFs etc.), and not
            # "text" when a caption rides along with an image.
            #   - media-only (no text)     → text=None routes through the perception path;
            #   - text + image attachment  → CO-PERCEPTION: caption text AND image are both
            #     perceived and merged (kg/ingest.py:_extract_all). text stays the caption.
            #   - text, no attachment      → a plain text note.
            if readable_media:
                modality = _attachment_modality(readable_media[0])
                # Fast-fail what the perception path can't serve, at the API boundary
                # (clear UnsupportedMedia) rather than deep in extraction:
                #   - an image in a format no vision provider accepts (.heic/.bmp/.tif…);
                #   - a media-only non-image file (PDF/audio/…), which would otherwise be
                #     routed through the vision path as bogus image bytes. A captioned
                #     non-image file stays valid: the caption is extracted, the file is
                #     stored-not-perceived (by design).
                ext = os.path.splitext(readable_media[0])[1].lower()
                if modality == "image" and ext not in SUPPORTED_IMAGE_EXTS:
                    raise UnsupportedMedia(
                        f"can't process image format {ext!r} — convert to JPEG, PNG, "
                        "WebP, or GIF")
                if modality == "file" and not has_text:
                    raise UnsupportedMedia(
                        f"can't extract content from a {ext or '(no extension)'!r} "
                        "attachment — only images (JPEG, PNG, WebP, GIF) can be "
                        "perceived; add a note text or convert the file")
            else:
                modality = "text"
            item = CorpusItem(id=nid, modality=modality,
                              source_ref=f"{note.source}/{nid}",
                              text=note.text if has_text else None,
                              image_path=readable_media[0] if readable_media else None,
                              created_at=note.created_at)
        item.media_paths = media_paths
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

    def import_conversations(self, path: str, source: str = "auto") -> ImportReport:
        """Cold-start bulk-import a chat-history export (ChatGPT / Claude / Gemini) into the
        graph (BUILD BRIEF). `source` is a validated, closed set — "chatgpt" | "claude" |
        "gemini" | "auto"; "auto" sniffs the export and resolves to one of the three or
        raises. Any other value → InvalidInput. Per-source parsing lives in kg/imports/; the
        export is normalized to session CorpusItems and pushed through the SAME ingest path a
        live note uses, so it is idempotent + resumable (content-hash dedup skips already-
        ingested sessions → a re-run after interruption resumes, ingesting nothing new).

        Runs in flushed batches so progress is observable and a crash loses at most one batch;
        the CueGated extractor keeps the bulk extraction cost bounded. Cold start is large and
        slow — the daemon should invoke this as a background job (see the brainbrain follow-up
        note below), not block on it.

        FOLLOW-UP (deferred, not this repo): expose this over the wire by adding an
        `import.conversations` verb to brainbrain/engine/daemon.py + bumping PROTOCOL_MINOR
        with a capability-probe (mirrors the ingest.repo / media additive pattern), and run it
        as an async background job with progress notifications (capture already went async)."""
        import time
        from .imports import SUPPORTED_SOURCES, build_corpus_items
        self._check()
        if source not in ("auto", *SUPPORTED_SOURCES):
            raise InvalidInput(
                f"source must be one of auto/{'/'.join(SUPPORTED_SOURCES)}, got {source!r}")
        t0 = time.time()
        report = ImportReport(source="" if source == "auto" else source)
        try:
            resolved, conversations, items, stats = build_corpus_items(
                path, source, self._g.extractor)
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise StoreError(f"import failed: {e}") from e
        report.source = resolved
        report.conversations = len(conversations)
        report.images_perceived = stats.images_perceived

        # Ingest in flushed batches: idempotent (hash-cache skips already-ingested sessions),
        # resumable (a re-run after a crash resumes), and observable (progress is logged and
        # the store is checkpointed per batch).
        batch = max(1, int(getattr(self._g.config, "ingest_flush_every", 200)) or 200)
        for start in range(0, len(items), batch):
            chunk = items[start:start + batch]
            try:
                r = self._g.ingest(chunk)
            except Exception as e:              # noqa: BLE001 — one bad batch must not lose the rest
                report.errors.append(f"batch@{start}: {e!r}")
                continue
            report.episodes_ingested += r.ingested
            report.skipped += r.skipped
            if r.notes:
                report.errors.extend(r.notes[:3])
            self._g.save()
            self._log("info", f"import {resolved}: {report.episodes_ingested} episodes "
                              f"ingested, {report.skipped} skipped "
                              f"({start + len(chunk)}/{len(items)} sessions)")
        self._g.save()
        report.seconds = round(time.time() - t0, 2)
        return report

    def import_vault(self, path: str, extract: bool = True) -> VaultImportReport:
        """Cold-import an Obsidian vault into the graph (BUILD BRIEF). A vault is ALREADY a
        personal knowledge graph — the author curated the links (`[[wikilinks]]`), the topics
        (`#tags` / frontmatter `tags:`), and the identity (`aliases:`) — so this leans on that
        explicit structure rather than inferring everything, yielding a dense, well-connected
        graph on day one.

        One `.md` file → one text Episode (frontmatter stripped, title from H1/filename, images
        perceived-and-inlined). Two novel bits over the chat importers:

          * Two-pass wikilink resolution — you can't point `[[B]]` at B's Episode id until B is
            ingested, so pass 1 ingests every note (building a name/path/alias → Episode map)
            and pass 2 resolves each note's wikilinks against that map and wires directional
            HYPERLINKS_TO edges (the deterministic reserved type; the engine symmetrizes for
            PPR so backlinks come free). A link to a note that doesn't exist is skipped
            silently — never stubbed, never a crash.
          * Deterministic tags — `#tags` and frontmatter `tags:` wire straight to TAGGED_AS via
            the normal tag canonicalization (so `#ml` ↔ `#machine-learning` still merge and
            share nodes with tags from chats/URLs), with NO LLM in the loop.

        `extract=True` (default) also runs the normal LLM body extraction so notes join the
        concept bridge to chats/URLs/code, not only each other (lean on CueGated to bound cost);
        `extract=False` is structure-only (wikilinks + tags + embeddings, no LLM) — viable only
        because the explicit links keep the graph connected without any inferred concepts.

        Idempotent + resumable: the content-hash cache means a re-import reprocesses only changed
        notes, and pass 2 re-wires every note's structure each run (structural edges collapse by
        identity, so re-wiring is a no-op for unchanged links).

        FOLLOW-UP (deferred, brainbrain, not this repo): expose over the wire with an
        `import.vault` daemon verb + a PROTOCOL_MINOR bump + capability-probe, run as an async
        background job with progress notifications — mirroring the import.conversations /
        ingest.repo additive pattern noted above."""
        import time
        from .imports import obsidian
        from .imports.normalize import NormalizeStats
        self._check()
        report = VaultImportReport()
        t0 = time.time()
        if not obsidian.is_vault(path):
            raise InvalidInput(
                f"{path!r} is not an Obsidian vault (a directory of .md notes, usually with "
                f"an .obsidian/ config dir)")

        notes = obsidian.parse_vault(path)
        report.notes = len(notes)

        # extract=False → swap in the model-free extractor for the whole ingest, so a
        # structure-only import makes ZERO LLM calls (images are inlined as filename
        # placeholders, note bodies produce empty extractions). Restored in `finally`.
        real_extractor = self._g.extractor
        if not extract:
            self._g.extractor = _StructureOnlyExtractor()
        try:
            stats = NormalizeStats()
            items = obsidian.to_corpus_items(notes, self._g.extractor, stats, perceive=extract)
            report.images_perceived = stats.images_perceived

            # PASS 1 — ingest every note (flushed batches: idempotent via the hash-cache,
            # resumable after a crash, observable per batch).
            batch = max(1, int(getattr(self._g.config, "ingest_flush_every", 200)) or 200)
            for start in range(0, len(items), batch):
                chunk = items[start:start + batch]
                try:
                    r = self._g.ingest(chunk)
                except Exception as e:          # noqa: BLE001 — one bad batch must not lose the rest
                    report.errors.append(f"batch@{start}: {e!r}")
                    continue
                report.episodes_ingested += r.ingested
                report.skipped += r.skipped
                if r.notes:
                    report.errors.extend(r.notes[:3])
                self._g.save()
                self._log("info", f"import vault: {report.episodes_ingested} notes ingested, "
                                  f"{report.skipped} skipped "
                                  f"({start + len(chunk)}/{len(items)})")

            # PASS 2 — now every note's Episode exists, wire the human-authored structure:
            # deterministic tags (TAGGED_AS) + resolved wikilinks (HYPERLINKS_TO).
            resolver = obsidian.build_resolver(notes)
            for note in notes:
                ep = self._note_episode_id(note.item_id)
                if ep is None:
                    continue                    # its whole batch failed above — nothing to wire
                report.tags += self._wire_tags(ep, note.tags, note.created_at)
                res, unres = self._wire_wikilinks(ep, note, resolver)
                report.links_resolved += res
                report.links_unresolved += unres
            self._g.save()
        except EngineError:
            raise
        except Exception as e:                  # noqa: BLE001 — taxonomy boundary (§7)
            raise StoreError(f"vault import failed: {e}") from e
        finally:
            self._g.extractor = real_extractor

        report.seconds = round(time.time() - t0, 2)
        return report

    def _note_episode_id(self, item_id: str) -> str | None:
        """The Episode id representing a note, resolved from the store (ingest owns the exact
        id). A short note is one Episode `ep_<id>`; a long one is chunked, so its first chunk
        `ep_<id>#c000` stands in as the link anchor; a re-import whose content CHANGED appended
        a `_vN` version, so the latest version wins. None → the note never landed."""
        store = self._g.store
        base = f"ep_{item_id}"
        if store.has_node(base):
            last = base
            v = 1
            while store.has_node(f"{base}_v{v}"):   # content changed on a re-import → newest wins
                last = f"{base}_v{v}"
                v += 1
            return last
        c0 = f"{base}#c000"
        return c0 if store.has_node(c0) else None

    def _wire_tags(self, ep_id: str, tags: list[str], ts: str | None) -> int:
        """Deterministically wire a note's tags → TAGGED_AS edges, canonicalizing each surface
        through the normal tag path (so `#ml` ↔ `#machine-learning` merge, and a vault tag lands
        on the SAME node a chat/URL tag would). No LLM. Returns the count wired. Idempotent — a
        re-run re-adds the identical structural edge, which collapses by identity."""
        canon = self._g.canon
        store = self._g.store
        node = store.get_node(ep_id)
        # tids already TAGGED_AS this episode (from a prior import's pass 2, or the LLM body
        # pass when extract=True) — so a re-import doesn't double-bump doc-frequency.
        existing = {tid for tid, _d in store.neighbors(ep_id, etypes={EdgeType.TAGGED_AS},
                                                        direction="out")}
        wired = 0
        seen: set[str] = set()          # distinct canonical tids on this episode this run
        for surface in tags:
            tid = canon.resolve_tag(surface)
            if not tid or tid in seen:  # two surfaces (#ml / #machine-learning) → one node
                continue
            seen.add(tid)
            store.add_edge(Edge(src=ep_id, dst=tid, etype=EdgeType.TAGGED_AS,
                                provenance=Provenance.DERIVED, confidence=1.0))
            if tid not in existing:     # genuinely new link → count it toward df once
                canon.bump_doc_frequency(tid)
            cname = store.get_node(tid).name
            if node is not None and cname not in node.tags:
                node.tags.append(cname)
                store.touch_node(ep_id)
            wired += 1
        return wired

    def _wire_wikilinks(self, ep_id: str, note, resolver) -> tuple[int, int]:
        """Resolve a note's wikilinks/embeds against the vault resolver and wire HYPERLINKS_TO
        (Episode → Episode) for each that lands on a real note. Human-authored, so DERIVED /
        confidence 1.0. Directional (the engine symmetrizes for PPR → backlinks free); a
        target that doesn't exist, or resolves to this same note, is skipped silently. Returns
        (resolved, unresolved). De-duped per target so a note linked twice counts once."""
        store = self._g.store
        resolved = unresolved = 0
        seen: set[str] = set()
        for link in note.wikilinks:
            target_item = resolver.resolve(link.target)
            if target_item is None or target_item == note.item_id:
                if target_item is None:
                    unresolved += 1
                continue
            tgt_ep = self._note_episode_id(target_item)
            if tgt_ep is None or tgt_ep == ep_id or tgt_ep in seen:
                continue
            seen.add(tgt_ep)
            store.add_edge(Edge(src=ep_id, dst=tgt_ep, etype=EdgeType.HYPERLINKS_TO,
                                provenance=Provenance.DERIVED, confidence=1.0))
            resolved += 1
        return resolved, unresolved

    def ingest_repo(self, path: str, since: str | None = None,
                    max_commits: int = 200) -> dict:
        """Ingest a git repo's memory: its commit history (event layer) + a repo summary
        (bridge) + its current source files (thin state layer) into this graph, all
        me-anchored. Idempotent per commit SHA; on a re-sync only `last..HEAD` new commits
        and the changed-file set are processed (a stored last-SHA per repo is the marker).

        Returns a small report dict (repo, head, counts). Branches / history rewrites are
        out of scope — linear history on the current branch is assumed.

        FOLLOW-UP (deferred, not this repo): expose this over the wire by adding an
        `ingest.repo` verb to brainbrain/engine/daemon.py and bumping PROTOCOL_MINOR with a
        capability-probe, per the additive v1.1 pattern (mirrors how ingest of media/links
        was surfaced). Nothing here needs to change for that."""
        self._check()
        from .code import ingest_repo as _ingest_repo
        from .code.git import GitError
        marker = self._code_sync_load()
        try:
            report = _ingest_repo(self._g, path, since=since,
                                  after_sha=marker.get(_repo_marker_key(path)),
                                  max_commits=max_commits)
        except GitError as e:
            raise InvalidInput(str(e)) from e
        except Exception as e:  # noqa: BLE001 — taxonomy boundary (§7)
            raise StoreError(f"repo ingest failed: {e}") from e
        if report.head_sha:                     # advance the sync marker to HEAD
            marker[_repo_marker_key(path)] = report.head_sha
            marker[report.repo] = report.head_sha
            self._code_sync_save(marker)
        self._g.save()
        return {"repo": report.repo, "head": report.head_sha,
                "summarized": report.summarized,
                "commits_ingested": report.commits_ingested,
                "commits_seen": report.commits_seen,
                "files_ingested": report.files_ingested,
                "files_superseded": report.files_superseded,
                "next_edges": report.next_edges, "modifies_edges": report.modifies_edges,
                "notes": report.notes}

    def _code_sync_path(self) -> str:
        return os.path.join(self._data_dir, "code_sync.json")

    def _code_sync_load(self) -> dict:
        try:
            with open(self._code_sync_path(), encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _code_sync_save(self, marker: dict) -> None:
        try:
            with open(self._code_sync_path(), "w", encoding="utf-8") as f:
                json.dump(marker, f)
        except OSError as e:  # noqa: BLE001 — a marker write failure must not lose the ingest
            self._log("warn", f"code_sync marker write failed: {e}")

    def repair_link_raw_text(self) -> int:
        """Backfill raw_text on LINK episodes written by builds that stored bare-URL
        captures with text=None. Such an episode whose page-title fetch also failed has
        neither a title nor raw_text, so clients render it as an untitled empty card.
        source_ref still holds the captured URL — copy it into raw_text.

        Idempotent, and deliberately narrow: only valid LINK episodes with EMPTY raw_text
        and an http(s) source_ref are touched; titles, descriptions, timestamps and any
        non-empty raw_text are never overwritten. Runs at the engine-open boundary."""
        store = self._g.store
        repaired = 0
        for node in store.nodes.values():
            if node.ntype != NodeType.EPISODE or not node.valid:
                continue
            if node.modality is not Modality.LINK:
                continue
            if (node.raw_text or "").strip():
                continue
            ref = (node.source_ref or "").strip()
            if not _BARE_URL.match(ref):
                continue
            node.raw_text = ref
            store.touch_node(node.id)
            repaired += 1
        if repaired:
            # the BM25 corpus is built over episode raw_text and cached against
            # episode_version — bump it so the repaired text becomes searchable.
            store.episode_version += 1
            self._g.save()
            self._log("info", f"link raw_text repair: {repaired} episode(s) recovered")
        return repaired

    def repair_legacy_media_paths(self) -> int:
        """Reconnect media files to episodes written by the broken packaged drainer.

        The append-only raw ledger still identifies those attachments, and the checked-in demo
        fixture already contains their bytes. Only files that resolve inside ``media/`` are
        restored.
        """
        self._check()
        from . import ledger

        raw_path = os.path.join(self._data_dir, "raw_inputs.jsonl")
        media_dir = os.path.join(self._data_dir, "media")
        if not os.path.isfile(raw_path) or not os.path.isdir(media_dir):
            return 0

        by_id: dict[str, list[str]] = {}
        by_capture: dict[tuple[str, str], list[str]] = {}
        try:
            with open(raw_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        except (OSError, json.JSONDecodeError):
            return 0

        for row in rows:
            if not isinstance(row, dict) or row.get("op"):
                continue
            recovered: list[str] = []
            references = list(row.get("media_paths") or row.get("attachments") or [])
            for reference in references:
                leaf = reference.get("file") if isinstance(reference, dict) else reference
                if not isinstance(leaf, str):
                    continue
                base = os.path.basename(leaf.replace("\\", "/"))
                candidates = [base]
                without_spool_prefix = re.sub(r"^att\d+_", "", base)
                if without_spool_prefix != base:
                    candidates.insert(0, without_spool_prefix)
                for candidate in candidates:
                    resolved = ledger.contained_leaf(media_dir, candidate)
                    if resolved and os.path.isfile(resolved):
                        relative = f"media/{os.path.basename(resolved)}"
                        if relative not in recovered:
                            recovered.append(relative)
                        break
            if not recovered:
                continue
            note_id = row.get("id")
            if isinstance(note_id, str) and note_id:
                by_id[f"ep_{note_id}"] = recovered
            created_at = ledger.normalize_iso(str(row.get("created_at") or ""))
            by_capture.setdefault((created_at, str(row.get("text") or "")), recovered)

        repaired = 0
        store = self._g.store
        for node in list(store.nodes.values()):
            if node.ntype != NodeType.EPISODE or node.media_paths:
                continue
            key = (ledger.normalize_iso(node.created_at or ""), node.raw_text or "")
            paths = by_id.get(node.id) or by_capture.get(key)
            if not paths:
                continue
            node.media_paths = list(paths)
            store.touch_node(node.id)
            repaired += 1
        if repaired:
            self._g.save()
        return repaired

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

    def episode_graph(self, episode_id: str, *, fact_index=None) -> dict:
        """The provenance subgraph for one ingest (PROTOCOL §3.6a), in the same wire shape
        as `graph_preview`: exactly the nodes and edges THIS episode created — the episode
        node, the entities it mentions, the direct episode→entity MENTIONS spokes, and the
        RELATED_TO fact edges this note asserted (edge `episode_id` == this note, believed).

        Differs from `graph_preview` (a topological one-hop neighbourhood, used for entity
        navigation) in three ways: it is scoped by PROVENANCE (relationships asserted by
        OTHER notes between this note's entities are excluded), it has no 22-node cap (one
        note bounds the graph), and a fact endpoint that is a pre-existing entity the note
        did not mention is still drawn — the note connected to it. Every drawn node still
        reports `external_connections` (its display neighbours not on screen) so clients
        keep the dashed continuation stubs to the rest of the graph.

        `fact_index` is an optional {episode_id: [(src, dst, predicate)]} map (see
        `_facts_by_episode`); pass it to share one edge scan across a whole list."""
        self._check()
        store = self._g.store
        root = store.get_node(episode_id)
        if root is None or not root.valid or root.ntype is not NodeType.EPISODE:
            raise NotFound(f"unknown episode graph root: {episode_id}")

        facts = (fact_index if fact_index is not None
                 else self._facts_by_episode()).get(episode_id, [])

        # Draw the episode, then the entities it mentions, then any (possibly pre-existing)
        # endpoints of the facts this note asserted that the mention star did not already
        # cover. Order fixes hop 0 to the episode; everything else is hop 1.
        drawn_ids: list[str] = [episode_id]
        seen_ids = {episode_id}
        for eid in self._episode_entity_ids(episode_id):
            if eid not in seen_ids:
                seen_ids.add(eid)
                drawn_ids.append(eid)
        for src, dst, _pred in facts:
            for eid in (src, dst):
                node = store.get_node(eid)
                if eid not in seen_ids and node is not None and node.valid:
                    seen_ids.add(eid)
                    drawn_ids.append(eid)
        drawn = set(drawn_ids)

        nodes = []
        for hop_pos, nid in enumerate(drawn_ids):
            n = store.get_node(nid)
            is_ep = n.ntype is NodeType.EPISODE
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
        for eid in self._episode_entity_ids(episode_id):
            key = (episode_id, eid, "MENTIONS", "")
            if eid in drawn and key not in seen:
                seen.add(key)
                edges.append({"src": episode_id, "dst": eid,
                              "etype": "MENTIONS", "label": ""})
        for src, dst, pred in facts:
            key = (src, dst, "RELATED_TO", pred)
            if src in drawn and dst in drawn and key not in seen:
                seen.add(key)
                edges.append({"src": src, "dst": dst,
                              "etype": "RELATED_TO", "label": pred})
        return {"nodes": nodes, "edges": edges}

    def _facts_by_episode(self) -> dict[str, list[tuple[str, str, str]]]:
        """Reverse index episode_id → [(src, dst, predicate)] over every believed
        RELATED_TO fact, in stored orientation. Mirrors `_entity_fact_edges`' belief rule
        (a RETRACTED fact was never true) but buckets by the edge's asserting `episode_id`
        instead of by endpoint, so `episode_graph` reads one bucket. One O(E) pass: the
        episodes list builds it once and shares it, keeping list enrichment ~O(E) rather
        than an O(N·deg(hub)) per-episode walk over a super-hub like SELF."""
        store = self._g.store
        index: dict[str, list[tuple[str, str, str]]] = {}
        for src, dst, data in store.all_edges():
            if (data.get("etype") != EdgeType.RELATED_TO.value
                    or not data.get("valid", True)
                    or data.get("belief") == Belief.RETRACTED.value):
                continue
            ep = data.get("episode_id")
            if not ep:
                continue
            rel = data.get("rel_tag") or ""
            rel_node = store.get_node(rel) if rel else None
            pred = rel_node.name if rel_node is not None else rel
            index.setdefault(ep, []).append((src, dst, pred))
        return index

    @staticmethod
    def _raw_node_label(n) -> str:
        """Display label for a raw store node: an episode shows its text (its `name` is a
        useless source_ref), everything else shows its canonical/surface name."""
        if n.ntype is NodeType.EPISODE:
            return " ".join((n.raw_text or n.description or n.name or "").split())[:80]
        return n.name or n.id

    def _all_neighbor_ids(self, node_id: str) -> set[str]:
        """Every distinct valid graph neighbour of a node across ALL edge types — the raw
        store degree the dev graph uses to size off-ingest continuation stubs."""
        store = self._g.store
        return {nbr for nbr, _d in store.neighbors(node_id, direction="both")}

    def _raw_node_wire(self, n, hop: int, drawn: set[str]) -> dict:
        """One raw wire node: the store node's real type in `kind`, its entity type (if any)
        in `category`, and the count of neighbours not drawn in `external_connections`."""
        return {
            "id": n.id, "name": self._raw_node_label(n), "kind": n.ntype.value,
            "category": n.entity_type.value if n.entity_type is not None else None,
            "hop": hop,
            "external_connections": len(self._all_neighbor_ids(n.id) - drawn),
        }

    def _raw_edge_label(self, data: dict) -> str:
        """A raw edge's label: its predicate name for a `RELATED_TO` fact (resolved from the
        rel_tag node), otherwise the raw edge type itself (`MENTIONED_IN`, `TAGGED_AS`, …)."""
        etype = data.get("etype") or EdgeType.RELATED_TO.value
        if etype == EdgeType.RELATED_TO.value:
            rel = data.get("rel_tag") or ""
            rel_node = self._g.store.get_node(rel) if rel else None
            return rel_node.name if rel_node is not None else (rel or etype)
        return etype

    def node_raw_graph(self, node_id: str, *, max_nodes: int | None = None) -> dict:
        """The raw one-hop neighbourhood of ANY store node — episode, entity, mention, tag,
        or relation — in the same wire shape as `episode_raw_graph`. Powers click-to-re-root
        in the dev graph: the clicked node becomes the centre (`hop` 0) and the nodes actually
        connected to it in the store (`hop` 1) are drawn, wired by their real edges (raw edge
        type, or the predicate for a fact). Neighbours are ranked by connectivity and capped
        at `max_nodes` (default 22); `external_connections` counts the rest so the client keeps
        the dashed continuation stubs. Any edge that exists between two drawn nodes rides along,
        so a shared fact or tag among the neighbours is visible too."""
        self._check()
        store = self._g.store
        root = store.get_node(node_id)
        if root is None or not root.valid:
            raise NotFound(f"unknown graph node: {node_id}")
        cap = max_nodes or self._PREVIEW_MAX_NODES
        neighbours = [nid for nid in self._all_neighbor_ids(node_id)
                      if (nb := store.get_node(nid)) is not None and nb.valid]
        ranked = sorted(neighbours, key=lambda i: (-len(self._all_neighbor_ids(i)), i))
        drawn_ids = [node_id] + ranked[:cap - 1]
        drawn = set(drawn_ids)
        nodes = [self._raw_node_wire(store.get_node(nid), 0 if i == 0 else 1, drawn)
                 for i, nid in enumerate(drawn_ids)]
        edges, seen = [], set()
        for nid in drawn_ids:                       # every stored edge appears once, as an
            for nbr, data in store.neighbors(nid, direction="out"):   # out-edge of its src
                if nbr not in drawn:
                    continue
                etype = data.get("etype") or EdgeType.RELATED_TO.value
                label = self._raw_edge_label(data)
                key = (nid, nbr, etype, label, data.get("rel_tag") or "",
                       data.get("valid_at", ""), data.get("seq", 0))
                if key in seen:
                    continue
                seen.add(key)
                edges.append({"src": nid, "dst": nbr, "etype": etype, "label": label})
        return {"nodes": nodes, "edges": edges}

    def episode_raw_graph(self, episode_id: str, *, fact_index=None) -> dict:
        """The RAW store subgraph for one ingest, in the same wire shape as `graph_preview`
        but UNCOOKED — the nodes and edges the engine actually holds for this note, for the
        dev graph. Where `episode_graph` collapses each mention star into one episode→entity
        `MENTIONS` spoke and folds entity types onto person/place/thing glyphs, this exposes
        the real structure: the episode, its per-occurrence `MENTION` nodes, the `ENTITY`
        anchors they `RESOLVES_TO`, the `TAG` nodes it is `TAGGED_AS`, and its fact-partner
        entities — wired by their real edges (`MENTIONED_IN`, `RESOLVES_TO`, `TAGGED_AS`,
        `RELATED_TO`) with the raw edge type (or, for a fact, its predicate) in `label`.

        Still provenance-scoped to this note (only its own mention stars, tags, and facts —
        an edge from another note is not drawn), but a shared endpoint node it points at is,
        with `external_connections` counting that node's neighbours outside this ingest so
        the client still draws dashed continuation stubs. `node.kind` is the raw NodeType
        (episode|mention|entity|tag|relation|…); `node.category` is the entity_type for an
        entity, else null. `hop` is 0 for the episode and 1 for every other node (the graph
        is genuinely multi-hop; the renderer only reads hop to pick the centre)."""
        self._check()
        store = self._g.store
        root = store.get_node(episode_id)
        if root is None or not root.valid or root.ntype is not NodeType.EPISODE:
            raise NotFound(f"unknown episode graph root: {episode_id}")

        drawn_ids: list[str] = [episode_id]
        seen = {episode_id}
        raw_edges: list[tuple[str, str, str, str]] = []   # (src, dst, etype, label)

        def draw(nid: str) -> bool:
            if nid in seen:
                return True
            node = store.get_node(nid)
            if node is None or not node.valid:
                return False
            seen.add(nid)
            drawn_ids.append(nid)
            return True

        # Mention stars: mention → episode (MENTIONED_IN) and mention → entity (RESOLVES_TO).
        for mid, _d in store.neighbors(episode_id, etypes={EdgeType.MENTIONED_IN},
                                       direction="in"):
            if not draw(mid):
                continue
            raw_edges.append((mid, episode_id, "MENTIONED_IN", "MENTIONED_IN"))
            for eid, _d2 in store.neighbors(mid, etypes={EdgeType.RESOLVES_TO},
                                            direction="out"):
                if draw(eid):
                    raw_edges.append((mid, eid, "RESOLVES_TO", "RESOLVES_TO"))
        # Topical tags: episode → tag (TAGGED_AS).
        for tid, _d in store.neighbors(episode_id, etypes={EdgeType.TAGGED_AS},
                                       direction="out"):
            if draw(tid):
                raw_edges.append((episode_id, tid, "TAGGED_AS", "TAGGED_AS"))
        # This note's facts: entity → entity (RELATED_TO), predicate name in the label.
        facts = (fact_index if fact_index is not None
                 else self._facts_by_episode()).get(episode_id, [])
        for src, dst, pred in facts:
            if draw(src) and draw(dst):
                raw_edges.append((src, dst, "RELATED_TO", pred))

        drawn = set(drawn_ids)
        nodes = [self._raw_node_wire(store.get_node(nid), 0 if i == 0 else 1, drawn)
                 for i, nid in enumerate(drawn_ids)]
        edges = [{"src": s, "dst": d, "etype": et, "label": lb}
                 for (s, d, et, lb) in raw_edges]
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
        # One provenance-fact scan shared across every row's episode_graph, so a super-hub
        # (SELF) is not re-walked per episode — the O(N·deg) trap the daemon's shared
        # neighbor cache also guards against.
        fact_index = self._facts_by_episode()
        rows = []
        for n in eps[offset:offset + limit]:
            entities, categories, concepts = self._episode_entities(n.id)
            try:
                gp = self.episode_raw_graph(n.id, fact_index=fact_index)
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
