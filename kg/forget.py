"""Erasure — make targeted memories unretrievable (forget ≠ supersede).

Supersession (kg/temporal.py) ENDS a fact but keeps it queryable as history — that is
updating, not forgetting. This module implements ERASURE for deletion requests
("forget my address"), at two levels:

`forget(store, episode_ids=…, match=…)` — the id-level primitive: tombstone whole
episode chunks, retract the facts extracted from them, invalidate whatever the removal
orphans. Every retrieval surface already filters on validity (projection, seeder,
`fact_active`), so tombstoned content is unreachable immediately, with zero recompute.

`erase(…, secret)` — the production entry point (exposed as `KnowledgeGraph.forget`):
query-and-trace-back. Sweeps EVERY chunk for the secret (exhaustive lexical scan +
dense cosine against the episode index — deletion must not be top-k), confirms each
candidate (deterministic phrase/coverage gate; optional LLM judge for paraphrased
restatements), REDACTS the matched sentences in place (the rest of the turn survives:
text rewritten with a [redacted] marker, chunk re-embedded locally, BM25 invalidated),
retracts the derived artifacts that came from the removed text (surface alignment;
optional single-chunk re-extract diff for paraphrased extractions), cascades orphan
invalidation, and loops until a re-sweep finds nothing. An optional final inference
audit asks an LLM to reconstruct the secret from what retrieval still returns and
escalates the contributing chunks to full tombstones if it can.

Redaction never rewrites text with an LLM — sentences are removed, never paraphrased,
so every stored sentence remains something that was actually said.

Scope honesty: this erases what the STORE can reach. Copies outside the store — ingest
caches, raw session logs — must be purged by the operator; re-ingestion resurrects
anything left there. (The content-hash dedup cache is deliberately left intact: it maps
the ORIGINAL content hash to the now-redacted episode, so accidentally re-ingesting the
same session does not resurrect the secret.)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .canonicalize import Canonicalizer
from .config import Config
from .embedders import Embedder
from .models import Belief, EdgeType, NodeType
from .store import GraphStore

REDACTED = "[redacted]"
_WORDS = re.compile(r"[a-z0-9']+")
# sentence boundaries: terminal punctuation, or the newlines chat turns are built from
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "for", "with", "my",
         "your", "his", "her", "their", "our", "i", "you", "he", "she", "it", "we",
         "they", "is", "are", "was", "were", "at", "by", "as", "that", "this"}


def _tokens(text: str) -> set[str]:
    return {t for t in _WORDS.findall((text or "").lower()) if t not in _STOP}


def _sentences(text: str) -> list[str]:
    return [s for s in (p.strip() for p in _SENT_SPLIT.split(text or "")) if s]


# --------------------------------------------------------------------------- #
# id-level primitive: tombstone whole chunks
# --------------------------------------------------------------------------- #
@dataclass
class ForgetReport:
    episodes: list[str] = field(default_factory=list)      # chunks tombstoned
    facts: list[tuple] = field(default_factory=list)        # (src, dst, gkey) retracted
    mentions: list[str] = field(default_factory=list)       # orphaned mention nodes
    entities: list[str] = field(default_factory=list)       # orphaned entity anchors
    tags: list[str] = field(default_factory=list)           # orphaned tags

    def total(self) -> int:
        return (len(self.episodes) + len(self.facts) + len(self.mentions)
                + len(self.entities) + len(self.tags))


def _expand_targets(store: GraphStore, episode_ids, match) -> set[str]:
    """Resolve the requested targets to concrete valid EPISODE chunk ids. A bare source
    id expands to all of its chunks; `match` sweeps episode raw text (case-insensitive
    literal) so directly-restated copies are caught in the same call."""
    targets: set[str] = set()
    wanted = set(episode_ids or [])
    needle = (match or "").lower()
    for nid, n in store.nodes.items():
        if n.ntype != NodeType.EPISODE or not n.valid:
            continue
        base = nid.split("#c")[0]
        if nid in wanted or base in wanted:
            targets.add(nid)
        elif needle and needle in (n.raw_text or "").lower():
            targets.add(nid)
    return targets


def _retract_edge(store: GraphStore, u: str, v: str, gkey: str, data: dict) -> bool:
    """Invalidate one edge (and retract it if it is a FACT, so `fact_active` rejects it
    in every temporal view). Returns True if anything changed."""
    changed = False
    if data.get("valid", True):
        data["valid"] = False
        changed = True
    if (data.get("etype") == EdgeType.RELATED_TO.value
            and data.get("belief") == Belief.ASSERTED.value):
        data["belief"] = Belief.RETRACTED.value
        changed = True
    if changed:
        store.touch_edge(u, v, gkey)
    return changed


def _cascade_orphans(store: GraphStore, report: ForgetReport) -> None:
    """Invalidate nodes whose every remaining incident edge is invalid. Mentions first —
    their invalidation is what orphans entities and tags."""
    def _orphaned(nid: str) -> bool:
        for _u, _v, _k, d in store.g.edges(nid, keys=True, data=True):
            if d.get("valid", True):
                return False
        for _u, _v, _k, d in store.g.in_edges(nid, keys=True, data=True):
            if d.get("valid", True):
                return False
        return True

    for layer, bucket in ((NodeType.MENTION, report.mentions),
                          (NodeType.ENTITY, report.entities),
                          (NodeType.TAG, report.tags)):
        for nid in sorted(store.nodes):
            n = store.nodes[nid]
            if n.ntype == layer and n.valid and _orphaned(nid):
                n.valid = False
                store.touch_node(nid)
                bucket.append(nid)


def forget(store: GraphStore, *, episode_ids: list[str] | None = None,
           match: str | None = None) -> ForgetReport:
    """Tombstone episode chunks (and everything only they support) so no retrieval
    surface can reach them. Idempotent; returns a report of what was invalidated."""
    report = ForgetReport()
    targets = _expand_targets(store, episode_ids, match)
    if not targets:
        return report

    for eid in sorted(targets):
        node = store.nodes[eid]
        node.valid = False
        store.touch_node(eid)
        report.episodes.append(eid)

    # every edge incident to a target, and every FACT extracted from a target
    # (RELATED_TO edges live between entities but carry episode_id provenance)
    for u, v, gkey, data in list(store.g.edges(keys=True, data=True)):
        incident = u in targets or v in targets
        from_target = data.get("episode_id", "") in targets
        if (incident or from_target) and _retract_edge(store, u, v, gkey, data):
            if data.get("etype") == EdgeType.RELATED_TO.value:
                report.facts.append((u, v, gkey))

    _cascade_orphans(store, report)
    return report


# --------------------------------------------------------------------------- #
# production entry point: semantic sweep → confirm → redact → trace-back → loop
# --------------------------------------------------------------------------- #
@dataclass
class EraseAction:
    episode_id: str
    kind: str                      # "redact" | "tombstone"
    reason: str                    # "exact" | "coverage" | "judge" | "audit"
    removed_sentences: list[str] = field(default_factory=list)
    artifacts_dropped: list[str] = field(default_factory=list)   # human-readable
    artifacts_kept: int = 0


@dataclass
class EraseReport:
    secret: str
    dry_run: bool
    actions: list[EraseAction] = field(default_factory=list)
    unconfirmed: list[str] = field(default_factory=list)   # fuzzy hits left alone
    orphans: ForgetReport = field(default_factory=ForgetReport)
    iterations: int = 0
    audit: str = ""                # "" (not run) | "clean" | "leaked -> escalated"
    llm_calls: int = 0

    def summary(self) -> str:
        red = sum(1 for a in self.actions if a.kind == "redact")
        tomb = sum(1 for a in self.actions if a.kind == "tombstone")
        return (f"erase({self.secret!r}{', DRY RUN' if self.dry_run else ''}): "
                f"{red} chunk(s) redacted, {tomb} tombstoned, "
                f"{sum(len(a.artifacts_dropped) for a in self.actions)} artifact(s) "
                f"retracted, {self.orphans.total()} orphan(s) invalidated, "
                f"{len(self.unconfirmed)} candidate(s) left unconfirmed, "
                f"audit={self.audit or 'off'}, llm_calls={self.llm_calls}")


class Eraser:
    """One erase request. Deterministic gates run first; the LLM (when provided) is an
    escalation for paraphrase confirmation, artifact attribution, and the final audit."""

    SEMANTIC_FLOOR = 0.45          # cosine below which a chunk is not even a candidate
    COVERAGE_STRONG = 0.8          # secret-token coverage in one sentence = confirmed
    COVERAGE_SPAN = 0.5            # sentence joins the redaction span at this coverage
    MAX_ITER = 5
    AUDIT_LEAK = 0.6               # token coverage of the secret in the audit's guess

    def __init__(self, store: GraphStore, embedder: Embedder, canon: Canonicalizer,
                 config: Config, *, extractor=None, client=None):
        self.store = store
        self.embedder = embedder
        self.canon = canon
        self.config = config
        self.extractor = extractor
        self.client = client

    # ------------------------------------------------------------------ sweep
    def _sweep(self, secret: str, done: set[str]) -> tuple[dict, list[str]]:
        """Exhaustive scan of every valid chunk (never top-k: deletion needs recall).
        Returns ({chunk_id: (reason, [sentence_idx])}, [unconfirmed fuzzy ids])."""
        toks = _tokens(secret)
        needle = secret.strip().lower()
        qv = self.embedder.embed([secret])[0]
        confirmed: dict = {}
        fuzzy: list[str] = []
        for eid in sorted(self.store.nodes):
            n = self.store.nodes[eid]
            if n.ntype != NodeType.EPISODE or not n.valid or eid in done:
                continue
            text = n.raw_text or ""
            sents = _sentences(text)
            if not sents:
                continue
            exact_idx = [i for i, s in enumerate(sents) if needle and needle in s.lower()]
            if exact_idx:
                confirmed[eid] = ("exact", exact_idx)
                continue
            if toks:
                cov = [len(_tokens(s) & toks) / len(toks) for s in sents]
                if max(cov) >= self.COVERAGE_STRONG and len(toks) >= 2:
                    idx = [i for i, c in enumerate(cov) if c >= self.COVERAGE_SPAN]
                    confirmed[eid] = ("coverage", idx)
                    continue
            vec = self.store.vectors.get("episode", eid)
            if vec is not None and float(vec @ qv) >= self.SEMANTIC_FLOOR:
                fuzzy.append(eid)
        return confirmed, fuzzy

    # ------------------------------------------------------------- escalation
    def _chat(self, prompt: str, max_tokens: int = 300) -> str:
        from .llm_client import resolve_model
        resp = self.client.chat.completions.create(
            model=resolve_model(self.config.llm_model), max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}])
        return (resp.choices[0].message.content or "").strip()

    def _judge_fuzzy(self, secret: str, eid: str) -> list[int] | None:
        """LLM confirmation for a semantic-only candidate: which sentences (if any)
        state or restate the secret? None = does not contain it."""
        sents = _sentences(self.store.nodes[eid].raw_text or "")
        numbered = "\n".join(f"{i}: {s}" for i, s in enumerate(sents))
        raw = self._chat(
            "You are auditing text for an erasure request. Information to erase: "
            f"{secret!r}\n\nDoes any numbered sentence below state, restate, or "
            "paraphrase that information? Judge only the sentences shown; topical "
            "similarity is NOT a match.\n\n"
            f"{numbered}\n\n"
            'Reply with JSON only: {"contains": true|false, "sentences": [indexes]}')
        try:
            payload = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
            if payload.get("contains") and payload.get("sentences"):
                return [i for i in payload["sentences"]
                        if isinstance(i, int) and 0 <= i < len(sents)]
        except (ValueError, KeyError, TypeError):
            pass
        return None

    # -------------------------------------------------------------- redaction
    def _chunk_artifacts(self, eid: str):
        """(kind, surface, handle) for every artifact derived from this chunk:
        mentions resolved from it, facts extracted from it, tags attached to it."""
        out = []
        for m, _v, k, d in self.store.g.in_edges(eid, keys=True, data=True):
            if d.get("etype") == EdgeType.MENTIONED_IN.value and d.get("valid", True):
                node = self.store.get_node(m)
                if node is not None and node.valid:
                    out.append(("mention", node.name or "", m))
        for _u, tag, k, d in self.store.g.edges(eid, keys=True, data=True):
            if d.get("etype") == EdgeType.TAGGED_AS.value and d.get("valid", True):
                node = self.store.get_node(tag)
                out.append(("tag", (node.name if node else "") or "", (eid, tag, k)))
        for u, v, k, d in self.store.g.edges(keys=True, data=True):
            if (d.get("etype") == EdgeType.RELATED_TO.value
                    and d.get("episode_id") == eid and d.get("valid", True)):
                su = (self.store.get_node(u) or type("x", (), {"name": u})).name or u
                sv = (self.store.get_node(v) or type("x", (), {"name": v})).name or v
                surface = f"{su} {d.get('rel_tag', '')} {sv}"
                out.append(("fact", surface, (u, v, k)))
        return out

    def _reextract_surfaces(self, text: str) -> set[str] | None:
        """Surfaces the extractor still finds in the REDACTED text (entity names, tags,
        relation endpoints/labels) — anything absent no longer has support."""
        if self.extractor is None:
            return None
        try:
            ext = self.extractor.extract_text(text)
        except Exception:  # noqa: BLE001 — extraction failure must not block erasure
            return None
        surfaces: set[str] = set()
        for e in ext.entities:
            surfaces.update(_tokens(e.name))
        for t in ext.tags:
            surfaces.update(_tokens(t))
        for r in ext.relations:
            surfaces.update(_tokens(r.source) | _tokens(r.target))
            for lb in r.labels:
                surfaces.update(_tokens(lb))
        return surfaces

    def _apply(self, eid: str, reason: str, idx: list[int], report: EraseReport,
               dry_run: bool, allow_reextract: bool) -> EraseAction:
        node = self.store.nodes[eid]
        sents = _sentences(node.raw_text or "")
        idx_set = set(idx)
        removed = [sents[i] for i in sorted(idx_set)]
        kept = [s for i, s in enumerate(sents) if i not in idx_set]

        # whole turn is the secret → tombstone the chunk
        if not kept or not _tokens(" ".join(kept)):
            action = EraseAction(episode_id=eid, kind="tombstone", reason=reason,
                                 removed_sentences=removed)
            if not dry_run:
                forget_report = forget(self.store, episode_ids=[eid])
                action.artifacts_dropped = [f"fact:{u}->{v}" for u, v, _ in
                                            forget_report.facts]
            return action

        action = EraseAction(episode_id=eid, kind="redact", reason=reason,
                             removed_sentences=removed)
        removed_text = " ".join(removed)
        kept_text = " ".join(kept)

        # artifact attribution: kept text wins; removed-only drops; unaligned escalates
        # to a single-chunk re-extract diff (or drops conservatively without one)
        reex: set[str] | None = None
        for kind, surface, handle in self._chunk_artifacts(eid):
            s_toks = _tokens(surface)
            if not s_toks:
                continue
            in_kept = s_toks <= _tokens(kept_text)
            in_removed = bool(s_toks & _tokens(removed_text))
            drop = False
            if in_kept:
                pass                                    # provable support survives
            elif in_removed:
                drop = True
            else:                                       # paraphrased extraction
                if allow_reextract and reex is None and not dry_run:
                    reex = self._reextract_surfaces(kept_text)
                    if reex is not None:
                        report.llm_calls += 1
                drop = not (reex is not None and s_toks <= reex)
            if not drop:
                action.artifacts_kept += 1
                continue
            action.artifacts_dropped.append(f"{kind}:{surface}")
            if dry_run:
                continue
            if kind == "mention":
                self.store.nodes[handle].valid = False
                self.store.touch_node(handle)
            elif kind == "tag":
                u, v, k = handle
                d = (self.store.g.get_edge_data(u, v) or {}).get(k)
                if d is not None:
                    _retract_edge(self.store, u, v, k, d)
            elif kind == "fact":
                u, v, k = handle
                d = (self.store.g.get_edge_data(u, v) or {}).get(k)
                if d is not None:
                    _retract_edge(self.store, u, v, k, d)

        if not dry_run:
            new_text = "\n".join((s if i not in idx_set else REDACTED)
                                 for i, s in enumerate(sents))
            node.raw_text = new_text
            self.store.touch_node(eid)
            self.store.vectors.add("episode", eid, self.embedder.embed([new_text])[0])
            self.store.episode_version += 1            # invalidate the lazy BM25 corpus
        return action

    # ------------------------------------------------------------------ audit
    def _audit(self, secret: str, report: EraseReport) -> None:
        """Adversarial reconstruction: retrieve for the secret, ask the model to state
        it from the surviving context. A successful guess escalates the contributing
        chunks to full tombstones."""
        from .rag import ContextBuilder
        from .retrieval import HybridRetriever
        retr = HybridRetriever(self.store, self.embedder, self.canon, self.config)
        res = retr.retrieve(secret, k=self.config.top_k)
        ctx_ids, _facts, blob = ContextBuilder(self.store, self.config).build(res)
        if not ctx_ids:
            report.audit = "clean"
            return
        guess = self._chat(
            "Some information was erased from the records below. From ONLY these "
            f"records, state the most likely value of: {secret!r}. If it cannot be "
            "determined, reply exactly UNKNOWN.\n\n" + blob[:20000], max_tokens=120)
        report.llm_calls += 1
        toks = _tokens(secret)
        if toks and len(_tokens(guess) & toks) / len(toks) >= self.AUDIT_LEAK:
            forget(self.store, episode_ids=sorted({c.split("#c")[0] for c in ctx_ids}))
            report.audit = "leaked -> escalated"
        else:
            report.audit = "clean"

    # -------------------------------------------------------------------- run
    def erase(self, secret: str, *, dry_run: bool = False,
              escalate: bool = True) -> EraseReport:
        report = EraseReport(secret=secret, dry_run=dry_run)
        use_llm = escalate and self.client is not None
        done: set[str] = set()
        for _ in range(self.MAX_ITER):
            report.iterations += 1
            confirmed, fuzzy = self._sweep(secret, done)
            if use_llm:
                for eid in fuzzy:
                    idx = self._judge_fuzzy(secret, eid)
                    report.llm_calls += 1
                    if idx:
                        confirmed[eid] = ("judge", idx)
                    else:
                        done.add(eid)                   # judged clean; don't re-ask
            else:
                report.unconfirmed.extend(e for e in fuzzy if e not in done)
                done.update(fuzzy)
            if not confirmed:
                break
            for eid, (reason, idx) in sorted(confirmed.items()):
                report.actions.append(self._apply(eid, reason, idx, report, dry_run,
                                                  allow_reextract=use_llm))
                done.add(eid)
            if dry_run:
                break                                   # nothing mutated; one pass only
        if not dry_run:
            _cascade_orphans(self.store, report.orphans)
            if use_llm:
                self._audit(secret, report)
        report.unconfirmed = sorted(set(report.unconfirmed))
        return report
