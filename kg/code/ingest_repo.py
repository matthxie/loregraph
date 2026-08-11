"""Codebase ingestion orchestration — a git repo → one unified, me-anchored graph.

REF-ANCHORED: everything is read out of an explicit git ref's tree (`ref`, e.g. "main") via
plumbing, so the checked-out branch is irrelevant — switching to an old or divergent branch
can't regress the graph.

Full ingest (no `base`): the last N commits reachable from `ref` (or --since) become
immutable commit Episodes, the repo becomes one summary Episode, and every source file in
`ref`'s tree becomes an embed-only file Episode (the thin state layer). Incremental
(`base` given): only `base..ref` commits are appended (idempotent by SHA) and only that
diff's changed files are re-embedded. `base` is trusted only when it is an ANCESTOR of
`ref` — after a rebase / force-push / branch swap the range is meaningless, so the run
falls back to a full ingest (cheap: content-hash dedup means unchanged files don't
re-extract). Commits are immutable and never superseded.

File Episodes are keyed by (ref, path) — `source_ref` is `file:<repo>@<ref>/<path>` — so
every pass ends by RECONCILING that ref: any file Episode whose path is no longer in the
ref's tree (deleted, renamed away, dropped by a force-push) is TOMBSTONED, per-ref, so
retrieval only ever sees code that currently exists. Multiple refs stay independent.
Commit Episodes stay ref-free (`commit:<repo>@<sha>`): a commit is the same commit on every
branch, and SHA keying is what makes re-sync idempotent.

Structure edges: NEXT chains the salient commits chronologically (the work timeline), and
MODIFIES joins each commit to the current file Episodes it touched (the event↔state join,
"which commits last touched this file") at a low PPR weight.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..chunkers import chunk_code
from ..corpus import CorpusItem
from ..models import Edge, EdgeType, Modality, NodeType, Provenance
from ..store import now_iso
from . import git
from .walk import gather_repo_signals

_NONWORD = re.compile(r"[^0-9A-Za-z]+")


@dataclass
class CodeIngestReport:
    repo: str = ""
    ref: str = ""                   # the ref ingested (branch name, or "HEAD")
    head_sha: str | None = None     # the SHA `ref` resolved to this run
    base_sha: str | None = None     # the base the range was taken from (None on a full pass)
    full: bool = True               # False when this was an incremental base..ref pass
    summarized: bool = False        # repo-summary Episode written this run
    commits_seen: int = 0           # commits in the log window
    commits_ingested: int = 0       # salient commits that became Episodes
    files_ingested: int = 0         # file Episodes written (chunks)
    files_superseded: int = 0       # prior file Episodes marked invalid
    files_removed: int = 0          # file Episodes tombstoned (path gone from ref's tree)
    next_edges: int = 0
    modifies_edges: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"repo={self.repo}@{self.ref} head={self.head_sha and self.head_sha[:12]} "
                f"mode={'full' if self.full else 'incremental'} "
                f"commits={self.commits_ingested}/{self.commits_seen} "
                f"files=+{self.files_ingested}/-{self.files_superseded} "
                f"removed={self.files_removed} "
                f"summary={'yes' if self.summarized else 'no'} "
                f"next={self.next_edges} modifies={self.modifies_edges}")


def _pathkey(path: str) -> str:
    return _NONWORD.sub("_", path).strip("_")


def _file_source_ref(name: str, ref: str, path: str) -> str:
    return f"file:{name}@{ref}/{path}"


def _file_episodes(store, name: str, ref: str, path: str) -> list:
    """Valid CODE file Episodes for one (ref, path) — all chunks of its newest content."""
    sref = _file_source_ref(name, ref, path)
    return [n for n in store.nodes.values()
            if n.ntype == NodeType.EPISODE and n.valid
            and n.modality == Modality.CODE and n.source_ref == sref]


def _supersede(store, nodes: list, new_id: str | None) -> int:
    for n in nodes:
        n.valid = False
        n.superseded_by = new_id
        store.touch_node(n.id)
    return len(nodes)


def _file_items(name: str, ref: str, path: str, content: str, ts: str, cfg) -> list[CorpusItem]:
    """Chunk one source file into embed-only file Episodes. The ref and a file-level content
    version (`ver`) are baked into the ids so a changed file mints NEW nodes (which coexist
    with the superseded old ones) and two refs never collide; an unchanged file re-mints the
    SAME ids and dedups away."""
    from ..ingest import _sha256
    chunks = chunk_code(content, target=int(cfg.chunk_target_chars),
                        max_chars=int(cfg.chunk_max_chars))
    # (text, line span) — an unchunked file is the whole file, so its span is every line.
    pieces = ([(c.text, [c.start_line, c.end_line]) for c in chunks] if chunks
              else [(content, [1, len(content.split("\n"))])])
    ver = _sha256("file", content)[:8]
    sref = _file_source_ref(name, ref, path)
    rk, pk = _pathkey(ref), _pathkey(path)
    items: list[CorpusItem] = []
    for ordinal, (text, span) in enumerate(pieces):
        items.append(CorpusItem(
            id=f"file_{name}_{rk}_{pk}_{ver}#c{ordinal:03d}", modality="code",
            source_ref=sref, title=path, text=text, created_at=ts, embed_only=True,
            meta={"line_span": span}))
    return items


def _reconcile_ref(store, name: str, ref: str, tree_paths: set[str]) -> int:
    """Phase 2b: tombstone every live file Episode of THIS ref whose path is no longer in the
    ref's tree — deleted files, the old side of a rename, and whatever a rebase / force-push
    dropped. Scoped by the `file:<name>@<ref>/` prefix, so another ref's copy of the same path
    is untouched, and commit Episodes (ref-free) are never candidates. Tombstoning (kg/forget)
    rather than supersession: the content is gone, not updated, so it must leave retrieval
    entirely instead of surviving as history."""
    from ..forget import forget
    prefix = f"file:{name}@{ref}/"
    stale = [n.id for n in store.nodes.values()
             if n.ntype == NodeType.EPISODE and n.valid and n.modality == Modality.CODE
             and (n.source_ref or "").startswith(prefix)
             and (n.source_ref or "")[len(prefix):] not in tree_paths]
    if not stale:
        return 0
    return len(forget(store, episode_ids=stale).episodes)


def _plan_files(store, g, repo_path: str, name: str, ref: str, ref_sha: str,
                paths: list[str], head_ts: str) -> tuple[list[CorpusItem], int]:
    """Build file Episodes for the given paths as of `ref_sha` and supersede the prior version
    of each (any valid Episode for that (ref, path) whose id isn't in the new set — so an
    unchanged re-ingest supersedes nothing). Returns (new items, count superseded)."""
    items: list[CorpusItem] = []
    superseded = 0
    for path in paths:
        content = git.read_file(repo_path, path, ref_sha)
        if content is None or not content.strip():
            continue
        new_items = _file_items(name, ref, path, content, head_ts, g.config)
        new_ids = {f"ep_{it.id}" for it in new_items}
        stale = [n for n in _file_episodes(store, name, ref, path) if n.id not in new_ids]
        superseded += _supersede(store, stale, next(iter(new_ids), None))
        items += new_items
    return items, superseded


def ingest_repo(g, repo_path: str, *, ref: str = "HEAD", base: str | None = None,
                since: str | None = None, max_commits: int = 200) -> CodeIngestReport:
    """Ingest a git repo's history + summary + file state AS OF `ref` into the graph `g`
    (a KnowledgeGraph), reading the ref's tree rather than the working directory.

    `base` (the caller's last-synced commit) → incremental `base..ref`, but only when base is
    an ANCESTOR of ref; otherwise (rebase / force-push / branch swap / stale marker) this
    silently does a FULL ingest of `ref` instead. The caller can therefore always pass its
    marker and let this decide."""
    if not git.is_repo(repo_path):
        raise git.GitError(f"{repo_path} is not a git work tree")
    ref = ref or "HEAD"
    ref_sha = git.resolve_ref(repo_path, ref)
    if ref_sha is None:
        raise git.GitError(f"ref {ref!r} does not resolve in {repo_path}")
    label = git.ref_label(repo_path, ref)
    name = git.repo_name(repo_path)
    store = g.store

    # `base` is only a usable range endpoint when it's reachable from ref; else fall back full.
    incremental = bool(base) and git.is_ancestor(repo_path, base, ref_sha)
    report = CodeIngestReport(repo=name, ref=label, head_sha=ref_sha,
                              base_sha=base if incremental else None, full=not incremental)
    if base and not incremental:
        report.notes.append(f"base {base[:12]} is not an ancestor of {label}; full ingest")
    head_ts = None
    items: list[CorpusItem] = []

    # 1. repo summary (first-ever ingest of this repo only)
    repo_ep_id = f"ep_repo_{name}"
    if not store.has_node(repo_ep_id):
        signals = gather_repo_signals(repo_path, ref_sha)
        items.append(CorpusItem(id=f"repo_{name}", modality="code",
                                source_ref=f"repo:{name}", title=name, text=None,
                                created_at=now_iso(), meta={"signals": signals}))
        report.summarized = True

    # 2. commits → salient Episodes (event layer)
    commits = git.get_commits(repo_path, ref=ref_sha, base=base if incremental else None,
                              since=since, max_count=max_commits)
    report.commits_seen = len(commits)
    if commits:
        head_ts = commits[-1].committed_at
    salient_eps: list[str] = []          # ep ids in chronological order (for NEXT)
    commit_touch: list[tuple[str, list[str]]] = []   # (ep_id, [source paths]) for MODIFIES
    diff_cap = int(g.config.extract_max_chars)
    for c in commits:
        sc = git.salient_commit(repo_path, c, diff_cap=diff_cap)
        if sc is None:
            continue
        ep_id = f"ep_{name}_{c.sha12}"
        items.append(CorpusItem(
            id=f"{name}_{c.sha12}", modality="code",
            source_ref=f"commit:{name}@{c.sha}", title=c.subject or c.sha12,
            text=None, created_at=c.committed_at,
            meta={"message": c.message, "diff": sc.diff}))
        salient_eps.append(ep_id)
        commit_touch.append((ep_id, sorted({ch.path for ch in sc.changes})))
    report.commits_ingested = len(salient_eps)

    # 3. file state layer, as of ref
    if head_ts is None:
        head_ts = now_iso()
    tree_paths = set(git.list_source_files(repo_path, ref_sha))
    if incremental:                      # only the base..ref changed set
        changed, _deleted = git.changed_files(repo_path, base, ref_sha)
        paths = sorted(changed & tree_paths)
    else:                                # full: every source file in ref's tree
        paths = sorted(tree_paths)
    file_items, sup = _plan_files(store, g, repo_path, name, label, ref_sha, paths, head_ts)
    report.files_superseded += sup
    items += file_items

    # 4. one ingest pass (extraction fans out; embed-only files skip the LLM)
    if items:
        ing_report = g.ingest(items)
        report.files_ingested = sum(1 for it in file_items
                                    if store.has_node(f"ep_{it.id}"))
        if ing_report.notes:
            report.notes.extend(ing_report.notes[:3])

    # 5. NEXT edges — chain the salient commits chronologically (the work timeline)
    nw = float(getattr(g.config, "next_weight", 0.5))
    for prev, cur in zip(salient_eps, salient_eps[1:]):
        if store.has_node(prev) and store.has_node(cur):
            store.add_edge(Edge(src=prev, dst=cur, etype=EdgeType.NEXT,
                                provenance=Provenance.DERIVED, confidence=1.0, weight=nw))
            report.next_edges += 1

    # 6. MODIFIES edges — commit → current file Episodes it touched (event↔state join)
    mw = float(getattr(g.config, "modifies_weight", 0.2))
    for ep_id, paths in commit_touch:
        if not store.has_node(ep_id):
            continue
        for path in paths:
            for fep in _file_episodes(store, name, label, path):
                store.add_edge(Edge(src=ep_id, dst=fep.id, etype=EdgeType.MODIFIES,
                                    provenance=Provenance.DERIVED, confidence=1.0, weight=mw))
                report.modifies_edges += 1

    # 7. reconcile (Phase 2b) — tombstone this ref's file Episodes whose path is gone from
    # its tree. Runs on BOTH passes: a full-tree check covers deletes, renames and the
    # rebase/force-push fallback uniformly, so no diff can be missed.
    report.files_removed = _reconcile_ref(store, name, label, tree_paths)

    g.save()
    return report

# TODO (Phase 3): fs.watch of the working tree, opt-in git hooks, an MCP kg_sync_repo tool,
# and multiple refs / worktrees per repo. All additive — everything here is keyed by
# (ref, path), so a second tracked ref reconciles independently of the first.
