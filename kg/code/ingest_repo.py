"""Codebase ingestion orchestration — a git repo → one unified, me-anchored graph.

Bootstrap (no prior SHA): the last N commits (or --since) become immutable commit Episodes,
the repo becomes one summary Episode, and every current source file becomes an embed-only
file Episode (the thin state layer). Incremental re-sync (a stored last SHA): only
`last..HEAD` new commits are appended (idempotent by SHA), and only the diff's changed files
are re-embedded — the prior version of each is SUPERSEDED (valid=False + superseded_by) so
retrieval stays current without bloat; deleted files are superseded outright. Commits are
immutable, so they are never superseded.

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
    head_sha: str | None = None
    summarized: bool = False        # repo-summary Episode written this run
    commits_seen: int = 0           # commits in the log window
    commits_ingested: int = 0       # salient commits that became Episodes
    files_ingested: int = 0         # file Episodes written (chunks)
    files_superseded: int = 0       # prior file Episodes marked invalid
    next_edges: int = 0
    modifies_edges: int = 0
    notes: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"repo={self.repo} head={self.head_sha and self.head_sha[:12]} "
                f"commits={self.commits_ingested}/{self.commits_seen} "
                f"files=+{self.files_ingested}/-{self.files_superseded} "
                f"summary={'yes' if self.summarized else 'no'} "
                f"next={self.next_edges} modifies={self.modifies_edges}")


def _pathkey(path: str) -> str:
    return _NONWORD.sub("_", path).strip("_")


def _file_episodes(store, name: str, path: str) -> list:
    """Valid CODE file Episodes for one path (all chunks), newest-content set."""
    ref = f"file:{name}/{path}"
    return [n for n in store.nodes.values()
            if n.ntype == NodeType.EPISODE and n.valid
            and n.modality == Modality.CODE and n.source_ref == ref]


def _supersede(store, nodes: list, new_id: str | None) -> int:
    for n in nodes:
        n.valid = False
        n.superseded_by = new_id
        store.touch_node(n.id)
    return len(nodes)


def _file_items(name: str, path: str, content: str, ts: str, cfg) -> list[CorpusItem]:
    """Chunk one source file into embed-only file Episodes. A file-level content version
    (`ver`) is baked into the ids so a changed file mints NEW nodes (which coexist with the
    superseded old ones); an unchanged file re-mints the SAME ids and dedups away."""
    from ..ingest import _sha256
    chunks = chunk_code(content, target=int(cfg.chunk_target_chars),
                        max_chars=int(cfg.chunk_max_chars))
    texts = [c.text for c in chunks] if chunks else [content]
    ver = _sha256("file", content)[:8]
    ref = f"file:{name}/{path}"
    pk = _pathkey(path)
    items: list[CorpusItem] = []
    for ordinal, text in enumerate(texts):
        items.append(CorpusItem(
            id=f"file_{name}_{pk}_{ver}#c{ordinal:03d}", modality="code",
            source_ref=ref, title=path, text=text, created_at=ts, embed_only=True))
    return items


def _plan_files(store, g, repo_path: str, name: str, paths: list[str], head_ts: str) \
        -> tuple[list[CorpusItem], int]:
    """Build file Episodes for the given changed paths and supersede the prior version of
    each (any valid Episode for the path whose id isn't in the new set — so an unchanged
    re-ingest supersedes nothing). Returns (new items, count superseded)."""
    items: list[CorpusItem] = []
    superseded = 0
    for path in paths:
        content = git.read_file(repo_path, path)
        if content is None or not content.strip():
            continue
        new_items = _file_items(name, path, content, head_ts, g.config)
        new_ids = {f"ep_{it.id}" for it in new_items}
        stale = [n for n in _file_episodes(store, name, path) if n.id not in new_ids]
        superseded += _supersede(store, stale, next(iter(new_ids), None))
        items += new_items
    return items, superseded


def ingest_repo(g, repo_path: str, *, since: str | None = None,
                after_sha: str | None = None, max_commits: int = 200) -> CodeIngestReport:
    """Ingest a git repo's history + summary + current-file state into the graph `g`
    (a KnowledgeGraph). `after_sha` set → incremental re-sync of `after_sha..HEAD`; else a
    bootstrap of the last `max_commits` commits (or `since`)."""
    if not git.is_repo(repo_path):
        raise git.GitError(f"{repo_path} is not a git work tree")
    name = git.repo_name(repo_path)
    store = g.store
    report = CodeIngestReport(repo=name, head_sha=git.head_sha(repo_path))
    head_ts = None
    items: list[CorpusItem] = []

    # 1. repo summary (bootstrap / first-ever sync of this repo only)
    repo_ep_id = f"ep_repo_{name}"
    if not store.has_node(repo_ep_id):
        signals = gather_repo_signals(repo_path)
        items.append(CorpusItem(id=f"repo_{name}", modality="code",
                                source_ref=f"repo:{name}", title=name, text=None,
                                created_at=now_iso(), meta={"signals": signals}))
        report.summarized = True

    # 2. commits → salient Episodes (event layer)
    commits = git.get_commits(repo_path, since=since, after_sha=after_sha,
                              max_count=max_commits)
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

    # 3. current-file state layer
    if head_ts is None:
        head_ts = now_iso()
    if after_sha:                        # incremental: only the changed set
        changed, deleted = _changed_files(repo_path, after_sha)
        for path in deleted:
            report.files_superseded += _supersede(store, _file_episodes(store, name, path), None)
        file_items, sup = _plan_files(store, g, repo_path, name, sorted(changed), head_ts)
    else:                                # bootstrap: every current source file
        file_items, sup = _plan_files(store, g, repo_path, name,
                                      git.list_source_files(repo_path), head_ts)
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
            for fep in _file_episodes(store, name, path):
                store.add_edge(Edge(src=ep_id, dst=fep.id, etype=EdgeType.MODIFIES,
                                    provenance=Provenance.DERIVED, confidence=1.0, weight=mw))
                report.modifies_edges += 1

    g.save()
    return report


def _changed_files(repo: str, after_sha: str) -> tuple[set[str], set[str]]:
    """(changed_or_added, deleted) source-file paths over `after_sha..HEAD`, following
    renames (an R is a delete of old_path + a change at new_path)."""
    changed: set[str] = set()
    deleted: set[str] = set()
    raw = git._run(repo, ["diff", "--name-status", "-M", f"{after_sha}..HEAD"])
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0].strip()
        if code[:1] in ("R", "C") and len(parts) >= 3:
            old, new = parts[1], parts[2]
            if git.is_source_path(old):
                deleted.add(old)
            if git.is_source_path(new):
                changed.add(new)
        elif len(parts) >= 2:
            path = parts[1]
            if not git.is_source_path(path):
                continue
            (deleted if code[:1] == "D" else changed).add(path)
    return changed, deleted
