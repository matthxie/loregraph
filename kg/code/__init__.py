"""Codebase ingestion (docs: the code-ingest build brief) — a git repo becomes memory.

Two layers, one unified me-anchored graph:

  * EVENT layer (the maintained graph): each salient git commit → an immutable Episode.
    Append-only, idempotent by SHA, naturally never stale (a commit never changes). This
    is the differentiator — a record of what the user *did* over time.
  * STATE layer (thin, borrowed): current source files at HEAD → embed-only Episodes,
    superseded-on-change. The "find this in what exists now" surface (Cursor-lite), NOT a
    maintained symbol-node graph.

  * REPO summary (bridge): the repo → one summary Episode + tech/concept entities.

Git is the oracle: `git log`, `git diff --name-status`, `git show`, and a stored
last-ingested SHA per repo as the sync marker. Branches / history rewrites are out of
scope — linear history on the current branch is assumed.
"""
from .ingest_repo import CodeIngestReport, ingest_repo

__all__ = ["ingest_repo", "CodeIngestReport"]
