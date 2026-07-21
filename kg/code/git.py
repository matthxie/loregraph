"""Thin subprocess wrapper over git — the oracle for codebase ingestion.

Everything here is deterministic and offline: `git log`, `git show`, `git diff
--name-status`, `git ls-files`. No network, no state beyond the repo on disk. The
salience gate (skip merges / version bumps / chore / lockfile-only / whitespace-only)
runs BEFORE any LLM call so the event layer stays a record of real work, not churn.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

# Field / record separators for the log format (unlikely to appear in commit text).
_FS = "\x1f"
_RS = "\x1e"
_LOG_FMT = f"%H{_FS}%P{_FS}%cI{_FS}%an{_FS}%B{_RS}"

# Source-file allowlist: the diffs/files worth perceiving. Everything else (data, assets,
# generated output) is out of scope for the event/state layers.
SOURCE_EXTS = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs", ".go", ".rs", ".java",
    ".kt", ".kts", ".scala", ".rb", ".php", ".c", ".h", ".cc", ".cpp", ".hpp", ".cxx",
    ".cs", ".swift", ".m", ".mm", ".sh", ".bash", ".zsh", ".sql", ".r", ".jl", ".lua",
    ".ex", ".exs", ".erl", ".clj", ".hs", ".ml", ".vue", ".svelte",
})

# Paths that are vendored / generated / lockfiles — never source changes worth a record.
_SKIP_PATH_RE = re.compile(
    r"(^|/)(node_modules|vendor|dist|build|out|\.venv|venv|__pycache__|"
    r"third_party|generated|migrations)/|"
    r"(package-lock\.json|yarn\.lock|pnpm-lock\.yaml|poetry\.lock|Cargo\.lock|"
    r"go\.sum|Pipfile\.lock|composer\.lock)$|"
    r"\.min\.(js|css)$|\.(map|lock)$", re.IGNORECASE)

# Commit messages that are pure churn — skip before the LLM.
_CHORE_RE = re.compile(
    r"^\s*(chore|style|ci|build|docs?)\b|"
    r"^\s*(bump|release|version)\b|"
    r"^\s*v?\d+\.\d+(\.\d+)?\s*$|"
    r"^\s*merge\b|^\s*Merge\b", re.IGNORECASE)

# Priority signals — surfaced so a bootstrap can prefer these when it caps the commit count.
_PRIORITY_RE = re.compile(r"^\s*(fix|feat|refactor|perf)\b", re.IGNORECASE)


class GitError(RuntimeError):
    """A git subprocess failed (not a repo, bad ref, git absent)."""


@dataclass
class Commit:
    sha: str
    parents: list[str]
    committed_at: str          # ISO-8601 (%cI)
    author: str
    message: str               # full %B (subject + body)

    @property
    def sha12(self) -> str:
        return self.sha[:12]

    @property
    def subject(self) -> str:
        return (self.message or "").strip().splitlines()[0].strip() if self.message.strip() else ""

    @property
    def is_merge(self) -> bool:
        return len(self.parents) >= 2

    @property
    def is_priority(self) -> bool:
        return bool(_PRIORITY_RE.match(self.subject))


@dataclass
class FileChange:
    status: str                # A | M | D | R | C (rename/copy collapsed to R/C + new path)
    path: str                  # the post-change path (for R/C, the new name)
    old_path: str = ""         # the pre-change path for renames/copies


@dataclass
class SalientCommit:
    """A commit that passed the salience gate, with its source-file changes + capped diff."""
    commit: Commit
    changes: list[FileChange] = field(default_factory=list)   # source-file changes only
    diff: str = ""                                            # whitespace-insensitive, capped


def _run(repo: str, args: list[str], *, check: bool = True) -> str:
    try:
        proc = subprocess.run(["git", "-C", repo, *args],
                              capture_output=True, text=True, check=False)
    except FileNotFoundError as e:  # git not installed
        raise GitError("git executable not found") from e
    if check and proc.returncode != 0:
        raise GitError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def is_repo(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    out = _run(path, ["rev-parse", "--is-inside-work-tree"], check=False).strip()
    return out == "true"


def head_sha(repo: str) -> str | None:
    out = _run(repo, ["rev-parse", "HEAD"], check=False).strip()
    return out or None


def repo_name(path: str) -> str:
    """The repo's display name — the toplevel directory basename."""
    top = _run(path, ["rev-parse", "--show-toplevel"], check=False).strip()
    return os.path.basename(top or os.path.abspath(path.rstrip("/")))


def is_source_path(path: str) -> bool:
    if _SKIP_PATH_RE.search(path):
        return False
    return os.path.splitext(path)[1].lower() in SOURCE_EXTS


def get_commits(repo: str, *, since: str | None = None, after_sha: str | None = None,
                max_count: int = 200) -> list[Commit]:
    """Commits in CHRONOLOGICAL order (oldest→newest, so NEXT edges chain parent→child).

    after_sha → only `after_sha..HEAD` (incremental re-sync). Else `since` (ISO date) or the
    last `max_count` commits (bootstrap). `--reverse` orders the selected set oldest-first."""
    args = ["log", f"--format={_LOG_FMT}", "--reverse"]
    if after_sha:
        args.append(f"{after_sha}..HEAD")
    elif since:
        args += ["--since", since]
    elif max_count:
        args += ["-n", str(max_count)]
    raw = _run(repo, args)
    commits: list[Commit] = []
    for rec in raw.split(_RS):
        rec = rec.strip("\n")
        if not rec.strip():
            continue
        fields = rec.split(_FS)
        if len(fields) < 5:
            continue
        sha, parents, cdate, author, message = fields[0], fields[1], fields[2], fields[3], fields[4]
        commits.append(Commit(sha=sha.strip(),
                              parents=[p for p in parents.strip().split() if p],
                              committed_at=cdate.strip(), author=author.strip(),
                              message=message.strip("\n")))
    return commits


def name_status(repo: str, sha: str) -> list[FileChange]:
    """The (status, path) file changes a commit introduced vs its first parent."""
    raw = _run(repo, ["show", sha, "--name-status", "--format=", "-M", "--no-color"])
    changes: list[FileChange] = []
    for line in raw.splitlines():
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split("\t")
        code = parts[0].strip()
        if code[:1] in ("R", "C") and len(parts) >= 3:
            changes.append(FileChange(status=code[:1], path=parts[2], old_path=parts[1]))
        elif len(parts) >= 2:
            changes.append(FileChange(status=code[:1], path=parts[1]))
    return changes


def commit_diff(repo: str, sha: str, paths: list[str], cap: int) -> str:
    """Whitespace-insensitive diff of a commit restricted to `paths`, capped at `cap` chars.
    `-w` means a whitespace-only change yields an empty diff — the salience gate keys off that."""
    if not paths:
        return ""
    args = ["show", sha, "--format=", "-w", "--no-color", "-M", "--", *paths]
    return _run(repo, args)[:cap]


def salient_commit(repo: str, commit: Commit, *, diff_cap: int) -> SalientCommit | None:
    """Apply the salience gate to one commit; return a SalientCommit or None if it's churn.

    Skips: merges, chore/version-bump/docs messages, commits touching no source files, and
    whitespace-only source changes (the `-w` diff comes back empty)."""
    if commit.is_merge:
        return None
    if _CHORE_RE.match(commit.subject):
        return None
    changes = [c for c in name_status(repo, commit.sha)
               if is_source_path(c.path) or (c.old_path and is_source_path(c.old_path))]
    if not changes:
        return None
    paths = sorted({c.path for c in changes} | {c.old_path for c in changes if c.old_path})
    diff = commit_diff(repo, commit.sha, paths, diff_cap)
    if not diff.strip():                    # whitespace-only (‑w collapsed it) → not real work
        return None
    return SalientCommit(commit=commit, changes=changes, diff=diff)


def list_source_files(repo: str) -> list[str]:
    """Source files tracked at HEAD, allowlisted + de-vendored, in sorted order."""
    raw = _run(repo, ["ls-files"])
    return sorted(p for p in (ln.strip() for ln in raw.splitlines())
                  if p and is_source_path(p))


def read_file(repo: str, path: str) -> str | None:
    """The current on-disk content of a tracked file (HEAD working tree). None if unreadable."""
    full = os.path.join(repo, path)
    try:
        with open(full, encoding="utf-8", errors="ignore") as f:
            return f.read()
    except OSError:
        return None
