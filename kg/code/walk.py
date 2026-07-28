"""Gather a repo's summary signals — README + dependency manifests + a file-tree sketch.

Deterministic and offline (reads the working tree only). Manifest dependencies are parsed
out cheaply so they can pre-populate the tech/library entities of the repo-summary record;
the LLM confirms them and adds domain concepts (extract_repo, mirroring the URL path).
"""
from __future__ import annotations

import json
import os
import re

from . import git

_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
_MANIFEST_NAMES = (
    "pyproject.toml", "requirements.txt", "setup.py", "setup.cfg", "Pipfile",
    "package.json", "Cargo.toml", "go.mod", "pom.xml", "build.gradle", "Gemfile",
    "composer.json",
)
_README_CAP = 8000
_MANIFEST_CAP = 4000


def _read(repo: str, name: str, cap: int, ref: str) -> str | None:
    body = git.read_file(repo, name, ref)
    return body[:cap] if body else None


def _tree_sketch(files: list[str], max_lines: int = 60) -> str:
    """A compact directory sketch: top-level dirs with per-dir file counts + a few leaves,
    so the LLM sees the project's shape without an exhaustive inventory."""
    by_dir: dict[str, list[str]] = {}
    for p in files:
        d = os.path.dirname(p) or "."
        by_dir.setdefault(d, []).append(os.path.basename(p))
    lines: list[str] = []
    for d in sorted(by_dir):
        names = sorted(by_dir[d])
        shown = ", ".join(names[:6]) + (f", …(+{len(names) - 6})" if len(names) > 6 else "")
        lines.append(f"{d}/  ({len(names)}): {shown}")
        if len(lines) >= max_lines:
            lines.append("…")
            break
    return "\n".join(lines)


def _parse_libraries(manifests: dict[str, str]) -> list[str]:
    """Best-effort dependency names from the common manifests (deterministic, fail-soft)."""
    libs: list[str] = []

    def add(name: str) -> None:
        name = (name or "").strip()
        if name and name.lower() not in {l.lower() for l in libs}:
            libs.append(name)

    for fname, body in manifests.items():
        if not body:
            continue
        low = fname.lower()
        try:
            if low == "package.json" or low == "composer.json":
                data = json.loads(body)
                for key in ("dependencies", "devDependencies", "require", "require-dev"):
                    for dep in (data.get(key) or {}):
                        add(dep)
            elif low == "requirements.txt":
                for line in body.splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        add(re.split(r"[<>=!~\[; ]", line, 1)[0])
            elif low == "pyproject.toml":
                # dependencies = ["pkg>=1", ...]  and  [tool.poetry.dependencies]
                block = re.search(r"dependencies\s*=\s*\[(.*?)\]", body, re.DOTALL)
                if block:
                    for m in re.finditer(r'["\']([A-Za-z0-9_.\-]+)', block.group(1)):
                        add(m.group(1))
                for m in re.finditer(r"^\s*([A-Za-z0-9_.\-]+)\s*=\s*[\"{]", body, re.MULTILINE):
                    if m.group(1).lower() not in ("python", "name", "version", "description"):
                        add(m.group(1))
            elif low == "cargo.toml":
                block = re.search(r"\[dependencies\](.*?)(?:\n\[|\Z)", body, re.DOTALL)
                if block:
                    for m in re.finditer(r"^\s*([A-Za-z0-9_\-]+)\s*=", block.group(1), re.MULTILINE):
                        add(m.group(1))
            elif low == "go.mod":
                for m in re.finditer(r"^\s*(?:require\s+)?([\w./\-]+)\s+v\d", body, re.MULTILINE):
                    add(m.group(1).split("/")[-1])
            elif low == "gemfile":
                for m in re.finditer(r"gem\s+['\"]([^'\"]+)['\"]", body):
                    add(m.group(1))
        except Exception:  # noqa: BLE001 — a malformed manifest must never fail the gather
            continue
    return libs[:40]


def gather_repo_signals(repo: str, ref: str = "HEAD") -> dict:
    """Assemble the repo-summary signal block: {name, readme, manifests, libraries, tree}, read
    out of `ref`'s tree. Purely deterministic; the LLM (extract_repo) turns it into the summary."""
    name = git.repo_name(repo)
    readme = ""
    for rn in _README_NAMES:
        got = _read(repo, rn, _README_CAP, ref)
        if got:
            readme = got
            break
    manifests: dict[str, str] = {}
    for mn in _MANIFEST_NAMES:
        got = _read(repo, mn, _MANIFEST_CAP, ref)
        if got:
            manifests[mn] = got
    files = git.list_source_files(repo, ref)
    return {
        "name": name,
        "readme": readme,
        "manifests": manifests,
        "libraries": _parse_libraries(manifests),
        "tree": _tree_sketch(files),
    }
