"""Codebase ingestion (kg/code/) wiring tests — a git repo becomes me-anchored memory.

Fully hermetic: git is deterministic and offline, so each test `git init`s a real fixture
repo (no network, no fetch switch needed) and a ScriptedExtractor keyed by commit message +
repo name stands in for the live first-person commit/repo LLM (same policy as
tests/test_temporal.py / tests/test_url_ingest.py). Embeddings use the real local bge model.
Run: python -m pytest tests/test_code_ingest.py -q
"""
from __future__ import annotations

import os
import subprocess
import tempfile

import pytest

from kg.engine import Engine, NoteInput
from kg.extractors import (Extraction, ExtractedEntity, ExtractedRelation,
                           ScriptedExtractor)
from kg.models import EdgeType, EntityType, Modality, NodeType, Provenance

# --------------------------------------------------------------------------- #
# Scripted records: the LLM's job (prose → typed graph) is stubbed; the thing under
# test (git → episodes → supersession) runs for real. Keyed by commit MESSAGE / repo NAME.
# --------------------------------------------------------------------------- #
_FEAT_MSG = "feat: add payments module"
_FIX_MSG = "fix: dedupe webhook retries on idempotency key"
_SEED_NOTE = "Been thinking about idempotency in payment systems all week."

# NOTE: 'me' is deliberately NOT in this record's entities[] — only in the relation. This
# mirrors the live LLM (which anchors relations to 'me' without listing it as an entity) and
# exercises the fallback self-routing in _resolve_endpoint → resolve_entity('me') → the self
# anchor, not a minted 'other' entity.
_FEAT_REC = Extraction(
    entities=[ExtractedEntity("payments.py", EntityType.WORK),
              ExtractedEntity("payments", EntityType.CONCEPT)],
    tags=["feature", "payments"],
    relations=[ExtractedRelation(source="me", target="payments.py",
                                 labels=["added"], provenance=Provenance.EXTRACTED)],
    description="Added a payments module with a charge() entry point.",
)
_FIX_REC = Extraction(
    entities=[ExtractedEntity("me", EntityType.PERSON),
              ExtractedEntity("webhook retry handler", EntityType.WORK),
              ExtractedEntity("idempotency", EntityType.CONCEPT)],
    tags=["bugfix", "webhooks"],
    relations=[ExtractedRelation(source="me", target="idempotency",
                                 labels=["fixed"], provenance=Provenance.EXTRACTED),
               ExtractedRelation(source="webhook retry handler", target="idempotency",
                                 labels=["handles"], provenance=Provenance.EXTRACTED)],
    description="Fixed a double-charge bug in webhook retry by de-duplicating on the "
                "idempotency key.",
)
_SEED_REC = Extraction(
    entities=[ExtractedEntity("me", EntityType.PERSON),
              ExtractedEntity("idempotency", EntityType.CONCEPT)],
    tags=["payments"],
    relations=[ExtractedRelation(source="me", target="idempotency",
                                 labels=["thinking_about"], provenance=Provenance.EXTRACTED)],
)

_TABLE = {_FEAT_MSG: _FEAT_REC, _FIX_MSG: _FIX_REC, _SEED_NOTE: _SEED_REC}

_PAY_V1 = (
    "def charge(amount, card):\n"
    "    # naive: retries can double-charge\n"
    "    return gateway.submit(amount, card)\n"
)
_PAY_V2 = (
    "def charge(amount, card, idempotency_key):\n"
    "    if seen(idempotency_key):\n"
    "        return cached(idempotency_key)\n"
    "    return gateway.submit(amount, card, idempotency_key)\n"
)


def _git(repo: str, *args: str, when: str | None = None) -> str:
    env = dict(os.environ)
    env.update({"GIT_AUTHOR_NAME": "Test", "GIT_AUTHOR_EMAIL": "t@e.co",
                "GIT_COMMITTER_NAME": "Test", "GIT_COMMITTER_EMAIL": "t@e.co"})
    if when:
        env["GIT_AUTHOR_DATE"] = env["GIT_COMMITTER_DATE"] = when
    out = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True,
                         check=True, env=env)
    return out.stdout


def _write(repo: str, rel: str, body: str) -> None:
    full = os.path.join(repo, rel)
    os.makedirs(os.path.dirname(full) or repo, exist_ok=True)
    with open(full, "w", encoding="utf-8") as f:
        f.write(body)


def _fixture_repo() -> str:
    repo = tempfile.mkdtemp(prefix="fixrepo_")
    _git(repo, "init", "-q")
    _git(repo, "config", "commit.gpgsign", "false")
    # 1. feat commit (creates payments.py + a README so the summary has real signal)
    _write(repo, "README.md", "# payapp\n\nA tiny payments service.\n")
    _write(repo, "requirements.txt", "flask>=3\nrequests>=2\n")
    _write(repo, "payments.py", _PAY_V1)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", _FEAT_MSG, when="2026-07-01T10:00:00")
    # 2. a chore commit that MUST be gated out by salience (version bump)
    _write(repo, "VERSION", "1.0.1")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "chore: bump version to 1.0.1", when="2026-07-02T10:00:00")
    # 3. fix commit (changes charge() — a known function — in payments.py)
    _write(repo, "payments.py", _PAY_V2)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", _FIX_MSG, when="2026-07-03T10:00:00")
    return repo


def _engine():
    eng = Engine.open(tempfile.mkdtemp(prefix="kgdata_"), {"kind": "mock"})
    eng._g.extractor = ScriptedExtractor(_TABLE)
    return eng


def _code_episodes(eng, kind_prefix: str) -> list:
    """Valid CODE episodes whose source_ref starts with kind_prefix (repo:/commit:/file:)."""
    store = eng._g.store
    return [n for n in store.nodes.values()
            if n.ntype == NodeType.EPISODE and n.valid and n.modality == Modality.CODE
            and (n.source_ref or "").startswith(kind_prefix)]


# --------------------------------------------------------------------------- #
# (a) each salient commit → one Episode; message AND diff both reflected
# --------------------------------------------------------------------------- #
def test_commits_become_episodes_message_and_diff():
    eng = _engine()
    repo = _fixture_repo()
    rep = eng.ingest_repo(repo)
    # salience gate dropped the version-bump chore commit → exactly 2 commit episodes
    assert rep["commits_seen"] == 3 and rep["commits_ingested"] == 2
    commit_eps = _code_episodes(eng, "commit:")
    assert len(commit_eps) == 2
    fix = next(n for n in commit_eps if "@" in n.source_ref and n.description
               and "idempotency" in n.description)
    ep = eng.episode(fix.id)
    assert ep["modality"] == "code"
    # message reflected: the first-person description is the retrieval surface
    assert ep["description"] == _FIX_REC.description
    # diff reflected: the full commit diff is preserved on an un-rankable SOURCE node
    store = eng._g.store
    src = [d for d, _ in store.neighbors(fix.id, etypes={EdgeType.PART_OF}, direction="out")]
    assert len(src) == 1
    src_node = store.get_node(src[0])
    assert src_node.ntype is NodeType.SOURCE
    assert "idempotency_key" in (src_node.raw_text or "")     # the actual code change
    eng.close()


# --------------------------------------------------------------------------- #
# (b) a commit's concept resolves onto a PRE-EXISTING conversation concept
# --------------------------------------------------------------------------- #
def test_commit_concept_fits_existing_conversation_concept():
    eng = _engine()
    # a prior note about idempotency mints the concept node...
    eng.ingest(NoteInput(text=_SEED_NOTE, created_at="2026-06-01T00:00:00Z"))
    store = eng._g.store
    idem_before = [n for n in store.nodes.values()
                   if n.ntype == NodeType.ENTITY and n.valid
                   and n.name.lower() == "idempotency"]
    assert len(idem_before) == 1
    # ...and the fix commit's 'idempotency' must resolve onto that SAME node, not a new one
    eng.ingest_repo(_fixture_repo())
    idem_after = [n for n in store.nodes.values()
                  if n.ntype == NodeType.ENTITY and n.valid
                  and n.name.lower() == "idempotency"]
    assert len(idem_after) == 1
    assert idem_after[0].id == idem_before[0].id
    eng.close()


# --------------------------------------------------------------------------- #
# (c) commit relations are me-anchored
# --------------------------------------------------------------------------- #
def test_commit_relations_are_me_anchored():
    eng = _engine()
    eng.ingest_repo(_fixture_repo())
    facts = eng.facts("me")
    assert facts["resolved"]
    preds = {(f["source"], f["predicate"], f["target"]) for f in facts["facts"]}
    assert ("me", "fixed", "idempotency") in preds
    assert ("me", "added", "payments.py") in preds
    eng.close()


# --------------------------------------------------------------------------- #
# (d) re-sync after a NEW commit appends exactly one Episode (idempotent by SHA)
# --------------------------------------------------------------------------- #
def test_resync_appends_one_episode_idempotent():
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo)
    before = {n.source_ref for n in _code_episodes(eng, "commit:")}
    assert len(before) == 2
    # re-sync with NO new commits → nothing appended (idempotent by SHA)
    rep_noop = eng.ingest_repo(repo)
    assert rep_noop["commits_ingested"] == 0
    assert {n.source_ref for n in _code_episodes(eng, "commit:")} == before
    # a new salient commit → exactly one new commit episode
    _write(repo, "payments.py", _PAY_V2 + "\n\ndef refund(charge_id):\n    return gateway.refund(charge_id)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: add refund path", when="2026-07-04T10:00:00")
    rep = eng.ingest_repo(repo)
    assert rep["commits_ingested"] == 1
    after = {n.source_ref for n in _code_episodes(eng, "commit:")}
    assert len(after) == 3 and before < after
    eng.close()


# --------------------------------------------------------------------------- #
# (e) editing a file SUPERSEDES its prior file Episode (not duplicate)
# --------------------------------------------------------------------------- #
def test_edit_supersedes_file_episode():
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo)
    store = eng._g.store
    ref = None
    valid = [n for n in _code_episodes(eng, "file:") if n.source_ref.endswith("/payments.py")]
    assert len(valid) == 1
    old_id = valid[0].id
    # edit payments.py and re-sync
    _write(repo, "payments.py", _PAY_V2 + "\n# a comment that changes content materially\nX = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "refactor: tweak payments", when="2026-07-05T10:00:00")
    eng.ingest_repo(repo)
    # exactly one VALID payments.py file episode, and it's the NEW one
    valid_now = [n for n in _code_episodes(eng, "file:") if n.source_ref.endswith("/payments.py")]
    assert len(valid_now) == 1
    assert valid_now[0].id != old_id
    # the prior version is superseded, not deleted
    old = store.get_node(old_id)
    assert old is not None and old.valid is False and old.superseded_by == valid_now[0].id
    eng.close()


# --------------------------------------------------------------------------- #
# (f) a deleted file's Episode is superseded, but its commit history survives
# --------------------------------------------------------------------------- #
def test_delete_supersedes_file_but_keeps_commit_history():
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo)
    store = eng._g.store
    commit_refs_before = {n.source_ref for n in _code_episodes(eng, "commit:")}
    _git(repo, "rm", "-q", "payments.py")
    _git(repo, "commit", "-q", "-m", "refactor: drop payments module", when="2026-07-06T10:00:00")
    eng.ingest_repo(repo)
    # the file episode is gone from the current-state view (superseded)
    valid_files = [n for n in _code_episodes(eng, "file:") if n.source_ref.endswith("/payments.py")]
    assert valid_files == []
    # but every commit episode that ever touched it still exists (immutable history)
    commit_refs_after = {n.source_ref for n in _code_episodes(eng, "commit:")}
    assert commit_refs_before <= commit_refs_after
    eng.close()


# --------------------------------------------------------------------------- #
# repo summary bridge + MODIFIES join
# --------------------------------------------------------------------------- #
def test_repo_summary_and_modifies_edges():
    eng = _engine()
    repo = _fixture_repo()
    rep = eng.ingest_repo(repo)
    assert rep["summarized"] is True
    repo_eps = _code_episodes(eng, "repo:")
    assert len(repo_eps) == 1
    # MODIFIES joins commits to the current file episodes they touched
    assert rep["modifies_edges"] >= 1
    store = eng._g.store
    fix = next(n for n in _code_episodes(eng, "commit:") if "idempotency" in (n.description or ""))
    touched = [d for d, _ in store.neighbors(fix.id, etypes={EdgeType.MODIFIES}, direction="out")]
    assert any(store.get_node(t).source_ref.endswith("/payments.py") for t in touched)
    # and NEXT chains the salient commits into a timeline
    assert rep["next_edges"] == 1
    eng.close()
