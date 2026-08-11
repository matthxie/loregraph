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
    _git(repo, "init", "-q", "-b", "main")
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


# --------------------------------------------------------------------------- #
# ref-anchoring (Phase 2a): what's ingested follows the REF, not the checkout
# --------------------------------------------------------------------------- #
def _file_text(eng, path_suffix: str) -> str:
    """Concatenated text of the valid file episodes for a path (chunks joined)."""
    eps = [n for n in _code_episodes(eng, "file:") if n.source_ref.endswith(path_suffix)]
    return "\n".join(n.raw_text or "" for n in eps)


def test_ingest_follows_the_ref_not_the_checked_out_branch():
    eng = _engine()
    repo = _fixture_repo()
    # a divergent branch whose payments.py is DIFFERENT, and leave it checked out
    _git(repo, "checkout", "-q", "-b", "side")
    _write(repo, "payments.py", "def only_on_side():\n    return 'side-branch-marker'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: side experiment", when="2026-07-10T10:00:00")

    rep = eng.ingest_repo(repo, ref="main")
    assert rep["ref"] == "main"
    assert rep["head"] == _git(repo, "rev-parse", "main").strip()
    # main's payments.py won, even though `side` is what's on disk
    text = _file_text(eng, "/payments.py")
    assert "idempotency_key" in text and "side-branch-marker" not in text
    # and the episodes are keyed by (ref, path) so 2b can reconcile per branch
    assert any(n.source_ref.startswith("file:") and "@main/" in n.source_ref
               for n in _code_episodes(eng, "file:"))
    eng.close()


def test_base_ancestor_ingests_only_the_commit_range():
    eng = _engine()
    repo = _fixture_repo()
    base = _git(repo, "rev-parse", "main").strip()
    _write(repo, "payments.py", _PAY_V2 + "\n\ndef refund(charge_id):\n    return gateway.refund(charge_id)\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: add refund path", when="2026-07-04T10:00:00")

    rep = eng.ingest_repo(repo, ref="main", base=base)
    assert rep["full"] is False and rep["base"] == base
    assert rep["commits_seen"] == 1 and rep["commits_ingested"] == 1
    eng.close()


def test_base_that_is_not_an_ancestor_falls_back_to_a_full_ingest():
    eng = _engine()
    repo = _fixture_repo()
    # a commit on a branch that main never sees — a stand-in for a rebase / force-push / swap
    _git(repo, "checkout", "-q", "-b", "orphan")
    _write(repo, "orphan.py", "X = 1\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feat: orphan work", when="2026-07-09T10:00:00")
    stale_base = _git(repo, "rev-parse", "orphan").strip()

    rep = eng.ingest_repo(repo, ref="main", base=stale_base)
    assert rep["full"] is True and rep["base"] is None
    assert any("not an ancestor" in note for note in rep["notes"])
    assert rep["commits_seen"] == 3      # main's whole history, not a range
    eng.close()


# --------------------------------------------------------------------------- #
# deletion reconciliation (Phase 2b): kg only ever holds files that exist on the ref
# --------------------------------------------------------------------------- #
def _delete_payments(repo: str) -> None:
    _git(repo, "rm", "-q", "payments.py")
    _git(repo, "commit", "-q", "-m", "refactor: drop payments module", when="2026-07-06T10:00:00")


@pytest.mark.parametrize("incremental", [False, True])
def test_deleted_file_is_tombstoned_on_both_passes(incremental):
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo, ref="main")
    assert _file_text(eng, "/payments.py")
    base = _git(repo, "rev-parse", "main").strip()
    _delete_payments(repo)

    if incremental:
        rep = eng.ingest_repo(repo, ref="main", base=base)
        assert rep["full"] is False
        removed = rep["files_removed"]
    else:
        # straight through the ingest path so no stored sync marker turns it incremental
        from kg.code import ingest_repo as _low
        report = _low(eng._g, repo, ref="main")
        assert report.full is True
        removed = report.files_removed
    assert removed >= 1
    assert _file_text(eng, "/payments.py") == ""
    # tombstoned, not superseded — nothing points at a replacement
    store = eng._g.store
    dead = [n for n in store.nodes.values()
            if (n.source_ref or "").endswith("@main/payments.py")]
    assert dead and all(not n.valid and not n.superseded_by for n in dead)
    eng.close()


def test_rename_tombstones_old_path_and_ingests_new():
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo, ref="main")
    _git(repo, "mv", "payments.py", "billing.py")
    _git(repo, "commit", "-q", "-m", "refactor: rename payments to billing", when="2026-07-07T10:00:00")

    eng.ingest_repo(repo, ref="main")
    assert _file_text(eng, "/payments.py") == ""
    assert "idempotency_key" in _file_text(eng, "/billing.py")
    eng.close()


def test_delete_on_one_ref_leaves_another_refs_copy_alone():
    eng = _engine()
    repo = _fixture_repo()
    _git(repo, "branch", "side")
    eng.ingest_repo(repo, ref="main")
    eng.ingest_repo(repo, ref="side")
    _delete_payments(repo)                       # on main only
    eng.ingest_repo(repo, ref="main")

    live = {n.source_ref for n in _code_episodes(eng, "file:")
            if n.source_ref.endswith("/payments.py")}
    assert live == {"file:%s@side/payments.py" % os.path.basename(repo)}
    eng.close()


def test_unchanged_file_is_not_spuriously_tombstoned():
    eng = _engine()
    repo = _fixture_repo()
    eng.ingest_repo(repo, ref="main")
    before = {n.id for n in _code_episodes(eng, "file:")}
    rep = eng.ingest_repo(repo, ref="main")
    assert rep["files_removed"] == 0 and rep["files_superseded"] == 0
    assert {n.id for n in _code_episodes(eng, "file:")} == before
    eng.close()


def test_unresolvable_ref_is_invalid_input():
    from kg.engine import InvalidInput
    eng = _engine()
    with pytest.raises(InvalidInput):
        eng.ingest_repo(_fixture_repo(), ref="no-such-branch")
    eng.close()


def test_code_chunks_carry_true_line_spans():
    """Spans must index the ORIGINAL file, so an agent can jump straight to the code."""
    from kg.chunkers import chunk_code

    src = "\n\n".join(f"def f{i}():\n    x = {i}\n    return x * {i}" for i in range(80))
    chunks = chunk_code(src, target=400, max_chars=800)
    assert len(chunks) > 1
    lines = src.split("\n")

    for c in chunks:
        assert 1 <= c.start_line <= c.end_line <= len(lines)
        # the chunk's first line is exactly where it claims to start
        assert c.text.split("\n")[0] == lines[c.start_line - 1]
    assert chunks[0].start_line == 1
    assert chunks[-1].end_line == len(lines)
    # consecutive and non-overlapping
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_line > a.end_line


def test_leading_blank_lines_do_not_shift_spans():
    from kg.chunkers import chunk_code

    body = "\n\n".join(f"def g{i}():\n    return {i}" for i in range(60))
    chunks = chunk_code("\n\n\n" + body, target=300, max_chars=600)
    assert chunks and chunks[0].start_line == 4      # 3 blank lines still cost line numbers
