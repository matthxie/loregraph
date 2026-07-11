"""Erasure (kg/forget.py): forgotten content must be unreachable on every retrieval
surface, everything else must be untouched, and the operation must be idempotent."""
from __future__ import annotations

import os
import tempfile
from unittest import mock

from kg.forget import forget
from kg.graph import KnowledgeGraph
from kg.models import NodeType
from kg.retrieval import HybridRetriever
from kg.store import fact_active
from tests.test_rag import becky_graph


def _valid_episode_ids(g):
    return {nid for nid, n in g.store.nodes.items()
            if n.ntype == NodeType.EPISODE and n.valid}


def test_forget_tombstones_episode_and_its_facts():
    g = becky_graph()
    before = _valid_episode_ids(g)
    victim = sorted(before)[0]
    rep = forget(g.store, episode_ids=[victim])
    assert victim in rep.episodes
    assert not g.store.nodes[victim].valid
    # facts extracted from the victim are retracted for every temporal view
    for u, v, k, d in g.store.g.edges(keys=True, data=True):
        if d.get("episode_id") == victim and d.get("etype") == "RELATED_TO":
            assert not fact_active(d, None)
            assert not fact_active(d, "2023-01-01")
    # everything else is untouched
    assert _valid_episode_ids(g) == before - {victim}


def test_forget_source_id_expands_to_chunks_and_match_sweeps_text():
    g = becky_graph()
    eps = sorted(_valid_episode_ids(g))
    base = eps[0].split("#c")[0]
    rep = forget(g.store, episode_ids=[base])
    assert all(e.split("#c")[0] == base for e in rep.episodes)
    # text sweep: forget by content, not id
    g2 = becky_graph()
    victim = sorted(_valid_episode_ids(g2))[1]
    text = (g2.store.nodes[victim].raw_text or "")[:40]
    if text.strip():
        rep2 = forget(g2.store, match=text)
        assert victim in rep2.episodes


def test_forgotten_content_is_unretrievable():
    g = becky_graph()
    retr = HybridRetriever(g.store, g.embedder, g.canon, g.config)
    res = retr.retrieve("Where does Becky live?", k=g.config.top_k)
    assert res.objects, "sanity: retrieval works before forgetting"
    top = [oid for oid, _ in res.objects]
    forget(g.store, episode_ids=[oid.split("#c")[0] for oid in top])
    res2 = retr.retrieve("Where does Becky live?", k=g.config.top_k)
    got = {oid for oid, _ in res2.objects}
    assert not (got & set(top)), "forgotten chunks still retrievable"
    # seeds must not reference invalid nodes either
    for nid in res2.seeds:
        n = g.store.get_node(nid)
        assert n is None or n.valid


def test_forget_is_idempotent_and_survives_save_load():
    g = becky_graph()
    victim = sorted(_valid_episode_ids(g))[0]
    r1 = forget(g.store, episode_ids=[victim])
    r2 = forget(g.store, episode_ids=[victim])
    assert r2.total() == 0, "second forget must be a no-op"
    g.store.save()
    reopened = KnowledgeGraph.open(g.store.path, g.config)
    assert not reopened.store.nodes[victim].valid, "tombstone lost on save/load"


# --------------------------------------------------------------------------- #
# erase(): the production sweep -> confirm -> redact -> trace-back pipeline
# --------------------------------------------------------------------------- #
import json as _json
import types as _types

from kg.forget import REDACTED, Eraser, _sentences


class _FakeJudge:
    """Minimal OpenAI-shaped client: judge fuzzy hits via a canned decision function,
    and answer the audit probe with a fixed string."""
    def __init__(self, decide=None, audit_reply="UNKNOWN"):
        self._decide = decide or (lambda prompt: None)
        self._audit_reply = audit_reply
        self.calls = []
        self.chat = _types.SimpleNamespace(completions=_types.SimpleNamespace(
            create=self._create))

    def _create(self, *, model, max_tokens, messages):
        prompt = messages[0]["content"]
        self.calls.append(prompt)
        if "state the most likely value" in prompt.lower():
            content = self._audit_reply
        else:
            idx = self._decide(prompt)
            content = _json.dumps({"contains": idx is not None,
                                   "sentences": idx or []})
        msg = _types.SimpleNamespace(content=content)
        choice = _types.SimpleNamespace(message=msg)
        return _types.SimpleNamespace(choices=[choice])


def _eraser(g, client=None):
    return Eraser(g.store, g.embedder, g.canon, g.config,
                  extractor=None, client=client)


def _inject(g, eid_hint, text):
    """Append a secret-bearing sentence to one valid episode chunk; re-embed so the
    vector matches the text (as ingest would have)."""
    from kg.models import NodeType
    eid = sorted(nid for nid, n in g.store.nodes.items()
                 if n.ntype == NodeType.EPISODE and n.valid)[eid_hint]
    node = g.store.nodes[eid]
    node.raw_text = (node.raw_text or node.name or "context sentence one.") + \
        " " + text
    g.store.vectors.add("episode", eid, g.embedder.embed([node.raw_text])[0])
    return eid


def test_erase_redacts_matched_sentence_and_keeps_the_rest():
    g = becky_graph()
    secret = "my private address is 42 Elm Street"
    eid = _inject(g, 0, f"{secret}. The pottery class was great fun today.")
    before = g.store.nodes[eid].raw_text
    rep = _eraser(g).erase(secret, escalate=False)
    acts = [a for a in rep.actions if a.episode_id == eid]
    assert acts and acts[0].kind == "redact"
    after = g.store.nodes[eid].raw_text
    assert "42 Elm Street" not in after
    assert REDACTED in after
    assert "pottery class" in after, "kept content must survive"
    assert before != after
    # the chunk's vector was re-embedded over the redacted text
    vec = g.store.vectors.get("episode", eid)
    import numpy as np
    ref = g.embedder.embed([after])[0]
    assert float(vec @ (ref / np.linalg.norm(ref))) > 0.99


def test_erase_is_exhaustive_across_restatements_and_reaches_fixpoint():
    g = becky_graph()
    secret = "the vault passcode is 8341"
    e1 = _inject(g, 0, f"{secret}.")
    e2 = _inject(g, 1, f"As mentioned, {secret}!")
    rep = _eraser(g).erase(secret, escalate=False)
    hit = {a.episode_id for a in rep.actions}
    assert {e1, e2} <= hit, "every literal restatement must be found (not top-k)"
    for eid in (e1, e2):
        assert "8341" not in (g.store.nodes[eid].raw_text or "")
    # a second erase finds nothing — the loop reached its fixpoint
    rep2 = _eraser(g).erase(secret, escalate=False)
    assert not rep2.actions


def test_erase_dry_run_mutates_nothing():
    g = becky_graph()
    secret = "my social insurance number is 998-123-456"
    eid = _inject(g, 0, f"{secret}.")
    before = g.store.nodes[eid].raw_text
    rep = _eraser(g).erase(secret, dry_run=True, escalate=False)
    assert rep.dry_run and any(a.episode_id == eid for a in rep.actions)
    assert g.store.nodes[eid].raw_text == before


def test_erase_judge_confirms_paraphrase_and_audit_runs():
    g = becky_graph()
    secret = "I keep a spare key under the blue flowerpot"
    eid = _inject(g, 0, "The spare key stays hidden beneath the blue flowerpot outside.")
    # deterministic gate can't confirm the paraphrase; the judge does
    def decide(prompt):
        if "flowerpot" in prompt:
            sents = _sentences(g.store.nodes[eid].raw_text or "")
            return [i for i, s in enumerate(sents) if "flowerpot" in s.lower()]
        return None
    client = _FakeJudge(decide=decide, audit_reply="UNKNOWN")
    rep = _eraser(g, client=client).erase(secret, escalate=True)
    assert any(a.episode_id == eid and a.reason == "judge" for a in rep.actions)
    assert "flowerpot" not in (g.store.nodes[eid].raw_text or "")
    assert rep.audit == "clean"
    assert rep.llm_calls >= 1


def test_erase_audit_leak_escalates_to_tombstone():
    g = becky_graph()
    secret = "my locker combination is 12-34-56"
    eid = _inject(g, 0, f"Reminder that {secret}.")
    # audit model "reconstructs" the secret -> contributing chunks must be tombstoned
    client = _FakeJudge(audit_reply="The locker combination is 12-34-56.")
    rep = _eraser(g, client=client).erase(secret, escalate=True)
    assert rep.audit == "leaked -> escalated"
