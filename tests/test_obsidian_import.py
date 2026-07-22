"""Obsidian vault import wiring tests (Engine.import_vault, the two-pass wikilink resolver).

A vault is already a personal knowledge graph, so the importer LEANS on the author's explicit
structure instead of inferring it: one `.md` file → one text Episode, `[[wikilinks]]` →
deterministic HYPERLINKS_TO edges, `#tags` / frontmatter `tags:` → deterministic TAGGED_AS,
`![[img.png]]` → a perceived-and-inlined `[image: …]` marker. These tests assert exactly that
structure lands, and that the two novel guarantees hold: (b) a wikilink becomes an Episode→
Episode edge between the RIGHT two notes; (c) a link to a note that doesn't exist is skipped,
never crashing or stubbing; (d) a vault tag RESOLVES onto the SAME canonical node a prior note
from another source already created (the connective guarantee); (f) re-import is idempotent;
(g) extract=False builds the whole wikilink/tag structure with ZERO LLM calls.

Fully hermetic (same policy as tests/test_image_ingest.py / test_import.py): a ScriptedExtractor
stands in for the live VLM/LLM — extract_image keyed by the embed name, extract_text by the note
body — so no OpenAI call is made. Embeddings use the real local bge model.
Run: python -m pytest tests/test_obsidian_import.py -q
"""
from __future__ import annotations

import os
import tempfile

from kg.engine import Engine, NoteInput
from kg.extractors import Extraction, ScriptedExtractor
from kg.imports import obsidian
from kg.models import EdgeType, NodeType

DIAGRAM_DESC = "A hand-drawn architecture diagram."


# --------------------------------------------------------------------------- #
# Fixture vault
# --------------------------------------------------------------------------- #
def _write(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _vault() -> str:
    """A tiny vault exercising every parse path: frontmatter (date + aliases + tags), inline
    `#tags`, a resolved `[[wikilink]]`, an alias-targeted link, an UNRESOLVED link, a note
    transclusion embed (`![[Alpha]]`), and an image embed (`![[diagram.png]]`) with real bytes."""
    root = tempfile.mkdtemp()
    os.makedirs(os.path.join(root, ".obsidian"), exist_ok=True)
    _write(os.path.join(root, "alpha.md"),
           "---\n"
           "date: 2024-03-10\n"
           "aliases: [Alpha Note]\n"
           "tags: [shared-tag, project]\n"
           "---\n"
           "# Alpha\n\n"
           "Some thoughts on #cycling. See [[Beta]] and [[Nonexistent]].\n\n"
           "![[diagram.png]]\n")
    _write(os.path.join(root, "notes", "beta.md"),
           "---\ntags: [shared-tag]\n---\n"
           "# Beta\n\n"
           "More on #cycling. Back to [[Alpha Note]] via its alias.\n\n"
           "![[Alpha]]\n")                       # a transclusion embed → still a HYPERLINKS_TO
    with open(os.path.join(root, "diagram.png"), "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nfake-png-bytes")
    return root


def _engine():
    return Engine.open(tempfile.mkdtemp(), {"kind": "mock"})


def _ep_id(rel_path: str) -> str:
    """The deterministic Episode id for a vault note (mirrors the importer's id derivation)."""
    return f"ep_{obsidian._item_id(rel_path)}"


def _scripted():
    # extract_image is keyed by the embed name (the alt the importer passes as label_hint);
    # note bodies aren't scripted → empty extraction (deterministic tags/links do the wiring).
    return ScriptedExtractor({"diagram.png": Extraction(description=DIAGRAM_DESC)})


def _tag_neighbors(eng, ep_id):
    return {tid for tid, _d in eng._g.store.neighbors(
        ep_id, etypes={EdgeType.TAGGED_AS}, direction="out")}


def _hyperlink_targets(eng, ep_id):
    return {dst for dst, _d in eng._g.store.neighbors(
        ep_id, etypes={EdgeType.HYPERLINKS_TO}, direction="out")}


# --------------------------------------------------------------------------- #
# (a) each note → an Episode with body surface + frontmatter date as created_at
# --------------------------------------------------------------------------- #
def test_note_becomes_episode_with_body_and_frontmatter_date():
    eng = _engine()
    eng._g.extractor = _scripted()
    rep = eng.import_vault(_vault())
    assert rep.notes == 2
    alpha = eng.episode(_ep_id("alpha.md"))
    assert alpha is not None
    assert alpha["title"] == "Alpha"                       # H1, not the filename
    assert "Some thoughts on #cycling" in alpha["text"]    # frontmatter stripped, body kept
    assert not alpha["text"].startswith("---")             # the YAML block is gone
    assert alpha["created_at"] == "2024-03-10T00:00:00+00:00"   # frontmatter date won
    eng.close()


# --------------------------------------------------------------------------- #
# (b) a wikilink → a HYPERLINKS_TO edge between the correct two episodes
# --------------------------------------------------------------------------- #
def test_wikilink_becomes_hyperlinks_to_edge():
    eng = _engine()
    eng._g.extractor = _scripted()
    rep = eng.import_vault(_vault())
    a, b = _ep_id("alpha.md"), _ep_id("notes/beta.md")
    # alpha `[[Beta]]` → alpha→beta; beta `[[Alpha Note]]` (alias) + `![[Alpha]]` → beta→alpha
    assert b in _hyperlink_targets(eng, a)
    assert a in _hyperlink_targets(eng, b)
    assert rep.links_resolved >= 2
    eng.close()


# --------------------------------------------------------------------------- #
# (c) an unresolved link is skipped — no stub, no crash
# --------------------------------------------------------------------------- #
def test_unresolved_wikilink_is_skipped_silently():
    eng = _engine()
    eng._g.extractor = _scripted()
    rep = eng.import_vault(_vault())
    assert rep.links_unresolved >= 1                       # `[[Nonexistent]]`
    # no stub Episode was minted for the missing target
    names = {n.name for n in eng._g.store.nodes_of_type(NodeType.EPISODE)}
    assert not any("Nonexistent" in n for n in names)
    eng.close()


# --------------------------------------------------------------------------- #
# (d) a #tag → TAGGED_AS resolving to a canonical tag SHARED with a prior note
# --------------------------------------------------------------------------- #
def test_tag_resolves_to_node_shared_with_prior_note():
    eng = _engine()
    # a prior note from ANOTHER source (a plain conversation note) establishes the shared-tag
    # node; the vault's `tags: [shared-tag]` must resolve ONTO it, not fork a parallel node.
    eng._g.extractor = ScriptedExtractor({
        "I love this project.": Extraction(tags=["shared-tag"]),
        "diagram.png": Extraction(description=DIAGRAM_DESC)})
    prior = eng.ingest(NoteInput(text="I love this project.",
                                 created_at="2024-01-01T00:00:00Z"))
    prior_tags = _tag_neighbors(eng, prior.episode_id)
    assert prior_tags, "prior note laid down no tag node"

    eng.import_vault(_vault())
    alpha_tags = _tag_neighbors(eng, _ep_id("alpha.md"))
    beta_tags = _tag_neighbors(eng, _ep_id("notes/beta.md"))
    # the connective guarantee: alpha, beta AND the prior note all share one tag node
    assert prior_tags & alpha_tags, "vault tag forked instead of resolving onto the prior node"
    assert alpha_tags & beta_tags, "the two notes' shared tag forked into parallel nodes"
    # `#cycling` (inline) is also present alongside the frontmatter tags
    tag_names = {eng._g.store.get_node(t).name for t in alpha_tags}
    assert any("cycling" in n for n in tag_names)
    eng.close()


# --------------------------------------------------------------------------- #
# (e) an embedded image → an inline [image: …] description
# --------------------------------------------------------------------------- #
def test_image_embed_is_perceived_and_inlined():
    eng = _engine()
    eng._g.extractor = _scripted()
    rep = eng.import_vault(_vault())
    alpha = eng.episode(_ep_id("alpha.md"))
    assert f"[image: {DIAGRAM_DESC}]" in alpha["text"]     # perceived, spliced into the body
    assert "![[diagram.png]]" not in alpha["text"]         # the raw embed token was replaced
    assert rep.images_perceived == 1
    eng.close()


# --------------------------------------------------------------------------- #
# (f) re-import is idempotent
# --------------------------------------------------------------------------- #
def test_reimport_is_idempotent():
    eng = _engine()
    eng._g.extractor = _scripted()
    vault = _vault()
    eng.import_vault(vault)
    store = eng._g.store
    eps_before = len(store.nodes_of_type(NodeType.EPISODE))
    a = _ep_id("alpha.md")
    tags_before = _tag_neighbors(eng, a)
    links_before = _hyperlink_targets(eng, a)

    rep2 = eng.import_vault(vault)             # unchanged vault → nothing new
    assert rep2.episodes_ingested == 0
    assert rep2.skipped == 2
    assert len(store.nodes_of_type(NodeType.EPISODE)) == eps_before
    assert _tag_neighbors(eng, a) == tags_before          # edges collapse by identity
    assert _hyperlink_targets(eng, a) == links_before
    eng.close()


# --------------------------------------------------------------------------- #
# (g) extract=False builds the wikilink/tag structure WITHOUT any LLM call
# --------------------------------------------------------------------------- #
class _RaisingExtractor:
    """Fails loudly if any extraction pass is invoked — proves structure-only import is
    model-free (the engine swaps in its own no-op extractor for the ingest)."""
    name = "raising"

    def __init__(self):
        from kg.extractors import UsageMeter
        self.meter = UsageMeter()

    def extract_text(self, text, title=""):
        raise AssertionError("extract_text called in structure-only import")

    def extract_image(self, image_path, label_hint=None):
        raise AssertionError("extract_image called in structure-only import")

    def extract_url(self, url):
        raise AssertionError("extract_url called in structure-only import")


def test_structure_only_import_makes_no_llm_call():
    eng = _engine()
    eng._g.extractor = _RaisingExtractor()                 # would raise if consulted
    rep = eng.import_vault(_vault(), extract=False)
    # the structure still landed: episodes, tags, resolved wikilinks
    assert rep.notes == 2 and rep.episodes_ingested == 2
    a, b = _ep_id("alpha.md"), _ep_id("notes/beta.md")
    assert b in _hyperlink_targets(eng, a)
    assert _tag_neighbors(eng, a) and _tag_neighbors(eng, b)
    # images were NOT perceived (no VLM) — a filename placeholder stands in
    assert rep.images_perceived == 0
    alpha = eng.episode(a)
    assert "[image: diagram.png]" in alpha["text"]
    # and the real extractor is restored afterward, untouched
    assert isinstance(eng._g.extractor, _RaisingExtractor)
    eng.close()
