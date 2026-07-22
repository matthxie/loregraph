"""Cold-start chat-history import tests (BUILD BRIEF §Tests), fully hermetic.

Tiny fixture exports for each supported source (ChatGPT mapping-tree incl. one abandoned
edit branch + an image_asset_pointer; Claude flat array + an attachment; Gemini activity
records). A ScriptedExtractor stands in for the live VLM/LLM — extract_image is keyed by
the resolved image path so an image turn gets a real inline description; unknown session
text yields an empty Extraction (the episode still writes). Embeddings use the real local
bge model.

Asserts: (a) detect classifies each source, an unknown file raises the closed-set error;
(b) ChatGPT linearizes the active branch and drops the abandoned regeneration; (c) an image
message yields an inline `[image: …]` description in the session text; (d) sessions carry
the correct created_at; (e) re-importing the same export is idempotent (skips); (f)
source="grok" raises InvalidInput.

Run: .venv/bin/python -m pytest tests/test_import.py -q
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile

import pytest

from kg.engine import Engine
from kg.errors import InvalidInput
from kg.extractors import Extraction, ScriptedExtractor
from kg.imports import build_corpus_items, detect_from_data
from kg.imports.detect import UNRECOGNIZED, load_export
from kg.imports.timeutil import to_iso

# --------------------------------------------------------------------------- #
# Fixtures — the three export formats, written to a temp dir on demand
# --------------------------------------------------------------------------- #
_T0 = 1700000000.0                      # a fixed ChatGPT create_time (Unix seconds)
_ASSET = "file-ABC123"                  # the image_asset_pointer stem


def _chatgpt_export(dirpath: str) -> str:
    """A ChatGPT conversations.json with an abandoned assistant regeneration (a1_lyon) that
    is NOT on the current_node path, plus a final image turn. Writes the bundled image file
    the asset pointer resolves to."""
    with open(os.path.join(dirpath, f"{_ASSET}-photo.png"), "wb") as f:
        f.write(b"\xff\xd8\xff\xe0fake")
    def msg(role, t, ctype, parts):
        return {"author": {"role": role}, "create_time": t,
                "content": {"content_type": ctype, "parts": parts}, "metadata": {}}
    mapping = {
        "root": {"id": "root", "message": None, "parent": None, "children": ["u1"]},
        "u1": {"id": "u1", "parent": "root", "children": ["a_lyon", "a_paris"],
               "message": msg("user", _T0, "text", ["What's the capital of France?"])},
        # abandoned regeneration — a sibling of the surviving answer, off the active path
        "a_lyon": {"id": "a_lyon", "parent": "u1", "children": [],
                   "message": msg("assistant", _T0 + 30, "text", ["It is Lyon."])},
        "a_paris": {"id": "a_paris", "parent": "u1", "children": ["u2"],
                    "message": msg("assistant", _T0 + 40, "text", ["It is Paris."])},
        "u2": {"id": "u2", "parent": "a_paris", "children": ["a2"],
               "message": msg("user", _T0 + 60, "text", ["Show me a photo."])},
        "a2": {"id": "a2", "parent": "u2", "children": [],
               "message": msg("assistant", _T0 + 90, "multimodal_text",
                              [{"content_type": "image_asset_pointer",
                                "asset_pointer": f"file-service://{_ASSET}"},
                               "Here you go."])},
    }
    conv = {"title": "Capitals", "conversation_id": "conv-1",
            "create_time": _T0, "mapping": mapping, "current_node": "a2"}
    path = os.path.join(dirpath, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([conv], f)
    return path


def _claude_export(dirpath: str) -> str:
    conv = {
        "uuid": "c-1", "name": "Trip planning", "created_at": "2024-03-01T09:00:00Z",
        "chat_messages": [
            {"sender": "human", "created_at": "2024-03-01T09:00:00Z",
             "text": "Plan a trip to Japan."},
            {"sender": "assistant", "created_at": "2024-03-01T09:01:00Z",
             "content": [{"type": "text", "text": "Sure, here's an itinerary."}]},
            {"sender": "human", "created_at": "2024-03-01T09:02:00Z",
             "text": "Here is a reference photo.",
             "attachments": [{"file_name": "kyoto.png"}]},   # metadata only, no bytes
        ],
    }
    path = os.path.join(dirpath, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([conv], f)
    return path


def _gemini_export(dirpath: str) -> str:
    records = [
        {"header": "Gemini Apps", "products": ["Gemini Apps"],
         "title": "Prompted What is the capital of France?",
         "time": "2024-01-01T10:00:00.000Z"},
        {"header": "Search", "title": "unrelated search", "time": "2024-01-01T11:00:00Z"},
        {"header": "Gemini Apps", "title": "What is 2+2?",
         "time": "2024-01-02T10:00:00Z",
         "subtitles": [{"name": "2 + 2 equals 4."}]},
    ]
    path = os.path.join(dirpath, "MyActivity.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f)
    return path


def _load(path: str):
    data, _ = load_export(path)
    return data


# --------------------------------------------------------------------------- #
# (a) detection
# --------------------------------------------------------------------------- #
def test_detect_classifies_each_source():
    d = tempfile.mkdtemp()
    assert detect_from_data(_load(_chatgpt_export(tempfile.mkdtemp()))) == "chatgpt"
    assert detect_from_data(_load(_claude_export(tempfile.mkdtemp()))) == "claude"
    assert detect_from_data(_load(_gemini_export(d))) == "gemini"


def test_detect_unknown_file_raises_closed_set_error():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"something": "else", "nope": [1, 2, 3]}], f)
    with pytest.raises(InvalidInput) as ei:
        detect_from_data(_load(path))
    assert UNRECOGNIZED in str(ei.value)


# --------------------------------------------------------------------------- #
# (b) ChatGPT linearizes the active branch, drops the abandoned regeneration
# --------------------------------------------------------------------------- #
def test_chatgpt_linearizes_and_drops_abandoned_branch():
    d = tempfile.mkdtemp()
    path = _chatgpt_export(d)
    resolved, convs, items, _ = build_corpus_items(path, "auto", ScriptedExtractor({}))
    assert resolved == "chatgpt"
    assert len(convs) == 1
    text = items[0].text
    assert "It is Paris." in text                       # surviving branch kept
    assert "It is Lyon." not in text                    # abandoned regeneration dropped
    # active path order preserved: prompt → surviving answer → follow-up
    assert text.index("capital of France") < text.index("It is Paris.")
    assert text.index("It is Paris.") < text.index("Show me a photo.")


# --------------------------------------------------------------------------- #
# (c) an image message inlines a real [image: …] description
# --------------------------------------------------------------------------- #
def test_chatgpt_image_turn_inlines_perceived_description():
    d = tempfile.mkdtemp()
    path = _chatgpt_export(d)
    img = os.path.join(d, f"{_ASSET}-photo.png")
    extractor = ScriptedExtractor(
        {img: Extraction(description="A hand-drawn map of France.")})
    _, _, items, stats = build_corpus_items(path, "chatgpt", extractor)
    assert "[image: A hand-drawn map of France.]" in items[0].text
    assert stats.images_perceived == 1


def test_chatgpt_header_and_turn_format_matches_chunker():
    # The session text must be byte-compatible with the "turns" chunker landmarks.
    from kg.chunkers import _HEADER, _TURN
    d = tempfile.mkdtemp()
    _, _, items, _ = build_corpus_items(_chatgpt_export(d), "chatgpt", ScriptedExtractor({}))
    text = items[0].text
    assert _HEADER.match(text)                          # [chat session — …] header
    assert len(_TURN.findall(text)) >= 2                # >= 2 User:/Assistant: turn starts


# --------------------------------------------------------------------------- #
# (d) sessions carry the correct created_at
# --------------------------------------------------------------------------- #
def test_session_created_at_is_first_message_time():
    d = tempfile.mkdtemp()
    _, _, items, _ = build_corpus_items(_chatgpt_export(d), "chatgpt", ScriptedExtractor({}))
    assert items[0].created_at == to_iso(_T0)           # first user message time


def test_gemini_reconstructs_prompt_response_and_dates():
    d = tempfile.mkdtemp()
    _, convs, items, _ = build_corpus_items(_gemini_export(d), "auto", ScriptedExtractor({}))
    # two Gemini records → two conversations (the Search row is skipped)
    assert len(convs) == 2
    assert items[0].created_at == to_iso("2024-01-01T10:00:00.000Z")
    # prompt prefix stripped; response reconstructed from subtitles on the 2nd record
    joined = "\n".join(i.text for i in items)
    assert "What is the capital of France?" in joined
    assert "2 + 2 equals 4." in joined


# --------------------------------------------------------------------------- #
# Claude: attachment with no bytes → unavailable placeholder, code untyped
# --------------------------------------------------------------------------- #
def test_claude_missing_attachment_bytes_yields_unavailable_placeholder():
    d = tempfile.mkdtemp()
    _, _, items, stats = build_corpus_items(_claude_export(d), "auto", ScriptedExtractor({}))
    assert "[image: unavailable]" in items[0].text
    assert stats.images_perceived == 0
    assert "Plan a trip to Japan." in items[0].text


def test_claude_text_attachment_is_inlined_not_mislabeled_image():
    """Real Claude exports carry PASTED TEXT DOCUMENTS as attachments (file_type txt/md/pdf,
    extracted_content, often an EMPTY file_name). They must be inlined as text — never turned
    into an `[image: unavailable]` placeholder that silently drops the content."""
    d = tempfile.mkdtemp()
    conv = {"uuid": "c-2", "name": "Application", "created_at": "2024-05-01T09:00:00Z",
            "chat_messages": [
                {"sender": "human", "created_at": "2024-05-01T09:00:00Z",
                 "text": "Here is the job description.",
                 "attachments": [{"file_name": "", "file_type": "txt", "file_size": 42,
                                  "extracted_content": "SENIOR ML ENGINEER at Acme Corp."}],
                 "files": [{"file_uuid": "u1", "file_name": "Resume.pdf"},
                           {"file_uuid": "u2", "file_name": ""}]}]}
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([conv], f)
    _, _, items, stats = build_corpus_items(path, "claude", ScriptedExtractor({}))
    text = items[0].text
    assert "SENIOR ML ENGINEER at Acme Corp." in text     # extracted_content inlined
    assert "[image:" not in text                          # NOT mislabeled as an image
    assert stats.images_perceived == 0 and stats.images_unavailable == 0
    assert "[attachment: Resume.pdf]" in text             # named non-image file noted
    # the empty-name, contentless uuid reference is dropped, not mislabeled
    assert text.count("[attachment") == 2                 # the txt doc + Resume.pdf only


# --------------------------------------------------------------------------- #
# (e) + (f) engine-level: idempotent re-import, and the closed source set
# --------------------------------------------------------------------------- #
def _engine_with(extractor) -> Engine:
    eng = Engine.open(tempfile.mkdtemp(), {"kind": "mock"})
    eng._g.extractor = extractor
    return eng


def test_reimport_is_idempotent():
    d = tempfile.mkdtemp()
    path = _chatgpt_export(d)
    img = os.path.join(d, f"{_ASSET}-photo.png")
    eng = _engine_with(ScriptedExtractor({img: Extraction(description="A map.")}))
    first = eng.import_conversations(path, source="chatgpt")
    assert first.source == "chatgpt"
    assert first.conversations == 1
    assert first.episodes_ingested > 0
    second = eng.import_conversations(path, source="chatgpt")
    assert second.episodes_ingested == 0                # nothing new on a re-run
    assert second.skipped > 0                           # every session already present
    eng.close()


def test_auto_source_resolves_through_engine():
    d = tempfile.mkdtemp()
    eng = _engine_with(ScriptedExtractor({}))
    rep = eng.import_conversations(_claude_export(d), source="auto")
    assert rep.source == "claude"
    assert rep.episodes_ingested > 0
    eng.close()


def test_invalid_source_raises_invalid_input():
    d = tempfile.mkdtemp()
    path = _chatgpt_export(d)
    eng = _engine_with(ScriptedExtractor({}))
    with pytest.raises(InvalidInput):
        eng.import_conversations(path, source="grok")
    eng.close()


def test_import_from_zip_directly():
    """Exports ship as .zip — import must consume the archive without a manual unzip."""
    d = tempfile.mkdtemp()
    export = _chatgpt_export(tempfile.mkdtemp())
    base = os.path.dirname(export)
    zpath = os.path.join(d, "export.zip")
    with zipfile.ZipFile(zpath, "w") as zf:
        for name in os.listdir(base):                     # conversations.json + the image file
            zf.write(os.path.join(base, name), name)
    resolved, convs, items, _ = build_corpus_items(zpath, "auto", ScriptedExtractor({}))
    assert resolved == "chatgpt" and len(convs) == 1 and items
    assert "It is Paris." in items[0].text


def test_untimed_conversation_yields_dateless_header_and_none_created_at():
    """A conversation with no timestamps anywhere must not crash: created_at falls to None
    (ingest uses wall clock) and the header degrades to a bare `[chat session]`."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"uuid": "u", "chat_messages": [
            {"sender": "human", "text": "hello there"},
            {"sender": "assistant", "text": "hi back"}]}], f)
    _, _, items, _ = build_corpus_items(path, "claude", ScriptedExtractor({}))
    assert items[0].created_at is None
    assert items[0].text.splitlines()[0] == "[chat session]"


def test_chatgpt_broken_parent_chain_falls_back_without_loss():
    """current_node whose parent is a missing id must fall back to a readable order, not
    return an empty conversation."""
    d = tempfile.mkdtemp()
    mp = {"a": {"id": "a", "parent": "GHOST", "children": [],
                "message": {"author": {"role": "user"}, "create_time": 1,
                            "content": {"content_type": "text", "parts": ["orphaned turn"]}}}}
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"mapping": mp, "current_node": "a", "conversation_id": "c"}], f)
    _, _, items, _ = build_corpus_items(path, "chatgpt", ScriptedExtractor({}))
    assert items and "orphaned turn" in items[0].text


def test_chatgpt_cyclic_parent_chain_terminates():
    """A malformed mapping with a parent cycle must not hang the linearizer."""
    d = tempfile.mkdtemp()
    def m(t):
        return {"author": {"role": "user"}, "create_time": t,
                "content": {"content_type": "text", "parts": [f"turn{t}"]}}
    mp = {"a": {"id": "a", "parent": "b", "children": [], "message": m(1)},
          "b": {"id": "b", "parent": "a", "children": ["a"], "message": m(2)}}
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"mapping": mp, "current_node": "a", "conversation_id": "c"}], f)
    # completes (the seen-guard breaks the cycle) — no assertion on content, just no hang
    _, _, items, _ = build_corpus_items(path, "chatgpt", ScriptedExtractor({}))
    assert items is not None


def test_auto_unrecognized_export_raises_closed_set_error():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "conversations.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{"totally": "unknown"}], f)
    eng = _engine_with(ScriptedExtractor({}))
    with pytest.raises(InvalidInput) as ei:
        eng.import_conversations(path, source="auto")
    assert UNRECOGNIZED in str(ei.value)
    eng.close()
