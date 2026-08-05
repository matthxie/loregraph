"""Media-format support tests: codex CLI image attachment + loud unprocessable-format errors.

Covers the fix for images silently dying on the subscription CLI providers (the model used
to receive "[image omitted: CLI provider is text-only]" and its can't-see-the-image reply
was persisted as the episode's description/retrieval surface):

  * CodexClient now decodes ``image_url`` data-URI blocks back to files and attaches them
    with ``codex exec -i`` (verified against the CLI's --image flag), so vision extraction
    works on the ChatGPT-subscription sign-in;
  * ClaudeClient (no image flag in ``claude -p``) raises ProviderError instead of sending
    a placeholder;
  * ``_sniff_image_mime`` raises UnsupportedMedia for formats no vision provider accepts
    (it used to default everything to image/jpeg — a PDF became bogus JPEG bytes);
  * Engine.ingest fast-fails UnsupportedMedia at the API boundary for a media-only
    non-image attachment or an unsupported image format (.heic etc.).

Hermetic: subprocess.run is stubbed; no CLI or network is touched.
Run: python -m pytest tests/test_media_support.py -q
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import types

import pytest

from kg import llm_client
from kg.engine import Engine, NoteInput
from kg.errors import ProviderError, UnsupportedMedia
from kg.extractors import _sniff_image_mime
from kg.llm_client import ClaudeClient, CodexClient

_PNG_BYTES = (b"\x89PNG\r\n\x1a\n" + b"\x00" * 16)   # header is all the sniffer needs
_PNG_DATA_URL = "data:image/png;base64," + base64.standard_b64encode(_PNG_BYTES).decode()


def _image_messages(url: str = _PNG_DATA_URL) -> list:
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": url}},
        {"type": "text", "text": "describe the image"},
    ]}]


# --------------------------------------------------------------------------- #
# CodexClient: images ride along as `-i` files
# --------------------------------------------------------------------------- #
def test_codex_attaches_image_via_i_flag(monkeypatch):
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["input"] = kw.get("input")
        # capture the attached file's bytes AT CALL TIME (the call dir is removed after)
        paths = [cmd[i + 1] for i, a in enumerate(cmd) if a == "-i"]
        seen["images"] = [(p, open(p, "rb").read()) for p in paths]
        return types.SimpleNamespace(stdout="A photo of a bike.", stderr="", returncode=0)

    monkeypatch.setattr(llm_client, "_codex_binary", lambda: "/fake/codex")
    monkeypatch.setattr(subprocess, "run", fake_run)
    resp = CodexClient().create(messages=_image_messages())

    assert resp.choices[0].message.content == "A photo of a bike."
    paths = seen["images"]
    assert len(paths) == 1
    path, data = paths[0]
    assert path.endswith(".png") and data == _PNG_BYTES
    # the prompt says the image is attached — never the text-only omission placeholder
    assert "[image attached]" in seen["input"]
    assert "omitted" not in seen["input"]


def test_codex_rejects_non_data_uri_image(monkeypatch):
    called = []
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: called.append(1))
    with pytest.raises(ProviderError, match="data URI"):
        CodexClient().create(messages=_image_messages("https://example.com/x.png"))
    assert not called          # rejected before any subprocess spawns


def test_codex_rejects_unsupported_image_mime(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not spawn")))
    url = "data:image/tiff;base64," + base64.standard_b64encode(b"II*\x00").decode()
    with pytest.raises(ProviderError, match="image/tiff"):
        CodexClient().create(messages=_image_messages(url))


# --------------------------------------------------------------------------- #
# ClaudeClient: no image flag → loud error, not a placeholder
# --------------------------------------------------------------------------- #
def test_claude_cli_errors_on_image():
    with pytest.raises(ProviderError, match="can't process images"):
        ClaudeClient().create(messages=_image_messages())


def test_claude_cli_text_path_unaffected(monkeypatch):
    monkeypatch.setattr(llm_client, "_claude_binary", lambda: "/fake/claude")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout='{"type":"result","result":"hi"}', stderr="", returncode=0))
    resp = ClaudeClient().create(messages=[{"role": "user", "content": "hello"}])
    assert resp.choices[0].message.content == "hi"


# --------------------------------------------------------------------------- #
# MIME sniff: extension, content magic, and the unsupported error
# --------------------------------------------------------------------------- #
def test_sniff_known_extension(tmp_path):
    assert _sniff_image_mime(str(tmp_path / "a.PNG")) == "image/png"
    assert _sniff_image_mime(str(tmp_path / "b.jpeg")) == "image/jpeg"


def test_sniff_magic_bytes_for_extensionless_tempfile(tmp_path):
    # kg/imports/normalize.py writes bytes-backed import images to a ".img" tempfile
    p = tmp_path / "upload.img"
    p.write_bytes(_PNG_BYTES)
    assert _sniff_image_mime(str(p)) == "image/png"


def test_sniff_raises_on_unsupported_format(tmp_path):
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.7 ...")
    with pytest.raises(UnsupportedMedia, match="pdf"):
        _sniff_image_mime(str(p))


# --------------------------------------------------------------------------- #
# Engine.ingest: fast-fail at the API boundary
# --------------------------------------------------------------------------- #
@pytest.fixture()
def engine():
    e = Engine.open(tempfile.mkdtemp(), {"kind": "mock"}, log=None)
    yield e
    e.close()


def _minimal_pdf(path, text: str) -> None:
    import fitz
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), text, fontsize=11)
    doc.save(str(path))
    doc.close()


def test_engine_ingests_media_only_pdf(engine, tmp_path):
    # a PDF is always processable (kg/pdf.py per-page classify+extract), unlike a generic
    # non-image file — no caption required.
    p = tmp_path / "resume.pdf"
    _minimal_pdf(p, "Five years of backend engineering experience at Acme Corp.")
    res = engine.ingest(NoteInput(text="", created_at="2026-07-22T09:00:00Z",
                                  attachments=[str(p)]))
    assert res.episode_id and not res.skipped


def test_engine_rejects_corrupt_pdf(engine, tmp_path):
    p = tmp_path / "broken.pdf"
    p.write_bytes(b"%PDF-1.7")
    with pytest.raises(UnsupportedMedia, match="PDF"):
        engine.ingest(NoteInput(text="", created_at="2026-07-22T09:00:00Z",
                                attachments=[str(p)]))


def test_engine_rejects_unsupported_image_format(engine, tmp_path):
    p = tmp_path / "photo.heic"
    p.write_bytes(b"\x00\x00\x00\x18ftypheic")
    with pytest.raises(UnsupportedMedia, match="heic"):
        engine.ingest(NoteInput(text="from my phone", created_at="2026-07-22T09:00:00Z",
                                attachments=[str(p)]))


def test_engine_captioned_file_still_stored_not_perceived(engine, tmp_path):
    # a non-image, non-pdf attachment WITH a caption stays valid by design: the caption
    # is extracted, the file rides along un-perceived.
    p = tmp_path / "notes.docx"
    p.write_bytes(b"PK\x03\x04")
    res = engine.ingest(NoteInput(text="Meeting notes from the Q3 planning session.",
                                  created_at="2026-07-22T09:00:00Z",
                                  attachments=[str(p)]))
    assert res.episode_id and not res.skipped


def test_engine_captioned_pdf_ingests_pages(engine, tmp_path):
    p = tmp_path / "handout.pdf"
    _minimal_pdf(p, "Roadmap for next quarter.")
    res = engine.ingest(NoteInput(text="Handout from the planning meeting.",
                                  created_at="2026-07-22T09:00:00Z",
                                  attachments=[str(p)]))
    assert res.episode_id and not res.skipped


def test_engine_supported_image_still_ingests(engine, tmp_path):
    p = tmp_path / "bike.png"
    p.write_bytes(_PNG_BYTES)
    res = engine.ingest(NoteInput(text="", created_at="2026-07-22T09:00:00Z",
                                  attachments=[str(p)]))
    assert res.episode_id and not res.skipped
