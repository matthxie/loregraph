"""kg/pdf.py: per-page classification + packing. Hermetic — builds tiny PDFs with
PyMuPDF itself rather than shipping binary fixtures."""
from __future__ import annotations

import fitz
import pytest

from kg.errors import UnsupportedMedia
from kg.pdf import _classify, _pack_pages, extract_pdf


def test_classify_text_slide_scanned_mixed():
    assert _classify("word " * 500, 0.0) == "text"
    assert _classify("Title\n- a\n- b\n- c", 0.0) == "slide"
    assert _classify("", 0.5) == "scanned"
    assert _classify("word " * 200, 0.1) == "mixed"


def test_pack_pages_caps_size_and_isolates_atomic_units(tmp_path):
    raw = [
        ("text", "para one. " * 20, "page 1", None),
        ("text", "para two. " * 20, "page 2", None),
        ("slide", "Title\n- a\n- b", "Title", None),
        ("scanned", "", "page 4", str(tmp_path / "scan.png")),
        ("mixed", "caption. " * 80, "page 5", str(tmp_path / "fig.png")),
    ]
    pages = _pack_pages(raw, target=300, max_chars=600)
    for p in pages:
        assert len(p.text or "") <= 600
    scanned = [p for p in pages if p.kind == "scanned"]
    assert len(scanned) == 1 and scanned[0].text is None and scanned[0].image_path
    mixed = [p for p in pages if p.kind == "mixed"]
    assert mixed and mixed[0].image_path


def _build_pdf(path):
    doc = fitz.open()
    doc.new_page().insert_text((72, 72), "A short body paragraph.", fontsize=10)
    p = doc.new_page()
    p.insert_text((72, 100), "Quarterly Results", fontsize=28)
    p.insert_text((72, 160), "- Revenue up\n- Costs down", fontsize=14)
    doc.save(str(path))
    doc.close()


def test_extract_pdf_end_to_end(tmp_path):
    pdf_path = tmp_path / "doc.pdf"
    _build_pdf(pdf_path)
    pages = extract_pdf(str(pdf_path), out_dir=str(tmp_path / "pages"),
                        target=2200, max_chars=4400)
    assert len(pages) == 2
    assert pages[1].kind == "slide"
    assert pages[1].breadcrumb == "Quarterly Results"


def test_extract_pdf_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.7")
    with pytest.raises(UnsupportedMedia):
        extract_pdf(str(bad), out_dir=str(tmp_path / "pages"), target=2200, max_chars=4400)
