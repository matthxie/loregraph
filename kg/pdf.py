"""PDF ingestion (docs/ARCHITECTURE.md §6, pdf case) — classify each page as
text / slide / scanned / mixed, pack same-kind runs to chunk_target/max chars, and hand
back page units that kg/ingest.py turns into ordinary text or image CorpusItems: a scanned
page becomes a plain image (perceived, no text), a mixed page becomes a captioned image
(page text = caption, its biggest embedded figure = the perceived image). Classification is
the only new concept here — perception reuses the existing image co-perception path.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from .chunkers import _pack, _split_oversized
from .errors import UnsupportedMedia

_MIN_PAGE_TEXT_CHARS = 40      # below this + a real figure on the page → scanned
_SLIDE_MAX_CHARS = 500         # short + sparse page → atomic "slide" unit, never merged
_SLIDE_MAX_LINES = 12
_FIGURE_MIN_AREA_RATIO = 0.04  # an embedded image must cover this much of the page to
                               # count as a figure (filters bullets/icons/logos)
_SCANNED_AREA_RATIO = 0.2
_RENDER_DPI = 150


@dataclass
class PdfPage:
    ordinal: int
    kind: str                       # "text" | "slide" | "scanned" | "mixed"
    text: str | None
    breadcrumb: str
    page_no: int                    # 1-based, first page of the unit
    image_path: str | None = None   # rendered whole page (scanned) or biggest figure (mixed)


def _breadcrumb(spans: list[tuple[str, float]], page_no: int) -> str:
    """The page's biggest-font line, if it stands out from the body text — same
    self-describing-chunk idea chunk_markdown's heading breadcrumb uses."""
    if not spans:
        return f"page {page_no}"
    sizes = sorted((s for _, s in spans), reverse=True)
    body = sizes[len(sizes) // 2]
    text, size = max(spans, key=lambda p: p[1])
    if size > body * 1.3 and text.strip():
        return text.strip()[:120]
    return f"page {page_no}"


def _classify(text: str, image_area_ratio: float) -> str:
    stripped = (text or "").strip()
    lines = [ln for ln in stripped.split("\n") if ln.strip()]
    if len(stripped) < _MIN_PAGE_TEXT_CHARS and image_area_ratio > _SCANNED_AREA_RATIO:
        return "scanned"
    if image_area_ratio >= _FIGURE_MIN_AREA_RATIO and len(stripped) >= _MIN_PAGE_TEXT_CHARS:
        return "mixed"
    if stripped and len(stripped) <= _SLIDE_MAX_CHARS and len(lines) <= _SLIDE_MAX_LINES:
        return "slide"
    return "text"


def _scan_page(page) -> tuple[str, float, list[tuple[str, float]], list[dict]]:
    text = page.get_text("text") or ""
    spans = [(s["text"], s["size"])
             for b in page.get_text("dict").get("blocks", [])
             for ln in b.get("lines", []) for s in ln.get("spans", [])
             if s.get("text", "").strip()]
    images = page.get_image_info(xrefs=False)
    page_area = (page.rect.width * page.rect.height) or 1.0
    image_area = sum(max(0.0, im["bbox"][2] - im["bbox"][0])
                     * max(0.0, im["bbox"][3] - im["bbox"][1]) for im in images)
    return text, image_area / page_area, spans, images


def extract_pdf(path: str, *, out_dir: str, target: int, max_chars: int) -> list[PdfPage]:
    """Open `path`, classify + pack every page, rendering whatever the classification
    needs (a whole scanned page, or a mixed page's biggest figure) into `out_dir`."""
    try:
        import fitz  # lazy import: a non-pdf install never pays this cost
    except ImportError as e:
        raise UnsupportedMedia(
            "PDF ingestion requires the optional 'pdf' extra "
            "(pip install 'loregraph[pdf]'). It is not installed by default because "
            "PyMuPDF is AGPL-3.0-or-commercial.") from e

    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    raw: list[tuple[str, str, str, str | None]] = []  # kind, text, breadcrumb, image_path
    try:
        doc = fitz.open(path)
    except Exception as e:  # noqa: BLE001 — corrupt/non-PDF bytes, not an engine bug
        raise UnsupportedMedia(f"can't open {os.path.basename(path)!r} as a PDF: {e}") from e
    try:
        for i, page in enumerate(doc, start=1):
            text, ratio, spans, images = _scan_page(page)
            kind = _classify(text, ratio)
            crumb = _breadcrumb(spans, i)
            image_path = None
            if kind == "scanned":
                image_path = os.path.join(out_dir, f"{stem}_p{i:04d}_scan.png")
                page.get_pixmap(dpi=_RENDER_DPI).save(image_path)
                text = ""
            elif kind == "mixed" and images:
                biggest = max(images, key=lambda im: (im["bbox"][2] - im["bbox"][0])
                             * (im["bbox"][3] - im["bbox"][1]))
                image_path = os.path.join(out_dir, f"{stem}_p{i:04d}_fig.png")
                page.get_pixmap(dpi=_RENDER_DPI, clip=fitz.Rect(biggest["bbox"])).save(image_path)
            raw.append((kind, text.strip(), crumb, image_path))
    finally:
        doc.close()
    return _pack_pages(raw, target=target, max_chars=max_chars)


def _pack_pages(raw: list[tuple[str, str, str, str | None]], *,
               target: int, max_chars: int) -> list[PdfPage]:
    out: list[PdfPage] = []
    ordinal = 0
    i, n = 0, len(raw)
    while i < n:
        kind, text, crumb, image_path = raw[i]
        page_no = i + 1
        if kind == "text":
            run = []
            while i < n and raw[i][0] == "text":
                run.append(raw[i])
                i += 1
            units = [f"{c}\n{t}" if c and t else (c or t) for _, t, c, _ in run]
            for piece in _pack(units, target, max_chars):
                out.append(PdfPage(ordinal=ordinal, kind="text", text=piece,
                                   breadcrumb=run[0][2], page_no=page_no))
                ordinal += 1
            continue
        pieces = _split_oversized(text, max_chars) if text else [""]
        for j, piece in enumerate(pieces):
            out.append(PdfPage(ordinal=ordinal, kind=kind, text=piece or None,
                               breadcrumb=crumb, page_no=page_no,
                               image_path=image_path if j == 0 else None))
            ordinal += 1
        i += 1
    return out
