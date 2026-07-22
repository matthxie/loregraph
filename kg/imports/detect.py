"""Format sniffing for a chat-history export → "chatgpt" | "claude" | "gemini".

The three supported exports are unofficial and drift, so detection keys off the ONE
structural landmark each format cannot be without, never a version string:

  * ChatGPT  — conversations.json: a list whose items carry a `mapping` node tree.
  * Claude   — conversations.json: a list whose items carry a flat `chat_messages[]`.
  * Gemini   — Google Takeout "My Activity": a list of activity records tagged to the
               Gemini Apps product (a `header`/`products` naming Gemini/Bard).

No match raises the closed-set error — never a silent partial parse (BUILD BRIEF). We
fail on the whole *file* only when nothing matches; an individual unknown block inside a
recognized file is fail-soft, handled by the mapper, not here.
"""
from __future__ import annotations

import json
import os
import tempfile
import zipfile

from ..errors import InvalidInput

UNRECOGNIZED = "unrecognized export format — supported: chatgpt, claude, gemini"

# Common file names inside an unzipped export, in the order we probe a directory.
_CANDIDATES = (
    "conversations.json",                                   # ChatGPT / Claude
    "MyActivity.json",                                      # Gemini Takeout (flat)
    os.path.join("My Activity", "Gemini Apps", "MyActivity.json"),
    os.path.join("Takeout", "My Activity", "Gemini Apps", "MyActivity.json"),
)


def load_export(path: str) -> tuple[object, str]:
    """Resolve an export path (a file, an unzipped export directory, OR a .zip of one) to
    (parsed_json, base_dir). base_dir is where bundled media live — the directory the JSON
    sits in — so a mapper can resolve relative attachment pointers against it. A .zip is
    extracted to a temp dir and treated as that directory (exports ship as zips, so consuming
    one directly is the common path).

    Raises InvalidInput when nothing loadable is found (a missing/corrupt export is a caller
    error, not a crash)."""
    p = os.path.expanduser(path)
    if os.path.isfile(p) and zipfile.is_zipfile(p):
        p = _extract_zip(p)
    if os.path.isdir(p):
        return _load_from_dir(p)
    if os.path.isfile(p):
        return _read_json(p), os.path.dirname(os.path.abspath(p))
    raise InvalidInput(f"import path not found: {path}")


def _extract_zip(zip_path: str) -> str:
    """Extract an export .zip to a fresh temp dir and return it. Python's extractall sanitizes
    member paths (drops `..`/absolute roots) so a hostile archive can't escape the temp dir."""
    dest = tempfile.mkdtemp(prefix="kg_import_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
    except (OSError, zipfile.BadZipFile) as e:
        raise InvalidInput(
            f"could not read export zip {os.path.basename(zip_path)}: {e}") from e
    return dest


def _load_from_dir(p: str) -> tuple[object, str]:
    """Find the export JSON in a directory: the known candidate names first (top level, then
    anywhere via a recursive walk — a zip may unpack into a nested Takeout/… folder), else the
    first top-level *.json."""
    for rel in _CANDIDATES:
        cand = os.path.join(p, rel)
        if os.path.isfile(cand):
            return _read_json(cand), os.path.dirname(cand)
    wanted = {os.path.basename(rel).lower() for rel in _CANDIDATES}
    for root, _dirs, files in os.walk(p):
        for name in sorted(files):
            if name.lower() in wanted:
                return _read_json(os.path.join(root, name)), root
    for name in sorted(os.listdir(p)):
        if name.lower().endswith(".json"):
            return _read_json(os.path.join(p, name)), p
    raise InvalidInput(UNRECOGNIZED)


def _read_json(path: str) -> object:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise InvalidInput(f"could not read export {os.path.basename(path)}: {e}") from e


def detect_from_data(data: object) -> str:
    """Sniff already-parsed export JSON to a source label, or raise the closed-set error."""
    if _looks_chatgpt(data):
        return "chatgpt"
    if _looks_claude(data):
        return "claude"
    if _looks_gemini(data):
        return "gemini"
    raise InvalidInput(UNRECOGNIZED)


def detect(path: str) -> tuple[str, object, str]:
    """Load + sniff an export path. Returns (source, parsed_json, base_dir)."""
    data, base_dir = load_export(path)
    return detect_from_data(data), data, base_dir


# --------------------------------------------------------------------------- #
# Per-source landmarks (structural, version-agnostic)
# --------------------------------------------------------------------------- #
def _first_dicts(data: object, limit: int = 20):
    """Yield up to `limit` dict items from a list-shaped export (both ChatGPT and Claude
    top-level are lists; Gemini Takeout is a list too). A dict top-level (some ChatGPT
    exports wrap the list) is probed for a nested list too."""
    if isinstance(data, dict):
        # some exports wrap: {"conversations": [...]} — probe the first list value
        for v in data.values():
            if isinstance(v, list):
                data = v
                break
    if isinstance(data, list):
        n = 0
        for item in data:
            if isinstance(item, dict):
                yield item
                n += 1
                if n >= limit:
                    return


def _looks_chatgpt(data: object) -> bool:
    for item in _first_dicts(data):
        if isinstance(item.get("mapping"), dict):
            return True
    return False


def _looks_claude(data: object) -> bool:
    for item in _first_dicts(data):
        if isinstance(item.get("chat_messages"), list):
            return True
    return False


def _looks_gemini(data: object) -> bool:
    for item in _first_dicts(data):
        header = str(item.get("header", "")).lower()
        products = " ".join(str(x) for x in (item.get("products") or [])).lower()
        tag = f"{header} {products}"
        if ("gemini" in tag or "bard" in tag) and ("time" in item or "titleUrl" in item):
            return True
    return False
