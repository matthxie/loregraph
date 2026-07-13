"""kg.modelpin — pinned, checksum-verified embedding-model provisioning (PROTOCOL §7.7, GAP-7).

Ports the Swift `EmbeddingModelPin` + `OnboardingModel` SECURITY behavior into the daemon:
pin ONE integrity-critical weight artifact at ONE revision, verify it at `model.ensure` and
on every launch, and WIPE-ON-MISMATCH so the engine never embeds with unverified weights.

CRITICAL — this is NOT the Swift pin. The Swift app pins fastembed's
`Xenova/bge-small-en-v1.5` `onnx/model.onnx` (sha 828e1496…40cf35, 133_093_490 bytes). This
engine embeds via **sentence-transformers** over `BAAI/bge-small-en-v1.5` (kg/embedders.py),
whose HF snapshot integrity artifact is `model.safetensors` under
`models--BAAI--bge-small-en-v1.5`. The sha256/bytes below were computed once from a
known-good local download of that artifact at the pinned revision — a DIFFERENT repo AND a
different file than the Swift ONNX pin (do not copy the Swift value).

The download itself is sentence-transformers'/HF-hub's (triggered through the same embedder
load path every later open uses); progress is observed by polling the cache-dir size (the
download is opaque, the dir size is not — the Swift poller technique). All of it runs
synchronously on the daemon's serial loop, emitting `model.progress` notifications.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import threading

# The pin. `sha256`/`bytes` are the REAL values of model.safetensors at `revision`, computed
# once from a verified local download (see the module docstring — never the Swift ONNX value).
EMBED_MODEL_PIN = {
    "repo": "BAAI/bge-small-en-v1.5",              # health.embedder.model
    "cache_component": "models--BAAI--bge-small-en-v1.5",  # HF hub cache dir (kg/embedders.py)
    "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",  # snapshots are rev-addressed
    "artifact": "model.safetensors",              # the weights file in the snapshot
    "sha256": "3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad",
    "bytes": 133466304,                           # drives the determinate progress bar
}

_HASH_CHUNK = 4 << 20   # 4 MiB — stream the ~130 MB artifact, never load it whole

# A hard transport/I-O failure (the download could not complete at all) raises the shared
# kg.errors.ModelUnavailable, which the daemon's dispatcher maps to -32009 (§7.9). NOT the
# checksum-gate outcome — that rides result.state='failed'.
from .errors import ModelUnavailable  # noqa: E402 — grouped with the module's one internal dep


# --------------------------------------------------------------------------- #
# cache locations
# --------------------------------------------------------------------------- #
def cache_root() -> str:
    """The HF hub cache root, using the same resolution kg/embedders.py:_model_is_cached does."""
    return (
        os.environ.get("HF_HUB_CACHE")
        or (os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME")
            else os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    )


def _model_dir(root: str, pin: dict) -> str:
    return os.path.join(root, pin["cache_component"])


def artifact_path(root: str, pin: dict) -> str | None:
    """Absolute path to the pinned weight artifact if present: prefer the pinned revision's
    snapshot, else the first snapshot that carries the artifact (mirrors Swift modelONNXPath's
    scan). os.path.isfile follows the HF blob symlink, so this resolves to real bytes."""
    base = os.path.join(_model_dir(root, pin), "snapshots")
    pinned = os.path.join(base, pin["revision"], pin["artifact"])
    if os.path.isfile(pinned):
        return pinned
    if os.path.isdir(base):
        for rev in sorted(os.listdir(base)):
            cand = os.path.join(base, rev, pin["artifact"])
            if os.path.isfile(cand):
                return cand
    return None


def sha256_file(path: str) -> str:
    """Streaming sha256 of a file (never loads the ~130 MB whole; the Swift sha256Hex port)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def dir_size(path: str) -> int:
    """Total bytes under a dir (the determinate-progress signal — the download is opaque, the
    cache-dir size is not). 0 when absent. Symlinks (HF blob layout) counted at target size."""
    total = 0
    if not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)     # follows symlinks → the blob's real size
            except OSError:
                pass
    return total


def wipe(root: str, pin: dict) -> None:
    """Remove the cached model dir entirely (the wipe-on-mismatch → re-onboard path)."""
    shutil.rmtree(_model_dir(root, pin), ignore_errors=True)


# --------------------------------------------------------------------------- #
# download loader (the normal embedder path, forced online)
# --------------------------------------------------------------------------- #
def _default_loader(pin: dict) -> None:
    """Trigger the download through the SAME embedder load path every later open uses, forced
    online (HF_HUB_OFFLINE=0 so kg/embedders.py:_model_is_cached takes the revalidating path).
    Restores the prior env after so later lazy loads keep their fast offline path. Blocking."""
    prev = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    try:
        from .embedders import SentenceTransformerEmbedder
        emb = SentenceTransformerEmbedder(pin["repo"], 384)
        emb.embed(["warmup"])     # forces the lazy model load → downloads if absent
    finally:
        if prev is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = prev


# --------------------------------------------------------------------------- #
# the ensure flow
# --------------------------------------------------------------------------- #
def _result(state: str, pin: dict, sha: str | None, *, wiped: bool, error: str | None) -> dict:
    return {"state": state, "model": pin["repo"], "revision": pin["revision"],
            "artifact": pin["artifact"], "sha256": sha, "expected_sha256": pin["sha256"],
            "bytes": pin["bytes"], "wiped": wiped, "error": error}


def ensure(notify=None, *, pin: dict = EMBED_MODEL_PIN, root: str | None = None,
           loader=None, poll_interval: float = 0.5, log=None) -> dict:
    """Eager, checksum-verified provisioning (PROTOCOL §7.7).

    1. Cached artifact whose sha matches the pin → verify-only fast path (no download).
    2. Else download via the normal embedder path while emitting `model.progress`.
    3. Re-hash (streaming) and compare to the pin. Mismatch / missing-after-download → WIPE the
       cached model dir and report `state:"failed", wiped:true, error:…` IN THE RESULT (not an
       error envelope) so the UI can show got-vs-expected sha.
    A hard download/I-O failure (could not complete at all) raises ModelUnavailable (-32009).

    `notify(params)` receives the §7.7 model.progress payloads; `loader(pin)` (injectable for
    tests) performs the blocking download. Runs synchronously — the caller (serial daemon loop)
    is blocked while it downloads."""
    notify = notify or (lambda params: None)
    log = log or (lambda level, msg: None)
    root = root or cache_root()
    total = pin["bytes"]

    # 1. fast path — cached + verified.
    art = artifact_path(root, pin)
    if art is not None:
        sha = sha256_file(art)
        if sha == pin["sha256"]:
            notify({"state": "ready", "received_bytes": total, "total_bytes": total})
            return _result("ready", pin, sha, wiped=False, error=None)

    # 2. download, polling the cache-dir size for determinate progress.
    mdir = _model_dir(root, pin)
    notify({"state": "downloading", "received_bytes": dir_size(mdir), "total_bytes": total})
    stop = threading.Event()

    def _poll() -> None:
        while not stop.wait(poll_interval):
            notify({"state": "downloading", "received_bytes": dir_size(mdir),
                    "total_bytes": total})

    poller = threading.Thread(target=_poll, name="model-progress", daemon=True)
    poller.start()
    try:
        (loader or _default_loader)(pin)
    except Exception as e:  # noqa: BLE001 — any load failure is a hard download failure
        stop.set()
        poller.join(timeout=2)
        log("error", f"model.ensure download failed: {e!r}")
        notify({"state": "failed", "received_bytes": dir_size(mdir), "total_bytes": total,
                "error": str(e) or e.__class__.__name__})
        raise ModelUnavailable(f"embedding-model download failed: {e}")
    finally:
        stop.set()
        poller.join(timeout=2)

    # 3. verify (re-hash) — checksum gate rides the RESULT, never an error envelope.
    notify({"state": "verifying", "received_bytes": total, "total_bytes": total})
    art = artifact_path(root, pin)
    if art is None:
        wipe(root, pin)
        err = "embedding model missing after download"
        notify({"state": "failed", "received_bytes": dir_size(mdir), "total_bytes": total,
                "error": err})
        return _result("failed", pin, None, wiped=True, error=err)
    sha = sha256_file(art)
    if sha != pin["sha256"]:
        wipe(root, pin)
        err = (f"embedding model failed checksum verification "
               f"(got {sha[:12]}…, expected {pin['sha256'][:12]}…); the cached copy was removed")
        notify({"state": "failed", "received_bytes": dir_size(mdir), "total_bytes": total,
                "error": err})
        return _result("failed", pin, sha, wiped=True, error=err)
    notify({"state": "ready", "received_bytes": total, "total_bytes": total})
    return _result("ready", pin, sha, wiped=False, error=None)
