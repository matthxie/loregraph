"""Regression tests for Finding 10 — a checksum-failed model must stop serving embeddings
in-process (kg.modelpin wipe-on-mismatch must evict kg.embedders._MODEL_CACHE), and the
pinned on-disk artifact must be hash-verified BEFORE any loader/warmup runs (verify-then-load).
"""
from __future__ import annotations

import hashlib
import os

import pytest

from kg import embedders, modelpin

GOOD = b"good-model-weights-payload"
GOOD_SHA = hashlib.sha256(GOOD).hexdigest()
CORRUPT = b"tampered-weights"


def make_pin(sha: str = GOOD_SHA) -> dict:
    return {
        "repo": "test/model",
        "cache_component": "models--test--model",
        "revision": "rev123",
        "artifact": "model.safetensors",
        "sha256": sha,
        "bytes": len(GOOD),
    }


def write_artifact(root: str, pin: dict, content: bytes) -> None:
    d = os.path.join(root, pin["cache_component"], "snapshots", pin["revision"])
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, pin["artifact"]), "wb") as f:
        f.write(content)


@pytest.fixture(autouse=True)
def _clean_cache():
    """Isolate the process-global model cache per test."""
    saved = dict(embedders._MODEL_CACHE)
    embedders._MODEL_CACHE.clear()
    try:
        yield
    finally:
        embedders._MODEL_CACHE.clear()
        embedders._MODEL_CACHE.update(saved)


# --------------------------------------------------------------------------- #
# the eviction helper (embedders)
# --------------------------------------------------------------------------- #
def test_evict_cached_model_pops_and_is_noop_when_absent():
    embedders._MODEL_CACHE["test/model"] = object()
    embedders.evict_cached_model("test/model")
    assert "test/model" not in embedders._MODEL_CACHE
    # absent key must not raise
    embedders.evict_cached_model("never/loaded")


def test_wipe_evicts_in_process_cache_and_removes_dir(tmp_path):
    root = str(tmp_path)
    pin = make_pin()
    write_artifact(root, pin, GOOD)
    embedders._MODEL_CACHE[pin["repo"]] = object()   # a warmed, now-suspect model

    modelpin.wipe(root, pin)

    assert not os.path.exists(os.path.join(root, pin["cache_component"]))
    assert pin["repo"] not in embedders._MODEL_CACHE


# --------------------------------------------------------------------------- #
# ensure() — verify-then-load (part a) and evict-on-failure (part b)
# --------------------------------------------------------------------------- #
def test_ensure_fast_path_matches_without_running_loader(tmp_path):
    root = str(tmp_path)
    pin = make_pin()
    write_artifact(root, pin, GOOD)

    def loader(_pin):
        raise AssertionError("loader must not run when the cached artifact already verifies")

    result = modelpin.ensure(pin=pin, root=root, loader=loader)
    assert result["state"] == "ready"
    assert result["sha256"] == GOOD_SHA
    assert result["wiped"] is False


def test_ensure_ondisk_mismatch_verifies_before_loader(tmp_path):
    """A corrupt on-disk copy is wiped+evicted BEFORE the loader/warmup runs, so no unverified
    on-disk bytes are ever warmed into the process cache."""
    root = str(tmp_path)
    pin = make_pin()
    write_artifact(root, pin, CORRUPT)               # on disk but does NOT match the pin
    embedders._MODEL_CACHE[pin["repo"]] = object()   # stale warmed entry from a prior run

    seen = {}

    def loader(_pin):
        # By the time the loader runs, the corrupt copy's cache entry must already be gone.
        seen["cache_had_repo_at_call"] = pin["repo"] in embedders._MODEL_CACHE
        write_artifact(root, pin, GOOD)              # simulate a clean re-download

    result = modelpin.ensure(pin=pin, root=root, loader=loader)

    assert seen["cache_had_repo_at_call"] is False   # evicted before loader ran
    assert result["state"] == "ready"
    assert result["sha256"] == GOOD_SHA


def test_ensure_download_mismatch_wipes_and_evicts_cache(tmp_path):
    """The finding's scenario: the download's warmup populates _MODEL_CACHE, then the checksum
    fails — ensure() must wipe the dir AND evict the in-process model."""
    root = str(tmp_path)
    pin = make_pin()

    def loader(_pin):
        write_artifact(root, pin, CORRUPT)           # download lands corrupt bytes
        embedders._MODEL_CACHE[pin["repo"]] = object()   # warmup embed caches the model

    result = modelpin.ensure(pin=pin, root=root, loader=loader)

    assert result["state"] == "failed"
    assert result["wiped"] is True
    assert pin["repo"] not in embedders._MODEL_CACHE     # no longer serving in-process
    assert not os.path.exists(os.path.join(root, pin["cache_component"]))


def test_ensure_missing_after_download_wipes_and_evicts(tmp_path):
    root = str(tmp_path)
    pin = make_pin()

    def loader(_pin):
        embedders._MODEL_CACHE[pin["repo"]] = object()   # warmed but nothing lands on disk

    result = modelpin.ensure(pin=pin, root=root, loader=loader)

    assert result["state"] == "failed"
    assert result["wiped"] is True
    assert pin["repo"] not in embedders._MODEL_CACHE
