"""Offline probe for chunk-level retargeting (rag_retarget / rag_provenance_promote).

Verifies the three failures the retargeting fix targets — the right SOURCE session wins
seats in _select_episodes but the decisive CHUNK of it is missing from context:

  25e5aa4f  "Where did I complete my Bachelor's degree in Computer Science?"  -> needs "UCLA"
  37f165cf  "What was the page count of the two novels I finished ...?"      -> needs "416 pages"
  099778bb  "What percentage of leadership positions do women hold ...?"    -> needs the
            "women occupy 20 of the leadership positions" chunk

No LLM calls: opens the cached per-instance store (store/cache/<qid>-*.db), runs the same
HybridRetriever + ContextBuilder path `ContextBuilder.build()` drives inside OpenAIAnswerer,
and prints the before (rag_retarget=off) / after (seed+lex + provenance_promote) context
chunk ids and whether the needle text now appears in the context blob.

Run: python spikes/retarget/probe.py   (needs the venv's sentence-transformers install:
.venv/Scripts/python.exe spikes/retarget/probe.py)
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kg.config import Config
from kg.graph import KnowledgeGraph
from kg.rag import ContextBuilder
from kg.retrieval import HybridRetriever

CASES = [
    dict(qid="25e5aa4f", cache="store/cache/25e5aa4f-a1e3a79ccfac.db",
        query="Where did I complete my Bachelor's degree in Computer Science?",
        as_of="2023-05-30T15:02:00+00:00", kind="single-session-user",
        needles=["UCLA"]),
    dict(qid="37f165cf", cache="store/cache/37f165cf-2e28930f3287.db",
        query="What was the page count of the two novels I finished in January and March?",
        as_of="2023-05-30T19:21:00+00:00", kind="multi-session",
        needles=["440 pages", "416-page", "416 pages"]),
    dict(qid="099778bb", cache="store/cache/099778bb-eb319bf41f76.db",
        query="What percentage of leadership positions do women hold in the my company?",
        as_of="2023-05-30T22:26:00+00:00", kind="multi-session",
        needles=["women occupy 20", "20%"]),
]


def _open(cache_path: str, cfg: Config) -> KnowledgeGraph:
    tmp = tempfile.mktemp(suffix=".db")
    shutil.copy(cache_path, tmp)
    return KnowledgeGraph.open(tmp, cfg)


def _run(case: dict, cfg: Config) -> tuple[list[str], str]:
    g = _open(case["cache"], cfg)
    retriever = HybridRetriever(g.store, g.embedder, g.canon, cfg)
    builder = ContextBuilder(g.store, cfg)
    result = retriever.retrieve(case["query"], k=cfg.top_k, as_of=case["as_of"],
                                kind=case["kind"])
    ep_ids, _facts, blob = builder.build(result)
    return ep_ids, blob


def main() -> None:
    base = Config.default()
    base.rag_parent_expand = 2
    base.rag_chunks_per_source = 2

    off = replace(base, rag_retarget="off", rag_provenance_promote=False)
    on = replace(base, rag_retarget="seed+lex", rag_provenance_promote=True)

    for case in CASES:
        print(f"\n==== {case['qid']} — {case['query']!r} ====")
        ep_off, blob_off = _run(case, off)
        ep_on, blob_on = _run(case, on)

        print("before (off) context chunks:")
        for c in ep_off:
            print(f"  {c}")
        print("after (seed+lex+provenance) context chunks:")
        for c in ep_on:
            print(f"  {c}")

        for needle in case["needles"]:
            before = needle.lower() in blob_off.lower()
            after = needle.lower() in blob_on.lower()
            flag = "FIXED" if (after and not before) else ("no change" if before == after
                                                            else "REGRESSED")
            print(f"  needle {needle!r}: before={before} after={after}  [{flag}]")


if __name__ == "__main__":
    main()
