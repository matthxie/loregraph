"""Local pass: GLiNER-Relex (knowledgator/gliner-relex-large-v1.0) joint NER+RE on the
same 20 sessions, GPU, timed. Writes spikes/gliner/local_extractions.json.

Mirrors the product's label conventions:
  - entity labels = natural-word set from kg/nlp_extractors.GLINER_LABELS, remapped to
    the 8 EntityType values via _GLINER_MAP (read-only import).
  - relation labels = the 30-predicate schema keys from GLINER2_REL_SCHEMA (the
    vocabulary the product already tuned for this corpus).
Chunking: strip chat noise (same regexes as the product), split on paragraphs, greedy
pack to ~140 words per chunk (DeBERTa 512-token window safety).
"""
import json, os, re, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from kg.nlp_extractors import GLINER_LABELS, _GLINER_MAP, GLINER2_REL_SCHEMA, _clean

MODEL = "knowledgator/gliner-relex-large-v1.0"
ENT_THR = 0.45
REL_THR = 0.45
REL_LABELS = list(GLINER2_REL_SCHEMA.keys())
MAX_WORDS = 140
DROP = {"user", "assistant", "human", "ai", "you", "i", "me", "my", "we", "us",
        "it", "they", "he", "she", "today", "tomorrow", "yesterday"}
# first-person surfaces kept for RELATIONS (normalized to 'me') but dropped as entities
FIRST_PERSON = {"i", "me", "my", "we", "us", "myself", "i'm", "i've", "i'd", "i'll",
                "my sister", "my brother", "my wife", "my husband"}


def chunks(text: str, max_words: int = MAX_WORDS):
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    cur, n = [], 0
    for p in paras:
        w = len(p.split())
        if w > max_words:                       # oversize paragraph: hard word split
            if cur:
                yield " ".join(cur); cur, n = [], 0
            ws = p.split()
            for i in range(0, len(ws), max_words):
                yield " ".join(ws[i:i + max_words])
            continue
        if n + w > max_words and cur:
            yield " ".join(cur); cur, n = [], 0
        cur.append(p); n += w
    if cur:
        yield " ".join(cur)


def norm_head(t: str) -> str:
    return "me" if t.lower().strip() in FIRST_PERSON else t


def main():
    import argparse
    ap = argparse.ArgumentParser()
    global ENT_THR, REL_THR, REL_LABELS
    ap.add_argument("--ent-thr", type=float, default=0.45)
    ap.add_argument("--rel-thr", type=float, default=0.45)
    ap.add_argument("--rel-labels", default=None,
                    help="JSON file with a list of relation labels (default: GLINER2 schema keys)")
    ap.add_argument("--out", default="local_extractions.json")
    args = ap.parse_args()
    ENT_THR, REL_THR = args.ent_thr, args.rel_thr
    if args.rel_labels:
        REL_LABELS = json.load(open(args.rel_labels, encoding="utf-8"))
    print(f"ent_thr={ENT_THR} rel_thr={REL_THR} n_rel_labels={len(REL_LABELS)}")

    sessions = json.load(open(os.path.join(HERE, "sessions.json"), encoding="utf-8"))
    from gliner import GLiNER
    t0 = time.perf_counter()
    m = GLiNER.from_pretrained(MODEL).to("cuda")
    load_s = time.perf_counter() - t0
    print(f"model loaded in {load_s:.1f}s")

    out, infer_total = [], 0.0
    for i, s in enumerate(sessions):
        text = _clean(s["text"])
        cks = list(chunks(text))
        ents: dict[str, dict] = {}
        rels: dict[tuple, dict] = {}
        t1 = time.perf_counter()
        for ck in cks:
            es, rs = m.predict_relations(ck, GLINER_LABELS, REL_LABELS,
                                         threshold=ENT_THR, relation_threshold=REL_THR)
            for e in es:
                name = e["text"].strip()
                key = name.lower()
                if not name or key in DROP or len(name) < 2:
                    continue
                et = _GLINER_MAP.get(e["label"].lower(), "other")
                sc = float(e["score"])
                if key not in ents or sc > ents[key]["score"]:
                    ents[key] = {"name": name, "type": et, "score": sc}
            for r in rs:
                h = norm_head(r["head"]["text"].strip())
                t = norm_head(r["tail"]["text"].strip())
                if not h or not t or h.lower() == t.lower():
                    continue
                if h.lower() in DROP - FIRST_PERSON or t.lower() in DROP - FIRST_PERSON:
                    continue
                k = (h.lower(), r["relation"], t.lower())
                sc = float(r["score"])
                if k not in rels or sc > rels[k]["score"]:
                    rels[k] = {"source": h, "target": t, "label": r["relation"], "score": sc}
        dt = time.perf_counter() - t1
        infer_total += dt
        out.append({"id": s["id"], "chars": s["chars"], "n_chunks": len(cks),
                    "seconds": round(dt, 3),
                    "entities": sorted(ents.values(), key=lambda e: -e["score"]),
                    "relations": sorted(rels.values(), key=lambda r: -r["score"])})
        print(f"[{i+1}/20] {s['id']}: {len(cks)} chunks, {len(ents)} ents, "
              f"{len(rels)} rels, {dt:.2f}s")

    res = {"model": MODEL, "ent_threshold": ENT_THR, "rel_threshold": REL_THR,
           "load_seconds": round(load_s, 1), "infer_seconds": round(infer_total, 2),
           "sessions_per_sec": round(len(sessions) / infer_total, 3), "sessions": out}
    with open(os.path.join(HERE, args.out), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(f"inference total {infer_total:.1f}s -> {len(sessions)/infer_total:.2f} sessions/s")


if __name__ == "__main__":
    main()
