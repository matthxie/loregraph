"""Ground truth: run the production LLM extractor (OpenAIExtractor, gpt-4o-mini,
reflexion on, sectioned exactly like ingest) on the 20 picked sessions.

Rate-limit hardened: the org's 10k RPD budget refills ~1 request / 8.64s, so we pace
calls at PACE_S apart, retry 429s patiently, and persist per-session results to
llm/<id>.json so a crash never loses finished work. Final merge -> llm_extractions.json.
"""
import json, os, sys, time, dataclasses

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from kg.config import Config
from kg.extractors import OpenAIExtractor, extract_text_sectioned

PACE_S = 9.5
CACHE = os.path.join(HERE, "llm")
os.makedirs(CACHE, exist_ok=True)

sessions = json.load(open(os.path.join(HERE, "sessions.json"), encoding="utf-8"))
cfg = Config.default()
ex = OpenAIExtractor(cfg)
print("model:", cfg.llm_model, "reflexion:", cfg.reflexion,
      "key set:", bool(os.environ.get("OPENAI_API_KEY")), flush=True)

# pace + patient-429 wrapper around the extractor's single LLM call site
_orig_call = ex._call
_last = [0.0]


def paced_call(blocks):
    import openai
    for attempt in range(30):
        wait = PACE_S - (time.perf_counter() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            _last[0] = time.perf_counter()
            return _orig_call(blocks)
        except openai.RateLimitError as e:
            print(f"  429 (attempt {attempt+1}), sleeping 30s", flush=True)
            time.sleep(30)
    raise RuntimeError("gave up after 30 rate-limit retries")


ex._call = paced_call

for i, s in enumerate(sessions):
    dst = os.path.join(CACHE, s["id"] + ".json")
    if os.path.exists(dst):
        print(f"[{i+1}/20] {s['id']}: cached, skip", flush=True)
        continue
    t0 = time.perf_counter()
    ext = extract_text_sectioned(ex, s["text"], long_doc_chars=cfg.long_doc_chars)
    dt = time.perf_counter() - t0
    rec = {
        "id": s["id"], "chars": s["chars"], "seconds": round(dt, 2),
        "entities": [{"name": e.name, "type": e.type.value} for e in ext.entities],
        "tags": ext.tags,
        "relations": [dataclasses.asdict(r) for r in ext.relations],
    }
    json.dump(rec, open(dst, "w", encoding="utf-8"), indent=1, default=str)
    print(f"[{i+1}/20] {s['id']}: {len(ext.entities)} ents, {len(ext.relations)} rels, "
          f"{dt:.1f}s", flush=True)

out = [json.load(open(os.path.join(CACHE, s["id"] + ".json"), encoding="utf-8"))
       for s in sessions]
res = {"usage": ex.meter.totals(), "sessions": out}
json.dump(res, open(os.path.join(HERE, "llm_extractions.json"), "w", encoding="utf-8"),
          indent=1, default=str)
print("DONE. usage:", ex.meter.totals(), flush=True)
