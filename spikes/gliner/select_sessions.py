"""Pick 20 sessions from the longmemeval sample tier: spread across instances,
mix of lengths. Writes spikes/gliner/sessions.json."""
import json, os, sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from kg.corpus import load_longmemeval

items = load_longmemeval("sample")
print(f"sample tier: {len(items)} sessions")
lens = sorted(len(i.text or "") for i in items)
print("len min/med/max:", lens[0], lens[len(lens)//2], lens[-1])

# spread: take every k-th after sorting by (question_id via source_ref, created_at order kept)
k = max(1, len(items) // 20)
picked = items[::k][:20]
out = [{"id": i.id, "source_ref": i.source_ref, "created_at": i.created_at,
        "chars": len(i.text or ""), "text": i.text} for i in picked]
dst = os.path.join(os.path.dirname(__file__), "sessions.json")
with open(dst, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=1)
print(f"picked {len(out)} sessions -> {dst}")
for s in out:
    print(" ", s["id"], s["chars"], "chars", s["created_at"], s["source_ref"])
