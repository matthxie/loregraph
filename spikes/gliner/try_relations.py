"""Probe predict_relations signature + smoke test on chat-like text."""
import inspect, json
from gliner import GLiNER

m = GLiNER.from_pretrained("knowledgator/gliner-relex-large-v1.0").to("cuda")
print(inspect.signature(m.predict_relations))
try:
    print(inspect.getsource(m.predict_relations)[:3000])
except Exception as e:
    print("no source:", e)

text = ("I moved from Boston to Seattle last month and started a new job at Stripe. "
        "My sister Emma still lives in Boston with her dog Baxter.")
ent_labels = ["person", "location", "organization", "pet"]
rel_labels = ["lives_in", "moved_to", "works_at", "sibling_of", "has_pet", "located_in"]
try:
    out = m.predict_relations(text, ent_labels, rel_labels, threshold=0.4)
    print(json.dumps(out, indent=1, default=str)[:2500])
except TypeError as e:
    print("TypeError:", e)
    # try kwargs variant
    out = m.predict_relations(text, entity_labels=ent_labels, relation_labels=rel_labels)
    print(json.dumps(out, indent=1, default=str)[:2500])
