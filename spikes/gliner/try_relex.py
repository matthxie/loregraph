"""Probe: can the installed gliner load knowledgator/gliner-relex-large-v1.0 and do
joint NER+RE? Prints library version, load result, and a smoke-test extraction."""
import sys, traceback

import gliner
print("gliner version:", gliner.__version__)
import torch
print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

MODEL = "knowledgator/gliner-relex-large-v1.0"
try:
    from gliner import GLiNER
    m = GLiNER.from_pretrained(MODEL)
    print("LOADED with GLiNER.from_pretrained; class:", type(m).__name__)
    print("has predict_relations:", hasattr(m, "predict_relations"))
    print("has predict_entities:", hasattr(m, "predict_entities"))
    m = m.to("cuda")
    text = ("I moved from Boston to Seattle last month and started a new job at Stripe. "
            "My sister Emma still lives in Boston with her dog Baxter.")
    labels = ["person", "location", "organization", "pet"]
    ents = m.predict_entities(text, labels, threshold=0.4)
    for e in ents:
        print("ENT", e)
    # try relation API variants
    for attr in ("predict_relations", "extract_relations", "predict_entities_and_relations",
                 "inference", "predict"):
        if hasattr(m, attr):
            print("HAS ATTR:", attr)
except Exception:
    traceback.print_exc()
    sys.exit(1)
