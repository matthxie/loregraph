"""LLM-free (and hybrid) extractors — a cost-reduction experiment (optimization.md).

The live OpenAIExtractor (kg/extractors.py) pulls entities/tags/relations from raw content in
one-or-two OpenAI calls per section — the dominant ingest cost. This module provides drop-in
alternatives that do the same `{entities[], tags[], relations[]}` job with local NLP models at
~$0, plus hybrids that keep the LLM only for the part NLP does worst (relations):

  * gliner_yake          — GLiNER zero-shot NER (8 KG types) + YAKE topical tags, NO relations.
  * gliner_yake_cooccur  — + co-occurrence/verb relations among the GLiNER entities (spaCy deps).
  * gliner_keybert_cooccur — KeyBERT(bge) tags instead of YAKE (semantic, reuses the project embedder).
  * spacy_svo            — classic spaCy NER + YAKE tags + dependency SVO relations (no torch model).
  * keyword_only         — YAKE keywords as both entities (type=concept) and tags (crude lower bound).
  * hybrid_llm_rel       — GLiNER entities + YAKE tags + ONE Claude call for relations-only/section.

All conform to the `Extractor` protocol (extract_text / extract_image / .name / .meter) so the
ingest pipeline is backend-blind. Models are module-level singletons (loaded once, not per
instance) and inference is serialized under a lock (spaCy `nlp()` / GLiNER are not thread-safe,
and the ingest pool fans out). Pure-NLP backends keep an empty meter → $0 in the dashboard; the
hybrid surfaces its single-call cost through the same meter the testrun drains.
"""
from __future__ import annotations

import re
import threading
from typing import Callable

from .config import Config
from .extractors import ExtractedEntity, ExtractedRelation, Extraction
from .metering import UsageMeter
from .models import EntityType, Provenance

# --------------------------------------------------------------------------- #
# Lazy, process-wide model singletons (loaded once; per-instance graphs reuse them)
# --------------------------------------------------------------------------- #
_LOCK = threading.Lock()          # serialize NLP inference (spaCy/GLiNER aren't thread-safe)
_SPACY = None
_GLINER: dict = {}
_GLINER2: dict = {}
_KEYBERT = None


def _spacy():
    global _SPACY
    if _SPACY is None:
        import spacy
        # keep tagger+parser+ner (we need POS/dep for relations + sentences); NER is cheap.
        _SPACY = spacy.load("en_core_web_sm")
    return _SPACY


def _gliner(model: str):
    if model not in _GLINER:
        from gliner import GLiNER
        _GLINER[model] = GLiNER.from_pretrained(model)
    return _GLINER[model]


def _keybert():
    global _KEYBERT
    if _KEYBERT is None:
        from keybert import KeyBERT
        from sentence_transformers import SentenceTransformer
        # reuse the project's embedding model (already cached) — no extra download
        _KEYBERT = KeyBERT(model=SentenceTransformer("BAAI/bge-small-en-v1.5"))
    return _KEYBERT


def _gliner2(model: str):
    """fastino GLiNER2 — one encoder model that does typed entities AND typed relations in a
    single schema-driven forward pass. CPU-first, but moved to MPS (Apple GPU) when available
    for a large speedup on the relation pass. Loaded once per process."""
    if model not in _GLINER2:
        from gliner2 import GLiNER2
        m = GLiNER2.from_pretrained(model)
        try:
            import torch
            if torch.backends.mps.is_available():
                m = m.to("mps")
        except Exception:  # noqa: BLE001 — fall back to CPU if the move fails
            pass
        _GLINER2[model] = m
    return _GLINER2[model]


# --------------------------------------------------------------------------- #
# Text cleaning (chat-format de-noising)
# --------------------------------------------------------------------------- #
_HEADER = re.compile(r"\[chat session[^\]]*\]", re.I)
# role markers at line start ("User:", "Assistant:", "Human:", "AI:") → drop the prefix so the
# NER models don't mint 'User'/'Assistant' as person entities.
_ROLE = re.compile(r"(?im)^\s*(user|assistant|human|ai)\s*:\s*")


def _clean(text: str) -> str:
    t = _HEADER.sub(" ", text or "")
    t = _ROLE.sub("", t)
    return t


# --------------------------------------------------------------------------- #
# Entities — GLiNER zero-shot NER mapped onto the 8 KG EntityTypes
# --------------------------------------------------------------------------- #
# GLiNER emits the label strings we feed it verbatim, so we feed a richer, more natural set and
# remap down to EntityType. (Feeding 'org' directly hurts recall — GLiNER likes natural words.)
GLINER_LABELS = [
    "person", "location", "organization", "company", "product",
    "creative work", "event", "date", "activity", "field of study",
]
_GLINER_MAP = {
    "person": "person",
    "location": "place", "place": "place", "city": "place", "country": "place",
    "organization": "org", "company": "org", "team": "org", "institution": "org",
    "product": "work", "creative work": "work", "work": "work",
    "book": "work", "film": "work", "song": "work", "app": "work",
    "event": "event",
    "date": "date",
    "activity": "concept", "field of study": "concept", "concept": "concept",
    "topic": "concept", "hobby": "concept",
}
# spaCy NER label → EntityType (for the spacy_svo variant)
_SPACY_MAP = {
    "PERSON": "person", "NORP": "org", "FAC": "place", "ORG": "org",
    "GPE": "place", "LOC": "place", "PRODUCT": "work", "EVENT": "event",
    "WORK_OF_ART": "work", "LAW": "concept", "LANGUAGE": "concept",
    "DATE": "date", "TIME": "date",
}
# chat / pronoun noise to never emit as entities
_DROP_ENT = {"user", "assistant", "human", "ai", "you", "i", "me", "my", "we", "us",
             "it", "they", "he", "she", "today", "tomorrow", "yesterday"}


def _sentence_chunks(doc, max_words: int = 160):
    """Group sentences into ~max_words windows so each GLiNER call stays inside its short
    token window (GLiNER degrades / truncates past ~384 tokens). spaCy already segmented."""
    cur, n = [], 0
    for sent in doc.sents:
        w = len(sent.text.split())
        if n + w > max_words and cur:
            yield " ".join(cur)
            cur, n = [], 0
        cur.append(sent.text)
        n += w
    if cur:
        yield " ".join(cur)


def _dedup_entities(pairs: list[tuple[str, str, float]]) -> list[ExtractedEntity]:
    """pairs = (name, etype_value, score); keep the highest-scoring type per normalized name."""
    best: dict[str, tuple[str, str, float]] = {}
    for name, etype, score in pairs:
        name = name.strip()
        key = name.lower()
        if not name or key in _DROP_ENT or len(name) < 2:
            continue
        if key not in best or score > best[key][2]:
            best[key] = (name, etype, score)
    out = []
    for name, etype, _ in best.values():
        try:
            out.append(ExtractedEntity(name=name, type=EntityType(etype)))
        except ValueError:
            out.append(ExtractedEntity(name=name, type=EntityType.OTHER))
    return out


def gliner_entities(doc, model: str, threshold: float) -> list[ExtractedEntity]:
    m = _gliner(model)
    pairs: list[tuple[str, str, float]] = []
    for chunk in _sentence_chunks(doc):
        for e in m.predict_entities(chunk, GLINER_LABELS, threshold=threshold):
            etype = _GLINER_MAP.get(e["label"].lower(), "other")
            pairs.append((e["text"], etype, float(e.get("score", 1.0))))
    return _dedup_entities(pairs)


def spacy_entities(doc) -> list[ExtractedEntity]:
    pairs = [(e.text, _SPACY_MAP.get(e.label_, "other"), 1.0) for e in doc.ents]
    return _dedup_entities(pairs)


# --------------------------------------------------------------------------- #
# Tags — topical keywords (YAKE statistical / KeyBERT semantic over bge)
# --------------------------------------------------------------------------- #
_WEEKDAYS = {"mon", "tue", "wed", "thu", "fri", "sat", "sun", "monday", "tuesday",
             "wednesday", "thursday", "friday", "saturday", "sunday"}
_TAG_STOP = {"chat session", "user", "assistant", "make", "get", "thing", "things",
             "lot", "time", "way", "ways", "day", "days", "today", "good", "great",
             "really", "sure", "help", "tips"} | _WEEKDAYS


def _clean_tags(raw: list[str], top: int) -> list[str]:
    out, seen = [], set()
    for k in raw:
        k = k.strip().lower()
        if not k or k in seen or k in _TAG_STOP or len(k) < 3 or k.isdigit():
            continue
        seen.add(k)
        out.append(k)
        if len(out) >= top:
            break
    return out


def yake_tags(text: str, doc=None, top: int = 12) -> list[str]:
    import yake
    kw = yake.KeywordExtractor(lan="en", n=2, dedupLim=0.7, top=top * 3)
    return _clean_tags([k for k, _ in kw.extract_keywords(text)], top)


def keybert_tags(text: str, doc=None, top: int = 12) -> list[str]:
    kb = _keybert()
    raw = kb.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english",
                              use_mmr=True, diversity=0.6, top_n=top * 3)
    return _clean_tags([k for k, _ in raw], top)


def nounchunk_tags(text: str, doc=None, top: int = 10) -> list[str]:
    """spaCy noun-chunk tags, ranked by in-doc frequency with a multi-word boost (research's
    recommended tagger: keeps salient compounds like 'soup kitchen' / 'bible study group' that
    drive the IDF-weighted SHARED_TAG bridges, where YAKE leaks verb fragments and KeyBERT
    collapses to generic unigrams). Reuses the doc already parsed by the extractor — corpus-IDF
    is applied downstream (canonicalize.idf_weight), so the extractor only needs good candidates."""
    from collections import Counter
    d = doc if doc is not None else _spacy()(text[:100000])
    freq: Counter = Counter()
    for chunk in d.noun_chunks:
        toks = [t.text.lower() for t in chunk
                if t.pos_ in ("NOUN", "PROPN", "ADJ") and not t.is_stop and not t.is_punct]
        if not toks or len(toks) > 3:
            continue
        ph = " ".join(toks)
        if len(ph) >= 3:
            freq[ph] += 1
    scored = sorted(freq.items(), key=lambda kv: -(kv[1] * (1.3 if " " in kv[0] else 1.0)))
    return _clean_tags([p for p, _ in scored], top)


# --------------------------------------------------------------------------- #
# Relations — LLM-free (co-occurrence + connecting verb; spaCy SVO)
# --------------------------------------------------------------------------- #
_END_CUE = re.compile(r"\b(former|formerly|ex|no longer|used to|left|until|quit|stopped)\b", re.I)
_REL_STOP = {"be", "have", "do", "say", "get", "go", "make", "see", "know", "think",
             "want", "tell", "ask", "give", "take", "come", "use", "feel", "seem"}


def _entity_surfaces(entities: list[ExtractedEntity]) -> dict[str, str]:
    return {e.name.lower(): e.name for e in entities}


def cooccur_relations(doc, entities: list[ExtractedEntity],
                      max_rels: int = 40) -> list[ExtractedRelation]:
    """Relate two extracted entities that co-occur in one sentence, labelling the edge with the
    sentence's salient verb lemma (ROOT verb preferred). Crude but free; captures 'who did what
    with whom' that drives the FACTS list. Termination cues (former/ex/left/until) → status ended."""
    surf = _entity_surfaces(entities)
    if len(surf) < 2:
        return []
    rels: list[ExtractedRelation] = []
    seen: set[tuple[str, str]] = set()
    for sent in doc.sents:
        low = sent.text.lower()
        present = [name for key, name in surf.items() if key in low]
        if len(present) < 2:
            continue
        verbs = [t.lemma_.lower() for t in sent if t.pos_ == "VERB"
                 and t.lemma_.lower() not in _REL_STOP]
        root = next((t.lemma_.lower() for t in sent if t.pos_ == "VERB" and t.dep_ == "ROOT"
                     and t.lemma_.lower() not in _REL_STOP), None)
        label = root or (verbs[0] if verbs else "related to")
        label = label.replace(" ", "_")
        ended = bool(_END_CUE.search(low))
        # connect consecutive distinct entities in surface order (cheap, avoids n^2 blow-up)
        ordered = sorted(set(present), key=lambda n: low.find(n.lower()))
        for a, b in zip(ordered, ordered[1:]):
            key = (a.lower(), b.lower())
            if a.lower() == b.lower() or key in seen:
                continue
            seen.add(key)
            rels.append(ExtractedRelation(
                source=a, target=b, labels=[label], provenance=Provenance.EXTRACTED,
                confidence=0.6, status="ended" if ended else "asserted"))
            if len(rels) >= max_rels:
                return rels
    return rels


def svo_relations(doc, entities: list[ExtractedEntity],
                  max_rels: int = 40) -> list[ExtractedRelation]:
    """Dependency SVO triples filtered to extracted entities: subject --verb--> object where
    both subject and object surfaces match a named entity."""
    surf = _entity_surfaces(entities)
    rels: list[ExtractedRelation] = []
    seen: set[tuple[str, str]] = set()

    def match(span_text: str) -> str | None:
        low = span_text.lower()
        for key, name in surf.items():
            if key in low or low in key:
                return name
        return None

    for tok in doc:
        if tok.pos_ != "VERB" or tok.lemma_.lower() in _REL_STOP:
            continue
        subs = [w for w in tok.lefts if w.dep_ in ("nsubj", "nsubjpass")]
        objs = [w for w in tok.rights if w.dep_ in ("dobj", "pobj", "attr", "dative", "obj")]
        for s in subs:
            sm = match(" ".join(t.text for t in s.subtree))
            if not sm:
                continue
            for o in objs:
                om = match(" ".join(t.text for t in o.subtree))
                if not om or om.lower() == sm.lower():
                    continue
                key = (sm.lower(), om.lower())
                if key in seen:
                    continue
                seen.add(key)
                ended = bool(_END_CUE.search(tok.sent.text))
                rels.append(ExtractedRelation(
                    source=sm, target=om, labels=[tok.lemma_.lower().replace(" ", "_")],
                    provenance=Provenance.EXTRACTED, confidence=0.6,
                    status="ended" if ended else "asserted"))
                if len(rels) >= max_rels:
                    return rels
    return rels


# --------------------------------------------------------------------------- #
# Hybrid: one Claude call for relations only, given the NLP-extracted entities
# --------------------------------------------------------------------------- #
_REL_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_relations",
        "description": "Emit the directed relationships among the GIVEN entities.",
        "parameters": {
            "type": "object",
            "properties": {
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "labels": {"type": "array", "items": {"type": "string"},
                                       "description": "1-3 short lowercase predicates read source→target"},
                            "status": {"type": "string", "enum": ["asserted", "ended"]},
                            "valid_from": {"type": "string"},
                            "valid_to": {"type": "string"},
                        },
                        "required": ["source", "target", "labels"],
                    },
                },
            },
            "required": ["relations"],
        },
    },
}

_REL_SYS = (
    "You are given a piece of content and a list of ENTITIES already extracted from it. Emit the "
    "key DIRECTED relationships BETWEEN THOSE ENTITIES only. Use the exact entity strings given. "
    "Each relationship has 1-3 short lowercase labels read source→target (e.g. 'works_with', "
    "'located_in', 'spent_on', 'attended'). If a relationship ended (former/ex/no longer/left/"
    "until X), emit the base predicate with status 'ended'. Set valid_from/valid_to only if a date "
    "is stated. Few or no relations is fine. Call emit_relations exactly once."
)


class _LlmRelations:
    def __init__(self, config: Config):
        import openai
        self.config = config
        self.client = openai.OpenAI()
        self.meter = UsageMeter()

    def __call__(self, text: str, entities: list[ExtractedEntity]) -> list[ExtractedRelation]:
        import json
        if len(entities) < 2:
            return []
        ent_list = ", ".join(e.name for e in entities[:60])
        prompt = (f"ENTITIES: {ent_list}\n\nCONTENT:\n{text[:self.config.extract_max_chars]}")
        try:
            msg = self.client.chat.completions.create(
                model=self.config.llm_model, max_tokens=self.config.extract_max_tokens,
                temperature=0,
                messages=[
                    {"role": "system", "content": _REL_SYS},
                    {"role": "user", "content": prompt},
                ],
                tools=[_REL_TOOL],
                tool_choice={"type": "function", "function": {"name": "emit_relations"}})
            self.meter.record("extract", self.config.llm_model, msg)
        except Exception:  # noqa: BLE001 — keep ingest alive; degrade to no relations
            return []
        names = {e.name.lower() for e in entities}
        out: list[ExtractedRelation] = []
        tc = getattr(msg.choices[0].message, "tool_calls", None) if msg.choices else None
        if not tc or tc[0].function.name != "emit_relations":
            return out
        for r in (json.loads(tc[0].function.arguments) or {}).get("relations", []) or []:
            s, t = (r.get("source") or "").strip(), (r.get("target") or "").strip()
            labels = [str(x).strip() for x in (r.get("labels") or []) if str(x).strip()]
            if not s or not t or not labels:
                continue
            if s.lower() not in names or t.lower() not in names:
                continue  # never relate something NLP didn't name
            out.append(ExtractedRelation(
                source=s, target=t, labels=labels[:3], provenance=Provenance.EXTRACTED,
                confidence=0.8,
                status="ended" if str(r.get("status", "")).lower() == "ended" else "asserted",
                valid_from=str(r.get("valid_from", "") or ""),
                valid_to=str(r.get("valid_to", "") or "")))
        return out


# --------------------------------------------------------------------------- #
# Composable extractor
# --------------------------------------------------------------------------- #
class NlpExtractor:
    """Composes an entity fn, a tag fn, and a relation fn into the Extractor protocol.

    entity_fn(doc) -> [ExtractedEntity];  tag_fn(text) -> [str];
    relation_fn(doc, entities) -> [ExtractedRelation]  (may be a no-op).
    If an `_LlmRelations` step is attached, its meter becomes THIS extractor's meter so the
    testrun's per-document cost drain captures the hybrid's single LLM call."""

    def __init__(self, name: str, config: Config, *,
                 entity_fn: Callable, tag_fn: Callable, relation_fn: Callable,
                 llm_relations: "_LlmRelations | None" = None):
        self.name = name
        self.config = config
        self._entity_fn = entity_fn
        self._tag_fn = tag_fn
        self._relation_fn = relation_fn
        self._llm = llm_relations
        self.meter = llm_relations.meter if llm_relations else UsageMeter()

    def extract_text(self, text: str, title: str = "") -> Extraction:
        body = f"{title}. {text}" if title else text
        clean = _clean(body)
        if not clean.strip():
            return Extraction()
        with _LOCK:                                  # serialize CPU/torch NLP inference
            doc = _spacy()(clean)
            entities = self._entity_fn(doc)
            tags = self._tag_fn(clean, doc)
            rels = self._relation_fn(doc, entities) if self._relation_fn else []
        if self._llm is not None:                    # LLM call OUTSIDE the lock (I/O-bound)
            rels = rels + self._llm(clean, entities)
        return Extraction(entities=entities, tags=tags, relations=rels)

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        # LongMemEval is text-only; NLP backends don't do vision. Fall back to tagging the hint.
        if label_hint:
            return Extraction(tags=yake_tags(label_hint, top=8), description=label_hint)
        return Extraction(description="An image.")


# --------------------------------------------------------------------------- #
# GLiNER2 — one local model: typed entities + typed relations (fastino/gliner2-*)
# --------------------------------------------------------------------------- #
# GLiNER2 emits the label strings we feed it; we feed natural labels and remap to EntityType.
GLINER2_ENT_LABELS = ["person", "location", "organization", "company", "concept",
                      "creative work", "event", "date", "product", "activity"]
_GLINER2_MAP = {
    "person": "person", "location": "place", "organization": "org", "company": "org",
    "concept": "concept", "activity": "concept", "creative work": "work",
    "product": "work", "event": "event", "date": "date",
}
# The relation vocabulary GLiNER2 scores against, AS A SCHEMA WITH DESCRIPTIONS (the richer
# "schema mode" — descriptions disambiguate each predicate and lift zero-shot recall vs bare
# labels). GLiNER2 only emits a relation if a pair fits one of these, so they ARE the edge labels.
GLINER2_REL_SCHEMA = {
    "lives_in": "person lives in a place", "moved_to": "person moved to a place",
    "works_at": "person works at an organization", "works_with": "person collaborates with another person",
    "member_of": "person is a member of a group or organization",
    "attended": "person attended an event, service, school, or institution",
    "visited": "person visited a place or attraction", "traveled_to": "person traveled to a place",
    "spent_on": "person spent money on an item or activity", "bought": "person bought or purchased an item",
    "owns": "person owns or has an item", "has_pet": "person has a pet animal",
    "knows": "person knows another person", "friend_of": "person is a friend of another person",
    "family_of": "person is family of another person", "located_in": "an entity is located in a place",
    "plays": "person plays a sport, game, or instrument", "interested_in": "person is interested in a topic or hobby",
    "studied": "person studied a subject or at a school", "volunteered_at": "person volunteered at an organization or event",
    "donated_to": "person donated to an organization or cause", "participated_in": "person participated in an event or activity",
    "gave_to": "person gave a gift or item to someone", "received": "person received something from someone or somewhere",
    "founded": "person founded an organization", "created": "person created a work or product",
    "manages": "person manages an organization or team", "part_of": "an entity is part of a larger entity",
    "diagnosed_with": "person was diagnosed with a medical condition", "celebrated": "person celebrated an event or occasion",
}
# Pronoun / chat-role / first-person endpoints to DROP entirely as relation endpoints.
# We deliberately do NOT fold first-person (I/me/my/we) into a single 'me' narrator node:
# every LongMemEval session is already narrated from the user's own perspective, so a 'me'
# node carries no new information yet hurts retrieval — every one of the user's facts collapses
# onto one over-connected super-hub, and Personalized-PageRank then walks *through* it, washing
# out the specific entities a query actually seeds on. So an I/me/my/we edge is dropped, not
# re-rooted; its concrete object (e.g. "coffee mugs") still survives as its own entity node.
_REL_DROP = {"user", "assistant", "human", "ai", "you", "it", "they", "he", "she", "one",
             "i", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves",
             "ourself", "i'm", "i've", "i'd", "i'll"}


def _word_chunks(text: str, n: int = 110):
    """GLiNER2's DeBERTa encoder has a ~512-token window; feed ~110-word windows so a long section
    never gets truncated to its head. Smaller windows also keep relation RECALL up (GLiNER2 only
    relates entity pairs co-located in the same chunk). The chunks of one section are now fed to
    GLiNER2 in a single *batched* forward pass (see Gliner2Extractor._gliner2), so the small
    window no longer costs serial latency."""
    w = text.split()
    for i in range(0, len(w), n):
        yield " ".join(w[i:i + n])


def _safe_etype(val: str) -> EntityType:
    try:
        return EntityType(val)
    except ValueError:
        return EntityType.OTHER


class Gliner2Extractor:
    """TYPED entities + TYPED relations from local GLiNER2 (no LLM, CPU/MPS-first, $0). For each
    section the word-chunks are run through GLiNER2 in TWO BATCHED forward passes — one over all
    chunks for entities, then ONE over only the chunks with >=2 distinct entities for relations.
    Post-filters: drop self-loops, drop chat-role/first-person pronouns, keep only relations whose
    endpoints are real entities (adding any missing endpoint as an OTHER entity so the edge
    survives). First-person endpoints are dropped, NOT folded into a 'me' node (see _REL_DROP).
    Tags come from the chosen keyword tagger (GLiNER2 doesn't emit topical tags)."""

    def __init__(self, config: Config, name: str = "gliner2", tag_fn=yake_tags):
        self.config = config
        self.name = name
        self.meter = UsageMeter()                 # always empty → $0 in the dashboard
        self._tag_fn = tag_fn
        self.model_id = getattr(config, "gliner2_model", "fastino/gliner2-large-v1")
        self.ent_thr = getattr(config, "gliner2_entity_threshold", 0.5)
        self.rel_thr = getattr(config, "gliner2_relation_threshold", 0.5)

    def extract_text(self, text: str, title: str = "") -> Extraction:
        body = f"{title}. {text}" if title else text
        clean = _clean(body)
        if not clean.strip():
            return Extraction()
        with _LOCK:
            model = _gliner2(self.model_id)
            ents, rels = self._gliner2(clean, model)
            tags = self._tag_fn(clean)             # yake ignores doc; nounchunk self-parses
        return Extraction(entities=ents, tags=tags, relations=rels)

    def _gliner2(self, text: str, model):
        """Two batched GLiNER2 passes per section (was: 2 serial calls per chunk).

        Speed wins (optimization.md Lever 7):
          1. BATCHING — feed every word-chunk to GLiNER2 in one `batch_extract_*` call instead
             of looping chunk-by-chunk, so the encoder runs them as a single GPU batch.
          2. COMBINED PASS — entities and relations are each one consolidated batched call for
             the whole section, replacing the old per-chunk extract_entities+extract_relations.
          3. SKIP <2-ENTITY CHUNKS — a chunk with fewer than two distinct (post-filter) entities
             cannot host a relation between two distinct entities (it only ever yielded dropped
             self-loops), so it is excluded from the relation pass entirely. When no chunk
             qualifies, the whole relation forward pass is skipped.
             Tradeoff: the relation head can occasionally surface an endpoint the entity head
             missed, so gating on the entity count can drop a rare edge whose two endpoints both
             came from the relation pass. This is the accepted speed/recall tradeoff of the gate
             (validated on the sample tier); raise/lower the >=2 threshold to trade speed for recall.
        """
        chunks = list(_word_chunks(text))
        if not chunks:
            return [], []
        names: dict[str, tuple[str, str]] = {}     # name.lower() -> (surface, etype value)
        per_chunk_ents: list[int] = []             # distinct-entity count per chunk (skip gate)
        # (1)+(2) one batched entity pass over all chunks
        for er in model.batch_extract_entities(chunks, GLINER2_ENT_LABELS, threshold=self.ent_thr):
            local: set[str] = set()
            for lbl, nms in (er.get("entities") or {}).items():
                et = _GLINER2_MAP.get(lbl.lower(), "other")
                for nm in (nms or []):
                    nm = (nm if isinstance(nm, str) else nm.get("text", "")).strip()
                    if nm and nm.lower() not in _DROP_ENT and len(nm) >= 2:
                        names.setdefault(nm.lower(), (nm, et))
                        local.add(nm.lower())
            per_chunk_ents.append(len(local))
        # (3) only chunks with >=2 distinct entities can host a relation
        rel_chunks = [chunks[i] for i, c in enumerate(per_chunk_ents) if c >= 2]
        rels: list[ExtractedRelation] = []
        if rel_chunks:
            seen: set[tuple] = set()
            # (1)+(2) one batched relation pass over the surviving chunks
            for rr in model.batch_extract_relations(rel_chunks, GLINER2_REL_SCHEMA,
                                                    threshold=self.rel_thr):
                for rtype, pairs in (rr.get("relation_extraction") or {}).items():
                    for p in (pairs or []):
                        if not (isinstance(p, (list, tuple)) and len(p) >= 2):
                            continue
                        h = str(p[0]).strip()
                        t = str(p[1]).strip()
                        if not h or not t or h.lower() == t.lower():           # self-loop filter
                            continue
                        if h.lower() in _REL_DROP or t.lower() in _REL_DROP:   # pronoun/role/1st-person
                            continue
                        key = (h.lower(), rtype, t.lower())
                        if key in seen:
                            continue
                        seen.add(key)
                        names.setdefault(h.lower(), (h, "other"))
                        names.setdefault(t.lower(), (t, "other"))
                        rels.append(ExtractedRelation(
                            source=names[h.lower()][0], target=names[t.lower()][0],
                            labels=[rtype], provenance=Provenance.EXTRACTED, confidence=0.7))
        ents = [ExtractedEntity(name=s, type=_safe_etype(et)) for s, et in names.values()]
        return ents, rels

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return Extraction(description=label_hint or "An image.")


# --------------------------------------------------------------------------- #
# Combination: GLiNER2 (typed, $0) UNION Haiku (clean open-vocab relations)
# --------------------------------------------------------------------------- #
class Gliner2HaikuExtractor:
    """Run BOTH the tuned GLiNER2 extractor and the Haiku LLM extractor on each section and MERGE
    their Extractions (union entities/tags, union relation labels per directed pair). The point:
    keep Haiku's clean open-vocabulary third-party relations AND add GLiNER2's extra typed entities
    and relations that Haiku's default prompt misses. Cost == Haiku (GLiNER2 is free); the question
    the eval answers is whether the GLiNER2 bonus lifts answer accuracy over Haiku alone. The meter
    is Haiku's, so the dashboard shows the (Haiku) cost."""

    def __init__(self, config: Config, name: str = "gliner2_haiku"):
        from .extractors import OpenAIExtractor
        self.config = config
        self.name = name
        self._g2 = Gliner2Extractor(config, name="gliner2", tag_fn=yake_tags)
        self._haiku = OpenAIExtractor(config)
        self.meter = self._haiku.meter          # cost attribution flows through Haiku's meter

    def extract_text(self, text: str, title: str = "") -> Extraction:
        a = self._g2.extract_text(text, title)
        b = self._haiku.extract_text(text, title)
        return a.merge(b)                        # union of both extractions

    def extract_image(self, image_path: str, label_hint: str | None = None) -> Extraction:
        return self._haiku.extract_image(image_path, label_hint)


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def _keyword_entities_and_tags(text: str, top: int = 14):
    """keyword_only lower-bound: YAKE keyphrases used as BOTH concept entities and tags."""
    tags = yake_tags(text, top=top)
    ents = _dedup_entities([(t, "concept", 1.0) for t in tags])
    return ents, tags


def build_nlp_extractor(backend: str, config: Config):
    model = getattr(config, "gliner_model", "urchade/gliner_small-v2.1")
    thr = getattr(config, "gliner_threshold", 0.5)
    no_rel = lambda doc, ents: []

    if backend == "gliner_yake":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=yake_tags, relation_fn=no_rel)
    if backend == "gliner_nounchunk":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=nounchunk_tags, relation_fn=no_rel)
    if backend == "gliner_nounchunk_cooccur":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=nounchunk_tags, relation_fn=cooccur_relations)
    if backend == "gliner_yake_cooccur":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=yake_tags, relation_fn=cooccur_relations)
    if backend == "gliner_keybert_cooccur":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=keybert_tags, relation_fn=cooccur_relations)
    if backend == "spacy_svo":
        return NlpExtractor(backend, config,
                            entity_fn=spacy_entities, tag_fn=yake_tags,
                            relation_fn=svo_relations)
    if backend == "hybrid_llm_rel":
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=yake_tags, relation_fn=no_rel,
                            llm_relations=_LlmRelations(config))
    if backend == "hybrid_nounchunk_rel":      # research ship-candidate (Design A, noun-chunk tags)
        return NlpExtractor(backend, config,
                            entity_fn=lambda doc: gliner_entities(doc, model, thr),
                            tag_fn=nounchunk_tags, relation_fn=no_rel,
                            llm_relations=_LlmRelations(config))
    if backend == "keyword_only":
        def ent_fn(doc):
            return _keyword_entities_and_tags(doc.text)[0]
        return NlpExtractor(backend, config, entity_fn=ent_fn, tag_fn=yake_tags,
                            relation_fn=no_rel)
    if backend == "gliner2":                   # typed entities + typed relations, YAKE tags
        return Gliner2Extractor(config, name="gliner2", tag_fn=yake_tags)
    if backend == "gliner2_nounchunk":         # same, with noun-chunk tags
        return Gliner2Extractor(config, name="gliner2_nounchunk", tag_fn=nounchunk_tags)
    if backend == "gliner2_haiku":              # GLiNER2 (me-facts, $0) UNION Haiku (open-vocab)
        return Gliner2HaikuExtractor(config, name="gliner2_haiku")
    raise ValueError(f"unknown NLP extractor backend: {backend!r}")


NLP_BACKENDS = {"gliner_yake", "gliner_nounchunk", "gliner_nounchunk_cooccur",
                "gliner_yake_cooccur", "gliner_keybert_cooccur", "spacy_svo",
                "hybrid_llm_rel", "hybrid_nounchunk_rel", "keyword_only",
                "gliner2", "gliner2_nounchunk", "gliner2_haiku"}
