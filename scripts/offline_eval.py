"""Offline read-side eval (docs/OFFLINE_EVAL.md) — NO paid LLM calls.

Round 2 (event representation). Measures whether the evidence an answer LLM would
need actually reaches KnowledgeGraph.search().context (the exact prompt blob ask()
would read) under the Round-2 configurations:

    baseline    — all knobs off (must reproduce the Round-1 recorded baseline)
    A_amended   — history_all_lanes=True: the closed-fact HISTORY delta on every lane
                  (amended variant A: closed-only lines outside STATE + recency cap)
    EV          — synth store REBUILT with event_facts=True: event predicates write
                  closed [d,d] occurrence edges (read side otherwise baseline)
    EV_A        — the promoted pair: event store + history_all_lanes=True

Stores:
    synth    — a synthetic first-person store built HERE with the ScriptedExtractor
               (no LLM), legacy write semantics (event_facts off).
    synth_ev — the SAME corpus rebuilt with event_facts=True (new write path).
    pilot    — a READ-ONLY COPY of store/events_pilot.db (real ingested chat data,
               extraction cost already sunk). Canary + context-bloat check at scale.

The run FAILS (exit 1) if any probe's hit/coverage under the promoted config drops
below the recorded Round-1 baseline (--baseline runs/offline_eval/results.json).

Run:  python scripts/offline_eval.py [--stores synth,pilot] [--out runs/offline_eval_round2]
Everything runs on the local models (bge-small embedder, ms-marco cross-encoder).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kg.graph as kg_graph                                    # noqa: E402
from kg import Config, KnowledgeGraph                          # noqa: E402
from kg.corpus import CorpusItem                               # noqa: E402
from kg.extractors import (ExtractedEntity, ExtractedRelation,  # noqa: E402
                           Extraction, ScriptedExtractor)
from kg.models import EntityType, NodeType, Provenance         # noqa: E402

# ---- hard no-LLM guard: kg auto-loads .env on import; drop any key it injected and
# make every KnowledgeGraph build a scripted (empty-table) extractor. search() itself
# never calls an LLM; this guarantees ingest paths can't either.
for _k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
    os.environ.pop(_k, None)
kg_graph.get_extractor = lambda config: ScriptedExtractor({})

def _E(name):
    return ExtractedEntity(
        name=name,
        type=EntityType.PERSON if name == "me" else EntityType.CONCEPT)


def _R(src, tgt, label, status="asserted", valid_from="", valid_to="", conf=0.95):
    return ExtractedRelation(source=src, target=tgt, labels=[label],
                             provenance=Provenance.EXTRACTED, confidence=conf,
                             status=status, valid_from=valid_from, valid_to=valid_to)


# --------------------------------------------------------------------------- #
# Synthetic first-person store
# --------------------------------------------------------------------------- #
# (id, created_at(date), text, [entities...], [relations...])   title = id
_SYNTH = [
    # ---- filler profile facts (early; they occupy the front of the unranked
    # facts_for iteration and make the rag_max_facts=30 cap bite on the me-hub) ----
    ("f01", "2019-03-04", "My favorite coffee shop is Cafe Vivace; I go most mornings.",
     ["me", "Cafe Vivace"], [("me", "Cafe Vivace", "frequents")]),
    ("f02", "2019-04-11", "Saw my dentist Dr. Patel for a checkup today.",
     ["me", "Dr. Patel"], [("me", "Dr. Patel", "patient_of")]),
    ("f03", "2019-05-20", "I subscribed to The Economist and to the Radiolab podcast.",
     ["me", "The Economist", "Radiolab"],
     [("me", "The Economist", "subscribes_to"), ("me", "Radiolab", "listens_to")]),
    ("f04", "2019-06-15", "Been practicing guitar in the evenings; also picking up Spanish.",
     ["me", "guitar", "Spanish"],
     [("me", "guitar", "plays"), ("me", "Spanish", "speaks")]),
    ("f05", "2019-07-09", "Joined Flex Fitness, the gym down the street.",
     ["me", "Flex Fitness"], [("me", "Flex Fitness", "member_of")]),
    ("f06", "2019-08-21", "I bank with Chase and use a MacBook for work.",
     ["me", "Chase", "MacBook"],
     [("me", "Chase", "banks_with"), ("me", "MacBook", "uses")]),
    ("f07", "2019-09-30", "Cheering for the Sounders this season as always.",
     ["me", "Sounders"], [("me", "Sounders", "supports")]),
    ("f08", "2019-10-12", "I grew up in Portland and studied at the University of Washington.",
     ["me", "Portland", "University of Washington"],
     [("me", "Portland", "grew_up_in"), ("me", "University of Washington", "studied_at")]),
    ("f09", "2019-11-25", "Bought an espresso machine; also started taking vitamin D.",
     ["me", "espresso machine", "vitamin D"],
     [("me", "espresso machine", "owns"), ("me", "vitamin D", "takes")]),
    ("f10", "2020-01-14", "I mostly read science fiction; also collecting houseplants now.",
     ["me", "science fiction", "houseplants"],
     [("me", "science fiction", "reads"), ("me", "houseplants", "collects")]),

    # ---- employment: open -> explicit "no longer" close -> new employer ----
    ("job1", "2019-02-01", "I work at Acme Corp as a data analyst.",
     ["me", "Acme Corp"], [("me", "Acme Corp", "employed_by")]),
    ("job2", "2022-08-15", "I no longer work at Acme Corp; last Friday was my final day.",
     ["me", "Acme Corp"], [("me", "Acme Corp", "employed_by", "ended")]),
    ("job3", "2022-09-06", "I started a new job at Globex as a product manager on September 6, 2022.",
     ["me", "Globex"], [("me", "Globex", "employed_by", "asserted", "2022-09-06")]),

    # ---- residence: functional supersede (Seattle -> Denver) ----
    ("home1", "2021-06-15", "I live in Seattle, in a small apartment near Green Lake.",
     ["me", "Seattle"], [("me", "Seattle", "lives_in")]),
    ("home2", "2023-05-01", "I moved to Denver on May 1, 2023 and I'm loving the mountain air.",
     ["me", "Denver"], [("me", "Denver", "lives_in", "asserted", "2023-05-01")]),

    # ---- volunteering: "used to X" closure on a non-functional predicate ----
    ("vol1", "2020-03-01", "I volunteer at the animal shelter on Saturdays.",
     ["me", "animal shelter"], [("me", "animal shelter", "volunteers_at")]),
    ("vol2", "2024-01-20", "I used to volunteer at the animal shelter, but I stopped last month.",
     ["me", "animal shelter"], [("me", "animal shelter", "volunteers_at", "ended")]),

    # ---- bounded interval: trip with valid_from AND valid_to ----
    ("trip1", "2023-10-20",
     "I'm going to Japan from November 1st to November 14th, 2023 — Tokyo and Kyoto.",
     ["me", "Japan"],
     [("me", "Japan", "traveled_to", "asserted", "2023-11-01", "2023-11-14")]),

    # ---- repeated undated events: went to the park x5 ----
    ("park1", "2025-01-05", "I went to the park today and walked the loop trail.",
     ["me", "the park"], [("me", "the park", "went_to")]),
    ("park2", "2025-02-02", "Went to the park again this afternoon and fed the ducks.",
     ["me", "the park"], [("me", "the park", "went_to")]),
    ("park3", "2025-03-09", "Spent the morning at the park; the cherry trees are blooming.",
     ["me", "the park"], [("me", "the park", "went_to")]),
    ("park4", "2025-04-13", "Took a long walk in the park after lunch.",
     ["me", "the park"], [("me", "the park", "went_to")]),
    ("park5", "2025-05-18", "Went to the park for a picnic with friends.",
     ["me", "the park"], [("me", "the park", "went_to")]),

    # ---- explicitly-dated occurrences (repeatable, distinct dates) ----
    ("yoga1", "2024-03-10", "I attended a yoga class on March 10, 2024.",
     ["me", "yoga class"], [("me", "yoga class", "attended", "asserted", "2024-03-10")]),
    ("yoga2", "2024-04-14", "Attended another yoga class on April 14, 2024.",
     ["me", "yoga class"], [("me", "yoga class", "attended", "asserted", "2024-04-14")]),
    ("yoga3", "2024-06-02", "Went to a yoga class on June 2, 2024 — tough session.",
     ["me", "yoga class"], [("me", "yoga class", "attended", "asserted", "2024-06-02")]),

    # ---- ordinary single-fact states (canaries) ----
    ("pet1", "2022-04-10", "I adopted a cat named Luna from the shelter.",
     ["me", "Luna"], [("me", "Luna", "has_pet")]),
    ("sis1", "2021-09-12", "My sister Mia lives in Boston with her husband.",
     ["me", "Mia", "Boston"],
     [("me", "Mia", "sibling_of"), ("Mia", "Boston", "lives_in")]),
    ("car1", "2023-07-22", "I bought a Subaru Outback last weekend.",
     ["me", "Subaru Outback"], [("me", "Subaru Outback", "drives")]),
    ("alg1", "2020-11-05", "Reminder for restaurants: I'm allergic to peanuts.",
     ["me", "peanuts"], [("me", "peanuts", "allergic_to")]),
    ("ten1", "2024-02-18", "I play tennis on Tuesday evenings with my friend Raj.",
     ["me", "tennis", "Raj"],
     [("me", "tennis", "plays"), ("me", "Raj", "friend_of")]),
    ("book1", "2023-02-09", "I joined a book club that meets every Thursday.",
     ["me", "book club"], [("me", "book club", "member_of")]),
    ("clmb1", "2024-08-03", "Tried bouldering at the climbing gym; I loved it.",
     ["me", "climbing gym"], [("me", "climbing gym", "went_to")]),
    ("hike1", "2022-06-19", "Hiked Mount Si this weekend — gorgeous views from the top.",
     ["me", "Mount Si"], [("me", "Mount Si", "hiked")]),
]

# More me-facts (push the me-hub well past rag_max_facts=30 so the unranked
# truncation lottery is real) + third-party distractors that lexically shadow the
# canaries (another cat, other parks, other moves/jobs, incl. third-party closures).
_SYNTH += [
    ("g01", "2020-02-08", "Started using a Garmin watch for my runs.",
     ["me", "Garmin watch"], [("me", "Garmin watch", "uses")]),
    ("g02", "2020-04-19", "I donate monthly to the food bank and to public radio.",
     ["me", "food bank", "public radio"],
     [("me", "food bank", "donates_to"), ("me", "public radio", "donates_to")]),
    ("g03", "2020-06-27", "Trying a sourdough starter; also fermenting hot sauce.",
     ["me", "sourdough starter", "hot sauce"],
     [("me", "sourdough starter", "bakes_with"), ("me", "hot sauce", "ferments")]),
    ("g04", "2020-08-14", "I keep a bullet journal and use Obsidian for notes.",
     ["me", "bullet journal", "Obsidian"],
     [("me", "bullet journal", "keeps"), ("me", "Obsidian", "uses_app")]),
    ("g05", "2020-10-03", "Signed up for a pottery class at the community center.",
     ["me", "pottery class"], [("me", "pottery class", "enrolled_in")]),
    ("g06", "2020-12-21", "My go-to takeout is the Thai place on 5th, Bai Tong.",
     ["me", "Bai Tong"], [("me", "Bai Tong", "orders_from")]),
    ("g07", "2021-02-11", "I meditate with the Headspace app most nights.",
     ["me", "Headspace"], [("me", "Headspace", "meditates_with")]),
    ("g08", "2021-04-25", "Picked up film photography; shooting with a Canon AE-1.",
     ["me", "Canon AE-1"], [("me", "Canon AE-1", "shoots_with")]),
    ("g09", "2021-07-08", "I'm on the neighborhood council and the trail cleanup crew.",
     ["me", "neighborhood council", "trail cleanup crew"],
     [("me", "neighborhood council", "serves_on"), ("me", "trail cleanup crew", "serves_on")]),
    ("g10", "2021-10-17", "Bought a standing desk and an ergonomic chair for the office.",
     ["me", "standing desk", "ergonomic chair"],
     [("me", "standing desk", "owns"), ("me", "ergonomic chair", "owns")]),
    ("g11", "2021-12-05", "I brew my own kombucha now, ginger flavor mostly.",
     ["me", "kombucha"], [("me", "kombucha", "brews")]),
    ("g12", "2022-01-29", "Adopted a rescue routine: swimming laps on Friday mornings.",
     ["me", "swimming"], [("me", "swimming", "does")]),
    ("g13", "2022-10-08", "Started learning woodworking; built a small bookshelf.",
     ["me", "woodworking"], [("me", "woodworking", "learns")]),
    ("g14", "2023-03-14", "I switched my phone to a Pixel 7.",
     ["me", "Pixel 7"], [("me", "Pixel 7", "uses_phone")]),
    ("g15", "2023-09-02", "Growing tomatoes and basil on the balcony this year.",
     ["me", "tomatoes", "basil"],
     [("me", "tomatoes", "grows"), ("me", "basil", "grows")]),

    # third-party distractors (people in my life doing shadow-versions of my facts)
    ("t01", "2022-03-05", "Becky adopted a cat named Whiskers last week.",
     ["Becky", "Whiskers"], [("Becky", "Whiskers", "has_pet")]),
    ("t02", "2023-04-02", "Raj takes his dog to the dog park every Sunday.",
     ["Raj", "the dog park"], [("Raj", "the dog park", "went_to")]),
    ("t03", "2021-11-30", "Becky moved from Austin to Chicago for her new role.",
     ["Becky", "Chicago"], [("Becky", "Chicago", "lives_in")]),
    ("t04", "2022-07-16", "Raj started working at Initech as a designer.",
     ["Raj", "Initech"], [("Raj", "Initech", "employed_by")]),
    ("t05", "2023-08-21", "Raj no longer works at Initech; he joined Hooli.",
     ["Raj", "Initech", "Hooli"],
     [("Raj", "Initech", "employed_by", "ended"), ("Raj", "Hooli", "employed_by")]),
    ("t06", "2022-09-18", "Mia is training for the Boston Marathon this spring.",
     ["Mia", "Boston Marathon"], [("Mia", "Boston Marathon", "trains_for")]),
    ("t07", "2023-01-12", "Becky plays tennis in a league on Thursdays.",
     ["Becky", "tennis"], [("Becky", "tennis", "plays")]),
    ("t08", "2023-06-10", "Priya organized the Acme alumni picnic at Discovery Park.",
     ["Priya", "Acme alumni picnic", "Discovery Park"],
     [("Priya", "Acme alumni picnic", "organized")]),
    ("t09", "2024-05-04", "Mia bought a Honda Civic after her old car died.",
     ["Mia", "Honda Civic"], [("Mia", "Honda Civic", "drives")]),
    ("t10", "2024-07-27", "Raj is allergic to shellfish, so we picked an Italian place.",
     ["Raj", "shellfish"], [("Raj", "shellfish", "allergic_to")]),
    ("t11", "2023-11-25", "Becky visited Seattle for a conference and loved Pike Place.",
     ["Becky", "Seattle"], [("Becky", "Seattle", "visited")]),
    ("t12", "2024-09-14", "Dana used to live in Denver before moving to Portland.",
     ["Dana", "Denver", "Portland"],
     [("Dana", "Denver", "lives_in", "ended"), ("Dana", "Portland", "lives_in")]),
    ("t13", "2022-12-03", "Mia's book club is reading mysteries this winter.",
     ["Mia", "book club"], [("Mia", "book club", "member_of")]),
    ("t14", "2024-10-19", "Priya teaches a yoga class at the Y on Wednesdays.",
     ["Priya", "yoga class"], [("Priya", "yoga class", "teaches")]),
    ("t15", "2023-05-30", "Raj went to Japan last year and keeps recommending Kyoto.",
     ["Raj", "Japan"], [("Raj", "Japan", "traveled_to")]),
    ("t16", "2024-11-08", "Becky's sister is a dentist in Chicago.",
     ["Becky's sister", "Chicago"], [("Becky's sister", "Chicago", "works_in")]),
    ("t17", "2025-01-19", "Dana joined a climbing gym and is hooked on bouldering.",
     ["Dana", "climbing gym"], [("Dana", "climbing gym", "went_to")]),
    ("t18", "2025-02-23", "Priya's guitar recital is next month at the community hall.",
     ["Priya", "guitar"], [("Priya", "guitar", "plays")]),
    ("t19", "2025-03-30", "Mia hiked half of the Appalachian Trail last summer.",
     ["Mia", "Appalachian Trail"], [("Mia", "Appalachian Trail", "hiked")]),
    ("t20", "2025-04-20", "Becky's team at Globex shipped their big release.",
     ["Becky", "Globex"], [("Becky", "Globex", "employed_by")]),
]


def build_synth_store(path: str, event_facts: bool = False) -> KnowledgeGraph:
    cfg = Config.default()
    cfg.embedder = "st"
    cfg.self_entity = True
    cfg.self_name = "me"
    cfg.event_facts = event_facts
    g = KnowledgeGraph.open(path, cfg)
    items, table = [], {}
    for eid, day, text, ents, rels in _SYNTH:
        items.append(CorpusItem(id=eid, modality="text", source_ref=f"synthetic/{eid}",
                                title=eid, text=text,
                                created_at=f"{day}T12:00:00+00:00"))
        table[text] = Extraction(
            entities=[_E(n) for n in ents],
            tags=["personal"],
            relations=[_R(*r) for r in rels])
    g.extractor = ScriptedExtractor(table)
    rep = g.ingest(items)
    g.save()
    return g


# --------------------------------------------------------------------------- #
# Probe sets: (id, category, question, as_of, gold_eps, gold_substrings)
# gold_substrings: case-insensitive; hit = ANY present in context; coverage = fraction.
# --------------------------------------------------------------------------- #
SYNTH_PROBES = [
    # preference / disposition (sharp edge #6: no lane, no seeding path)
    ("p1", "preference", "What do I like to do for fun?", None,
     ["ep_park1", "ep_ten1", "ep_clmb1", "ep_hike1"],
     ["park", "tennis", "climbing", "Mount Si"]),
    ("p2", "preference", "What hobbies do I have?", None,
     ["ep_f04", "ep_ten1", "ep_clmb1"],
     ["guitar", "tennis", "climbing", "book club"]),
    ("p3", "preference", "What do I usually do on weekends?", None,
     ["ep_park1", "ep_hike1"], ["park", "Mount Si"]),

    # history (closed / ended facts)
    ("h1", "history", "Where did I use to work?", None,
     ["ep_job1", "ep_job2"], ["Acme"]),
    ("h2", "history", "Which companies have I worked for?", None,
     ["ep_job1", "ep_job3"], ["Acme", "Globex"]),
    ("h3", "history", "Do I still volunteer at the animal shelter?", None,
     ["ep_vol1", "ep_vol2"], ["stopped", "ended"]),
    ("h4", "history", "Did I ever live in Seattle?", None,
     ["ep_home1"], ["Seattle"]),
    ("h5", "history", "Tell me about my time at Acme.", None,
     ["ep_job1", "ep_job2"], ["Acme", "data analyst"]),
    ("h6", "history", "How has my career changed over time?", None,
     ["ep_job1", "ep_job2", "ep_job3"], ["Acme", "Globex"]),

    # as-of (point-in-time)
    ("a1", "asof", "Where did I live in 2022?", "2022-06-01",
     ["ep_home1"], ["Seattle"]),
    ("a2", "asof", "Where did I work in 2020?", "2020-06-01",
     ["ep_job1"], ["Acme"]),
    ("a3", "asof", "Where was I in early November 2023?", "2023-11-05",
     ["ep_trip1"], ["Japan"]),
    ("a4", "asof", "Who was my employer in 2023?", "2023-03-01",
     ["ep_job3"], ["Globex"]),
    ("a5", "asof-text", "Where did I live in 2022?", None,   # as_of NOT passed: edge #5
     ["ep_home1"], ["Seattle"]),

    # counting / aggregation
    ("c1", "counting", "How many times did I go to the park?", None,
     ["ep_park1", "ep_park2", "ep_park3", "ep_park4", "ep_park5"], ["park"]),
    ("c2", "counting", "How many yoga classes did I attend?", None,
     ["ep_yoga1", "ep_yoga2", "ep_yoga3"],
     ["2024-03-10", "2024-04-14", "2024-06-02"]),
    ("c3", "counting", "How many times have I traveled abroad?", None,
     ["ep_trip1"], ["Japan"]),
    ("c4", "counting", "How often do I go to the park?", None,
     ["ep_park1", "ep_park2", "ep_park3", "ep_park4", "ep_park5"], ["park"]),

    # projection-sensitive (gold reachable mainly through a CLOSED me-edge)
    ("b1", "projection",
     "Who put together the get-together for former employees of my old company?", None,
     ["ep_t08"], ["Priya"]),
    ("b2", "projection", "Which cities did I visit on my trip abroad?", None,
     ["ep_trip1"], ["Tokyo", "Kyoto"]),

    # ordinary lookups (regression canaries)
    ("n1", "canary", "What is my cat's name?", None, ["ep_pet1"], ["Luna"]),
    ("n2", "canary", "Where does my sister live?", None, ["ep_sis1"], ["Boston"]),
    ("n3", "canary", "What car do I drive?", None, ["ep_car1"], ["Subaru"]),
    ("n4", "canary", "What am I allergic to?", None, ["ep_alg1"], ["peanuts"]),
    ("n5", "canary", "What sport do I play on Tuesdays?", None, ["ep_ten1"], ["tennis"]),
    ("n6", "canary", "Where do I live now?", None, ["ep_home2"], ["Denver"]),
    ("n7", "canary", "Where do I work now?", None, ["ep_job3"], ["Globex"]),
    ("n8", "canary", "When does my book club meet?", None, ["ep_book1"], ["Thursday"]),
    ("n9", "canary", "Who is my dentist?", None, ["ep_f02"], ["Patel"]),
    ("n10", "canary", "What instrument do I play?", None, ["ep_f04"], ["guitar"]),
]

# events_pilot canaries: gold_eps are SESSION PREFIXES (chunk ids start with them)
PILOT_PROBES = [
    ("e1", "canary", "What paints have I been using for my tank model?", None,
     ["ep_06f04340__14f9ee3c"], ["Vallejo"]),
    ("e2", "canary", "What rare album did I add to my vinyl collection?", None,
     ["ep_06f04340__150756fc_2"], ["Miles Davis"]),
    ("e3", "canary", "What game did I decide to play with my sister?", None,
     ["ep_06f04340__b459f888_3"], ["Mysterium"]),
    ("e4", "canary", "What did my sister give me as a gift?", None,
     ["ep_06f04340__728deb4d_4"], ["espresso machine"]),
    ("e5", "canary", "What did I recently replace in my bathroom?", None,
     ["ep_06f04340__0844dea6"], ["light fixture"]),
    ("e6", "canary", "What workout class have I been taking for cardio?", None,
     ["ep_06f04340__fea299b4"], ["Zumba"]),
    ("e7", "counting", "How did our monthly game night go?", None,
     ["ep_06f04340__b459f888_3"], ["lost"]),
    ("e8", "canary", "Whose concert did I attend where the crowd sang Bad Guy?", None,
     ["ep_06f04340__cf543226_2"], ["Billie"]),
]


# Round-2 event probes (synth_ev only): occurrence grammar + as-of behavior of events.
# 7th element = neg_golds: substrings that must NOT appear in the FACTS/HISTORY sections
# (episode text is exempt — raw text legitimately mentions anything).
EVENT_PROBES = [
    # a past bounded event must render as an occurrence, never as an open/closed STATE
    ("ev1", "event", "Have I ever been to Japan?", None,
     ["ep_trip1"], ["Japan", "2023-11-01 -> 2023-11-14"],
     ["japan (since", "japan (mentioned", "until 2023-11-14"]),
    # current view: repeated point events must not read as a standing state
    ("ev2", "event", "Tell me about the park.", None,
     ["ep_park1"], ["park"],
     ["the park (since", "the park (mentioned"]),
    # as-of AFTER an event: findable; later occurrences (post-T) filtered from the delta
    ("ev3", "event-asof", "What did I do at the park?", "2025-01-10",
     ["ep_park1"], ["on 2025-01-05"],
     ["on 2025-02-02", "on 2025-03-09", "on 2025-04-13", "on 2025-05-18"]),
    # as-of BEFORE the event: MY trip must be absent from every fact-derived section
    # (Raj's earlier Japan trip legitimately predates T and may render)
    ("ev4", "event-asof", "Have I traveled to Japan?", "2023-06-01",
     [], [], ["me --traveled_to--> japan", "2023-11-01"]),
    # counting: the five park visits surface as five dated occurrence rows
    ("ev5", "event-count", "How many times did I go to the park?", None,
     ["ep_park1", "ep_park2", "ep_park3", "ep_park4", "ep_park5"],
     ["on 2025-01-05", "on 2025-02-02", "on 2025-03-09", "on 2025-04-13",
      "on 2025-05-18"], []),
]


# --------------------------------------------------------------------------- #
# Round-2 run plan: (store_key, variant_name, config_overrides, probe_sets)
# --------------------------------------------------------------------------- #
RUN_PLAN = {
    "synth":    [("baseline", {}), ("A_amended", {"history_all_lanes": True})],
    "synth_ev": [("EV", {"event_facts": True}),
                 ("EV_A", {"event_facts": True, "history_all_lanes": True})],
    "pilot":    [("baseline", {}), ("A_amended", {"history_all_lanes": True})],
}
# the config recommended for promotion; per-probe hit/coverage must be >= the recorded
# Round-1 baseline under it
PROMOTED = {"synth_ev": "EV_A", "pilot": "A_amended"}

SNAPSHOT_QIDS = {"synth": ["h2", "c3", "n1", "a5"],
                 "synth_ev": ["h2", "c1", "c2", "c3", "a5", "ev1", "ev3"],
                 "pilot": ["e2", "e5"]}


def _sections(blob: str) -> dict:
    """Split the context blob into its EPISODES / FACTS / HISTORY sections."""
    eps, facts, hist = blob, "", ""
    if "\nFACTS currently valid" in blob:
        eps, rest = blob.split("\nFACTS currently valid", 1)
        facts = rest
        if "\nHISTORY (" in rest:
            facts, hist = rest.split("\nHISTORY (", 1)
    return {"episodes": eps, "facts": facts, "history": hist}


def run_probes(g: KnowledgeGraph, probes, variant: str, store_name: str,
               out_dir: str) -> list[dict]:
    rows = []
    snap_dir = os.path.join(out_dir, "contexts", store_name, variant)
    os.makedirs(snap_dir, exist_ok=True)
    for probe in probes:
        qid, cat, q, as_of, gold_eps, golds = probe[:6]
        negs = probe[6] if len(probe) > 6 else []
        t0 = time.perf_counter()
        res = g.search(q, as_of=as_of)
        ms = (time.perf_counter() - t0) * 1000
        blob = res.context or ""
        low = blob.lower()
        secs = {k: v.lower() for k, v in _sections(blob).items()}
        hit_eps = [h.episode_id for h in res.hits]
        def ep_hit(ge):
            return any(he == ge or he.startswith(ge + "#") for he in hit_eps)
        recall = (sum(ep_hit(ge) for ge in gold_eps) / len(gold_eps)) if gold_eps else None
        found = [s for s in golds if s.lower() in low]
        where = {s: [sec for sec in ("episodes", "facts", "history")
                     if s.lower() in secs[sec]] for s in found}
        # negs are checked against the fact-derived sections only: episode raw text may
        # legitimately mention anything, but FACTS/HISTORY must not
        fact_secs = secs["facts"] + secs["history"]
        neg_viol = [s for s in negs if s.lower() in fact_secs]
        n_hist = secs["history"].count("\n- ") if secs["history"] else 0
        rows.append({
            "store": store_name, "variant": variant, "qid": qid, "cat": cat,
            "question": q, "as_of": as_of, "lane": res.lane,
            "latency_ms": round(ms, 1), "ctx_chars": len(blob),
            "n_facts": len(res.facts), "n_hist_lines": n_hist,
            "hit": bool(found) if golds else not neg_viol,
            "coverage": round(len(found) / len(golds), 3) if golds
                        else (0.0 if neg_viol else 1.0),
            "found": found, "found_in": where, "neg_violations": neg_viol,
            "ep_recall": None if recall is None else round(recall, 3),
            "hits": hit_eps,
        })
        if qid in SNAPSHOT_QIDS.get(store_name, []):
            with open(os.path.join(snap_dir, f"{qid}.txt"), "w") as f:
                f.write(blob)
    return rows


# --------------------------------------------------------------------------- #
# D: PPR mass concentration with and without the self cap
# --------------------------------------------------------------------------- #
def ppr_mass_analysis(store_path: str, base_cfg: Config, queries: list[str]) -> list[dict]:
    from kg.retrieval import (Seeder, personalized_pagerank, projected_graph,
                              self_like_ids)
    out = []
    for guard in ("none", "cap"):
        cfg = replace(base_cfg, self_guard=guard)
        g = KnowledgeGraph.open(store_path, cfg)
        seeder = Seeder(g.store, g.embedder, g.canon, cfg)
        G = projected_graph(g.store, cfg)
        self_ids = self_like_ids(g.store, cfg)
        for q in queries:
            seeds = seeder.seed(q)
            pers = {nid: s * g.canon.idf_weight(nid) for nid, s in seeds.items()
                    if nid in G and s > 0}
            if not pers:
                continue
            ppr = personalized_pagerank(G, alpha=cfg.ppr_damping, personalization=pers)
            eps = sorted(((nid, sc) for nid, sc in ppr.items()
                          if g.store.get_node(nid)
                          and g.store.get_node(nid).ntype == NodeType.EPISODE),
                         key=lambda x: -x[1])
            tot = sum(sc for _n, sc in eps) or 1.0
            shares = [sc / tot for _n, sc in eps]
            ent = -sum(p * math.log(p) for p in shares if p > 0)
            out.append({
                "guard": guard, "query": q,
                "self_mass": round(sum(ppr.get(i, 0.0) for i in self_ids), 5),
                "n_episodes": len(eps),
                "top1_share": round(shares[0], 4) if shares else 0,
                "top3_share": round(sum(shares[:3]), 4),
                "top5_share": round(sum(shares[:5]), 4),
                "entropy": round(ent, 4),
                "top5": [n for n, _s in eps[:5]],
            })
    return out


# --------------------------------------------------------------------------- #
def _fact_edge_stats(path: str) -> dict:
    """Open/closed/event fact-edge counts for a store (edge-count delta reporting)."""
    from kg.models import EdgeType
    from kg.store import GraphStore
    st = GraphStore.open(path, Config.default())
    open_f = closed_f = event_f = total = 0
    for _u, _v, d in st.g.edges(data=True):
        if d.get("etype") != EdgeType.RELATED_TO.value:
            continue
        total += 1
        if d.get("event"):
            event_f += 1
        if d.get("invalid_at"):
            closed_f += 1
        else:
            open_f += 1
    return {"fact_edges": total, "open": open_f, "closed": closed_f, "event": event_f}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", default="synth,synth_ev,pilot")
    ap.add_argument("--out", default="runs/offline_eval_round2")
    ap.add_argument("--pilot-db", default="store/events_pilot.db")
    ap.add_argument("--baseline", default="runs/offline_eval/results.json",
                    help="Round-1 results.json holding the recorded per-probe baseline")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    stores = [s.strip() for s in args.stores.split(",") if s.strip()]
    work = tempfile.mkdtemp(prefix="offline-eval-")

    paths, edge_stats = {}, {}
    for key, event_facts in (("synth", False), ("synth_ev", True)):
        if key not in stores:
            continue
        sp = os.path.join(work, key, "kg.db")
        os.makedirs(os.path.dirname(sp), exist_ok=True)
        print(f"building {key} store (event_facts={event_facts}) ...", flush=True)
        g = build_synth_store(sp, event_facts=event_facts)
        print("  ", g.store.stats().get("nodes"), "nodes")
        del g
        paths[key] = sp
        edge_stats[key] = _fact_edge_stats(sp)
    if "pilot" in stores:
        pp = os.path.join(work, "pilot", "kg.db")
        os.makedirs(os.path.dirname(pp), exist_ok=True)
        shutil.copyfile(args.pilot_db, pp)   # never touch the original
        paths["pilot"] = pp

    probe_sets = {"synth": SYNTH_PROBES, "synth_ev": SYNTH_PROBES + EVENT_PROBES,
                  "pilot": PILOT_PROBES}
    all_rows = []
    for store_name, path in paths.items():
        for variant, over in RUN_PLAN[store_name]:
            cfg = Config.default()
            cfg.embedder = "st"
            if store_name.startswith("synth"):
                cfg.self_entity = True
                cfg.self_name = "me"
            for k, v in over.items():
                setattr(cfg, k, v)
            g = KnowledgeGraph.open(path, cfg)
            rows = run_probes(g, probe_sets[store_name], variant, store_name, args.out)
            all_rows.extend(rows)
            nhit = sum(r["hit"] for r in rows)
            print(f"{store_name:8s} {variant:10s} hit {nhit}/{len(rows)} "
                  f"cov {sum(r['coverage'] for r in rows)/len(rows):.3f} "
                  f"chars {sum(r['ctx_chars'] for r in rows)//len(rows)}", flush=True)
            del g

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(all_rows, f, indent=1)
    if edge_stats:
        with open(os.path.join(args.out, "edge_stats.json"), "w") as f:
            json.dump(edge_stats, f, indent=1)
        for k, s in edge_stats.items():
            print(f"edges {k:8s} {s}")

    # ---- gate: promoted configs must not regress any recorded-baseline probe --------
    recorded = {}
    if os.path.exists(args.baseline):
        for r in json.load(open(args.baseline)):
            if r["variant"] == "baseline":
                recorded[(r["store"], r["qid"])] = r
    regressions, checked = [], 0
    for store_name, variant in PROMOTED.items():
        base_store = "synth" if store_name.startswith("synth") else store_name
        for r in all_rows:
            if r["store"] != store_name or r["variant"] != variant:
                continue
            base = recorded.get((base_store, r["qid"]))
            if base is None:
                continue           # new (Round-2) probe: no recorded baseline to hold
            checked += 1
            if r["hit"] < base["hit"] or r["coverage"] < base["coverage"] - 1e-9:
                regressions.append(
                    f"{store_name}/{variant} {r['qid']}: hit {base['hit']}->{r['hit']} "
                    f"cov {base['coverage']}->{r['coverage']}")
    for r in all_rows:
        if r.get("neg_violations"):
            regressions.append(f"{r['store']}/{r['variant']} {r['qid']}: "
                               f"NEG violated {r['neg_violations']}")
    print(f"baseline gate: {checked} probes checked against {args.baseline}")
    if regressions:
        print("REGRESSIONS:")
        for line in regressions:
            print("  " + line)
        sys.exit(1)
    print("no regressions. done ->", args.out)


if __name__ == "__main__":
    main()
