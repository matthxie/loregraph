"""Extraction-completeness metrics for aggregate-shaped ("how many" / "how much")
LongMemEval questions. This automates the manual audit in `spikes/completeness/REPORT.md`
so extraction regressions/improvements show up as a per-run trend instead of a one-off spike.

TIER 1 — "quantity capture rate" (deterministic, $0): for aggregate-shaped questions, scan
the gold-evidence sessions' raw text for money amounts, then check whether each amount also
shows up in some ingested node's name. Mirrors the REPORT's "are quantities parseable when
present?" check (amounts are stored as plain-string entity node names, e.g. "$800").

TIER 2 — "occurrence completeness" (LLM-assisted, behind a flag): port of
`spikes/completeness/ground_truth.py` (one LLM pass enumerates every true occurrence from
the gold-evidence sessions) + `spikes/completeness/dump_edges.py` (episode_id-scoped,
fuzzy node-name audit) — automated into CAPTURED / COLLAPSED / MISSING per occurrence.
"""
from __future__ import annotations

import json
import re

# --------------------------------------------------------------------------- #
# Question-shape detection — single source of truth for "is this question the kind
# where a naive COUNT/SUM over the graph would even apply?" (REPORT's two shapes).
# --------------------------------------------------------------------------- #
_AGG_RE = re.compile(
    r"\bhow many\b|\bhow much\b|\bhow often\b|\btotal\b|\baltogether\b|\bin all\b", re.I)
_SUM_RE = re.compile(r"\bhow much\b|\btotal\b|\baltogether\b|\bin all\b", re.I)
_COUNT_RE = re.compile(r"\bhow many\b|\bhow often\b", re.I)


def is_aggregate_question(text: str) -> bool:
    return bool(_AGG_RE.search(text or ""))


def question_shape(text: str) -> str:
    """'sum' (money/amount aggregation) or 'count' (discrete-event counting); 'sum' wins
    a question that (rarely) matches both patterns, since SUM is the harder failure mode."""
    t = text or ""
    if _SUM_RE.search(t):
        return "sum"
    if _COUNT_RE.search(t):
        return "count"
    return "other"


# --------------------------------------------------------------------------- #
# TIER 1 — quantity capture rate
# --------------------------------------------------------------------------- #
# Money ($800 / 1,300 dollars) — the numeral shape the REPORT found being dropped at
# extraction time (9 amounts across 3 SUM questions, only 3 ever became graph nodes).
# Deliberately NOT matching bare small numerals too: free-standing digits are everywhere
# in chat text (numbered lists, dates, times) with no reliable way to tell "a count the
# question is aggregating" from "a list index" by regex alone — REPORT's own finding was
# that discrete-event COUNTs are usually captured fine at extraction time; it's SUM/amount
# questions that silently drop the numbers, so that's what this deterministic tier checks.
_QUANTITY_RE = re.compile(
    r"\$\s?(?P<money1>\d[\d,]*(?:\.\d+)?)"
    r"|(?P<money2>\d[\d,]*(?:\.\d+)?)\s?dollars\b", re.I)


def normalize_amount(raw: str) -> str:
    """'$1,300.00' / '1300 dollars' / '3' -> a canonical bare-number string ('1300', '3')
    so text mentions and node-name mentions of the same quantity compare equal."""
    v = raw.replace(",", "").replace("$", "").strip()
    try:
        f = float(v)
    except ValueError:
        return v
    return str(int(f)) if f == int(f) else str(f)


def find_amounts_in_text(text: str) -> list[str]:
    """Every distinct normalized quantity mentioned in `text`, in first-seen order
    (repeated mentions of the same amount count once — REPORT's dedup rule)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in _QUANTITY_RE.finditer(text or ""):
        raw = m.group("money1") or m.group("money2")
        if not raw:
            continue
        amt = normalize_amount(raw)
        if amt not in seen:
            seen.add(amt)
            out.append(amt)
    return out


def node_amount_set(store) -> set[str]:
    """Every normalized quantity that appears in any ingested node's name — the graph-side
    half of the capture check (REPORT: amounts land as a plain string entity node name,
    e.g. name=="$800", not a typed numeric field)."""
    found: set[str] = set()
    for n in store.nodes.values():
        name = getattr(n, "name", "") or ""
        for m in _QUANTITY_RE.finditer(name):
            raw = m.group("money1") or m.group("money2")
            if raw:
                found.add(normalize_amount(raw))
    return found


def quantity_capture_for_question(question_id: str, question_text: str,
                                  evidence_text: str, store) -> dict | None:
    """Tier-1 result for one question, or None if it doesn't apply (not aggregate-shaped,
    or its gold evidence carries no parseable quantity at all — "n/a", not a misleading 0)."""
    if not is_aggregate_question(question_text):
        return None
    amounts = find_amounts_in_text(evidence_text)
    if not amounts:
        return None
    in_graph = node_amount_set(store)
    hit = sum(1 for a in amounts if a in in_graph)
    return {
        "question_id": question_id, "shape": question_shape(question_text),
        "amounts_in_text": len(amounts), "amounts_in_graph": hit,
        "capture_rate": round(hit / len(amounts), 3),
    }


def summarize_tier1(records: list[dict]) -> dict | None:
    """Aggregate per-run tier-1 numbers, or None if no question in the run produced a
    tier-1 record (no aggregate-shaped questions with a parseable quantity at all)."""
    if not records:
        return None
    total_text = sum(r["amounts_in_text"] for r in records)
    total_graph = sum(r["amounts_in_graph"] for r in records)
    return {
        "n_questions": len(records),
        "amounts_in_text": total_text, "amounts_in_graph": total_graph,
        "capture_rate": round(total_graph / total_text, 3) if total_text else None,
        "per_question": records,
    }


# --------------------------------------------------------------------------- #
# TIER 2 — occurrence completeness (LLM-assisted)
# --------------------------------------------------------------------------- #
# Same enumeration prompt/schema as spikes/completeness/ground_truth.py, generalized to
# any question (not just the spike's 8 hand-picked ids).
_OCCURRENCE_PROMPT = """You are auditing a personal-memory dataset. Below are full chat \
sessions (in chronological order, each labeled with its session_id and date) belonging to \
one user. A question was asked about this user: "{question}"

Read ALL sessions carefully and enumerate EVERY distinct occurrence of the specific event, \
item, or amount that the question is counting or summing (be careful: some sessions are \
irrelevant distractors; some occurrences may be mentioned as scheduled/future/cancelled — \
note that in your quote). For each occurrence give: the session_id, a short exact quote \
(<=200 chars) that establishes it, and if it's a monetary/numeric amount, the number itself.

Respond ONLY with JSON of this shape:
{{"occurrences": [{{"session_id": "...", "quote": "...", "amount": null}}, ...],
  "notes": "any caveats, e.g. ambiguity or whether an occurrence is future/cancelled"}}

SESSIONS:
{sessions}
"""


def enumerate_occurrences_llm(client, model: str, question_id: str, question_text: str,
                              evidence_sessions: list[dict], meter=None) -> list[dict]:
    """One LLM call enumerating ground-truth occurrences over `question_id`'s gold-evidence
    sessions only (each a {"session_id","date","text"} dict). Records cost on `meter` under
    site "audit.completeness" if given. Returns [] on any parse/call failure — tier 2 is
    best-effort audit, never fatal to the run."""
    sessions_text = "\n\n".join(
        f"--- session_id={s['session_id']} date={s.get('date', '')} ---\n{s['text']}"
        for s in evidence_sessions)
    prompt = _OCCURRENCE_PROMPT.format(question=question_text, sessions=sessions_text)
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}, temperature=0)
        if meter is not None:
            meter.record("audit.completeness", model, resp, label=question_id)
        content = resp.choices[0].message.content
        parsed = json.loads(content)
        occs = parsed.get("occurrences") or []
        return [o for o in occs if isinstance(o, dict) and o.get("session_id")]
    except Exception:  # noqa: BLE001 — audit is best-effort, never sinks the run
        return []


_STRUCTURAL_ETYPES = {"MENTIONED_IN", "RESOLVES_TO", "TAGGED_AS",
                      "SIMILAR_TO", "SHARED_TAG", "SHARED_ENTITY"}


def _fact_node_names(store) -> list[str]:
    """Every node name touched by a non-structural (fact) edge, anywhere in the store —
    the automatable equivalent of dump_edges.py's manual per-episode dump, pooled across
    ALL of this question's gold-evidence episodes rather than gated to one at a time: a
    fact first asserted while reading one session (e.g. "Zumba" -> "Tuesdays and
    Thursdays") is recorded under THAT session's episode_id even when a later occurrence's
    quote re-mentions the same fact in a different gold session, so per-single-session
    episode_id matching would wrongly call the later mention MISSING. Since
    run_per_instance builds one FRESH graph per question (kg/testrun.py), every edge in
    the store already belongs to this one question — episode_id here only needs to
    separate real facts from structural bookkeeping edges, not one session from another."""
    names: list[str] = []
    for u, v, data in store.all_edges():
        if data.get("etype") in _STRUCTURAL_ETYPES:
            continue
        for nid in (u, v):
            n = store.get_node(nid)
            if n is not None and getattr(n, "name", ""):
                names.append(n.name)
    return names


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> set[str]:
    # trailing-'s' stripped so a plural in a compound node name ("Tuesdays and Thursdays")
    # still matches the singular day mentioned in one occurrence's quote ("every Tuesday")
    return {w.rstrip("s") for w in _WORD_RE.findall(s.lower()) if len(w) > 2}


def _occurrence_matches(name: str, occ: dict, shape: str = "sum") -> bool:
    """Fuzzy match a node name against one LLM-enumerated occurrence. For "sum"-shaped
    (money) questions with a numeric `amount`, compare normalized numbers; otherwise treat
    the node name as matching if it's a literal substring of the quote OR they share a
    distinctive word token (entity names are pulled verbatim from source text, so this
    mirrors eyeballing dump_edges.py's dump by hand — a compound node like "Tuesdays and
    Thursdays" should still be found by a quote that only mentions "Tuesday").

    NB: for "count"-shaped questions the enumeration prompt's `amount` field is sometimes
    a multiplicity annotation, not a literal number ("twins" -> amount 2, a 4-item list ->
    amount 4) — there's no node named "2", so amount-matching would wrongly call every
    such occurrence MISSING. Only "sum" questions get the number-matching path."""
    if shape == "sum" and occ.get("amount") is not None:
        target = normalize_amount(str(occ["amount"]))
        name_amounts = {normalize_amount(m.group("money1") or m.group("money2"))
                       for m in _QUANTITY_RE.finditer(name)
                       if m.group("money1") or m.group("money2")}
        return target in name_amounts
    norm_name = name.strip().lower()
    quote = (occ.get("quote") or "").lower()
    if not norm_name:
        return False
    if len(norm_name) > 1 and norm_name in quote:
        return True
    return bool(_tokens(name) & _tokens(quote))


_STOP_CAP = {"I", "The", "A", "An", "My", "Our", "She", "He", "It", "They", "We", "This",
            "That", "Some", "Also", "Just", "Should", "Let", "There", "Their", "His", "Her"}
_CAP_WORD_RE = re.compile(r"\b[A-Z][a-z]{2,}\b")


def _proper_nouns(quote: str) -> set[str]:
    return {w for w in _CAP_WORD_RE.findall(quote or "") if w not in _STOP_CAP}


def dedup_occurrences(occurrences: list[dict]) -> list[dict]:
    """Collapse raw LLM-enumerated occurrences that are the SAME real-world fact re-told
    in a later gold session — ground_truth.py's enumeration prompt reads every gold session
    but has no cross-session memory, so a fact repeated in a follow-up session (very common
    in LongMemEval's dated chat logs) comes back as its own quote. Keyed on shared proper
    nouns in the quote, since names are the one thing repeated verbatim across retellings;
    an occurrence with no proper noun (nothing to key on) is never merged."""
    groups: list[set[str]] = []
    kept: list[dict] = []
    for occ in occurrences:
        keys = _proper_nouns(occ.get("quote", ""))
        if not keys:
            kept.append(occ)
            continue
        idx = next((i for i, g in enumerate(groups) if g & keys), None)
        if idx is None:
            groups.append(set(keys))
            kept.append(occ)
        else:
            groups[idx] |= keys   # widen so a later occurrence bridging two names still matches
    return kept


def classify_occurrences(store, question_id: str, occurrences: list[dict],
                         shape: str = "sum") -> list[dict]:
    """CAPTURED / COLLAPSED / MISSING per occurrence (REPORT's definitions), after
    deduping same-fact re-mentions (`dedup_occurrences`):
      CAPTURED  — a distinct fact node in this question's graph names it.
      COLLAPSED — a node names it, but the SAME node is also the only match for >=1
                  OTHER occurrence (a compound node flattening >1 fact, e.g. REPORT's
                  "Tuesdays and Thursdays" node for two Zumba occurrences).
      MISSING   — no fact-edge node in the graph matches at all.
    `question_id` is unused (kept for API stability / future per-episode narrowing);
    matching pools every fact node in `store`, since run_per_instance builds one fresh
    graph per question — see `_fact_node_names`. `shape` (`question_shape()`) gates
    whether an occurrence's `amount` is treated as a literal number to find in a node
    name (only meaningful for "sum"/money questions — see `_occurrence_matches`).

    NB: does NOT auto-dedup same-fact re-mentions (`dedup_occurrences` is available but
    deliberately not called here) — merging is itself ambiguous (a repeated NAME can mean
    "the same fact retold" or "two genuinely distinct occurrences sharing one subject",
    e.g. two different Zumba class days), so REPORT's own numbers came from a human
    resolving that case by case. Leaving it unmerged means a fact repeated verbatim across
    several gold sessions can inflate this run's occurrence count — a known, documented
    slack in the automated tier, not a correctness bug.
    """
    candidates = _fact_node_names(store)
    matched_node: dict[int, str] = {}
    for i, occ in enumerate(occurrences):
        for nm in candidates:
            if _occurrence_matches(nm, occ, shape):
                matched_node[i] = nm
                break
    claims: dict[str, list[int]] = {}
    for i, nm in matched_node.items():
        claims.setdefault(nm, []).append(i)

    out = []
    for i, occ in enumerate(occurrences):
        nm = matched_node.get(i)
        if nm is None:
            status = "MISSING"
        elif len(claims[nm]) > 1:
            status = "COLLAPSED"
        else:
            status = "CAPTURED"
        out.append({**occ, "status": status, "node": nm})
    return out


def summarize_tier2(classified_by_question: dict[str, tuple[str, list[dict]]]) -> dict | None:
    """Aggregate per-run tier-2 numbers from {question_id: (shape, [classified occurrence])}.
    None if no question produced any classified occurrence (tier 2 off, or no aggregate
    questions with gold evidence in this run — "n/a", not a misleading 0)."""
    all_occs = [o for _shape, occs in classified_by_question.values() for o in occs]
    if not all_occs:
        return None

    def _bucket(occs: list[dict]) -> dict:
        n = len(occs)
        captured = sum(1 for o in occs if o["status"] == "CAPTURED")
        collapsed = sum(1 for o in occs if o["status"] == "COLLAPSED")
        missing = sum(1 for o in occs if o["status"] == "MISSING")
        return {"n": n, "captured": captured, "collapsed": collapsed, "missing": missing,
                "pct_captured": round(captured / n, 3) if n else None,
                "pct_collapsed": round(collapsed / n, 3) if n else None,
                "pct_missing": round(missing / n, 3) if n else None}

    by_shape: dict[str, list[dict]] = {}
    for shape, occs in classified_by_question.values():
        by_shape.setdefault(shape, []).extend(occs)

    return {
        **_bucket(all_occs),
        "by_shape": {shape: _bucket(occs) for shape, occs in by_shape.items()},
        "per_question": {qid: {"shape": shape, "occurrences": occs}
                         for qid, (shape, occs) in classified_by_question.items()},
    }
