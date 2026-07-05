"""Enumerate ground-truth occurrences for each target question via an LLM pass over
that instance's *evidence* sessions (gpt-4o-mini). Paced to respect the org's RPD limit.

Usage: .venv\\Scripts\\python.exe spikes\\completeness\\ground_truth.py
"""
import json
import os
import sys
import time

os.environ.pop("OPENAI_API_KEY", None)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import kg  # noqa: F401  (triggers .env load)
import openai

QUESTIONS = {
    "00ca467f": "How many doctor's appointments did I go to in March?",
    "2788b940": "How many fitness classes do I attend in a typical week?",
    "2e6d26dc": "How many babies were born to friends and family members in the last few months?",
    "2b8f3739": "What is the total amount of money I earned from selling my products at the markets?",
    "36b9f61e": "What is the total amount I spent on luxury items in the past few months?",
    "129d1232": "How much money did I raise in total through all the charity events I participated in?",
    "21d02d0d": "How many fun runs did I miss in March due to work commitments?",
    "0a995998": "How many items of clothing do I need to pick up or return from a store?",
}

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ground_truth.json")

PROMPT = """You are auditing a personal-memory dataset. Below are full chat sessions (in \
chronological order, each labeled with its session_id and date) belonging to one user. \
A question was asked about this user: "{question}"

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


def load_evidence(qid: str) -> list[dict]:
    eps = []
    with open(os.path.join(ROOT, "dataset/longmemeval/small/episodes.jsonl"), encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["question_id"] == qid and r.get("is_evidence"):
                eps.append(r)
    eps.sort(key=lambda e: e["created_at"])
    return eps


def main() -> None:
    client = openai.OpenAI()
    results = {}
    if os.path.exists(OUT_PATH):
        results = json.load(open(OUT_PATH, encoding="utf-8"))

    for qid, question in QUESTIONS.items():
        if qid in results:
            print(f"{qid}: already done, skipping")
            continue
        eps = load_evidence(qid)
        sessions_text = "\n\n".join(
            f"--- session_id={e['session_id']} date={e['date']} ---\n{e['text']}" for e in eps
        )
        prompt = PROMPT.format(question=question, sessions=sessions_text)
        print(f"{qid}: calling gpt-4o-mini over {len(eps)} evidence sessions "
              f"({len(sessions_text)} chars) ...")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = resp.choices[0].message.content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"raw": content}
        results[qid] = parsed
        json.dump(results, open(OUT_PATH, "w", encoding="utf-8"), indent=2)
        print(f"  -> {len(parsed.get('occurrences', []))} occurrences found")
        time.sleep(10)  # pace ~9.5s+ apart per the org RPD budget

    print("\nWrote", OUT_PATH)


if __name__ == "__main__":
    main()
