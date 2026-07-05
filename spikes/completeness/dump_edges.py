"""Dump edges touching a given episode (session) for manual CAPTURED/COLLAPSED/MISSING
classification. Usage:
  .venv\\Scripts\\python.exe spikes\\completeness\\dump_edges.py <qid> <session_id> [<session_id> ...]
"""
import json
import sqlite3
import sys
import os

STORE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stores")


def node_name(cur, node_id: str) -> str:
    cur.execute("SELECT payload FROM nodes WHERE id=?", (node_id,))
    row = cur.fetchone()
    if not row:
        return "<missing node>"
    try:
        p = json.loads(row[0])
    except json.JSONDecodeError:
        return "<bad payload>"
    return p.get("name") or p.get("canonical_name") or p.get("text") or node_id


def main() -> None:
    qid = sys.argv[1]
    session_ids = sys.argv[2:]
    con = sqlite3.connect(os.path.join(STORE_DIR, f"{qid}.db"))
    cur = con.cursor()
    for sid in session_ids:
        ep_id = f"ep_{qid}__{sid}"
        print(f"\n=== episode {ep_id} ===")
        cur.execute(
            "SELECT src, dst, etype, rel_tag, confidence, valid_at, invalid_at, valid "
            "FROM edges WHERE episode_id=? ORDER BY etype", (ep_id,)
        )
        rows = cur.fetchall()
        if not rows:
            print("  (no edges with this episode_id)")
        for src, dst, etype, rel_tag, conf, valid_at, invalid_at, valid in rows:
            if etype in ("MENTIONED_IN", "RESOLVES_TO", "TAGGED_AS", "SIMILAR_TO", "SHARED_TAG", "SHARED_ENTITY"):
                continue  # structural edges, not facts; skip for readability
            sn = node_name(cur, src)
            dn = node_name(cur, dst)
            print(f"  [{etype}] {sn!r} -[{rel_tag}]-> {dn!r}  conf={conf} "
                  f"valid_at={valid_at} invalid_at={invalid_at} valid={valid}")


if __name__ == "__main__":
    main()
