"""Inspect store/testrun.db: schema, entity type vocabulary, counts, sample rows."""
import sqlite3, json, sys

db = sys.argv[1] if len(sys.argv) > 1 else "store/testrun.db"
c = sqlite3.connect(db)
tables = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
print("tables:", tables)
for t in tables:
    n = c.execute(f"select count(*) from {t}").fetchone()[0]
    cols = [r[1] for r in c.execute(f"pragma table_info({t})")]
    print(f"\n== {t} ({n} rows) cols={cols}")
    for row in c.execute(f"select * from {t} limit 3"):
        s = str(row)
        print("  ", s[:300])
