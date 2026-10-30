"""run_pipeline.py : exécute les fichiers sql/*.sql dans l'ordre alphabétique sur churn.duckdb."""
import time
from pathlib import Path

import duckdb

DB_PATH = "churn.duckdb"

con = duckdb.connect(DB_PATH)
for f in sorted(Path("sql").glob("*.sql")):
    t0 = time.time()
    print(f"-> {f.name} ...", end=" ", flush=True)
    con.execute(f.read_text(encoding="utf-8"))
    print(f"OK ({time.time() - t0:.1f}s)")

print("\nTables présentes dans la base :")
print(con.sql("""
    SELECT table_schema AS schema, table_name AS table
    FROM information_schema.tables
    ORDER BY 1, 2
"""))
con.close()