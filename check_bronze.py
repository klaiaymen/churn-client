"""check_bronze.py : contrôle que Bronze = copie fidèle des CSV."""
import duckdb

con = duckdb.connect("churn.duckdb", read_only=True)
tables = ["customers", "subscriptions", "orders", "complaints", "interactions"]

print(f"{'table':<15}{'lignes CSV':>12}{'lignes bronze':>15}  statut")
ok = True
for t in tables:
    n_csv = con.execute(f"SELECT count(*) FROM read_csv('data/raw/{t}.csv', header=true)").fetchone()[0]
    n_br = con.execute(f"SELECT count(*) FROM bronze.{t}").fetchone()[0]
    status = "OK" if n_csv == n_br else "ECART"
    ok &= n_csv == n_br
    print(f"{t:<15}{n_csv:>12,}{n_br:>15,}  {status}")

print("\nColonnes de bronze.orders :")
print(con.execute("DESCRIBE bronze.orders").df()[["column_name", "column_type"]].to_string(index=False))

print("\nAperçu :")
print(con.execute("SELECT * FROM bronze.orders LIMIT 3").df().to_string(index=False))
print("\nRésultat global :", "TOUT EST OK" if ok else "PROBLEME")