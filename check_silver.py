"""check_silver.py : contrôles de qualité sur la couche Silver."""
import duckdb

con = duckdb.connect("churn.duckdb", read_only=True)


def one(sql):
    return con.execute(sql).fetchone()[0]


tests = {
    "orders : pas de doublon d'order_id":
        one("SELECT count(*) - count(DISTINCT order_id) FROM silver.orders") == 0,
    "orders : aucun montant <= 0":
        one("SELECT count(*) FROM silver.orders WHERE montant <= 0") == 0,
    "orders : toutes les dates sont typées":
        one("SELECT count(*) FROM silver.orders WHERE date_commande IS NULL") == 0,
    "customers : villes standardisées (8 villes + Inconnue)":
        one("SELECT count(DISTINCT ville) FROM silver.customers") == 9,
    "subscriptions : 3 plans exactement":
        one("SELECT count(DISTINCT plan) FROM silver.subscriptions") == 3,
    "customers : clés uniques":
        one("SELECT count(*) - count(DISTINCT customer_id) FROM silver.customers") == 0,
    "orders : tous les clients existent":
        one("""SELECT count(*) FROM silver.orders o
               LEFT JOIN silver.customers c USING (customer_id) WHERE c.customer_id IS NULL""") == 0,
}

print("=== Tests ===")
for name, passed in tests.items():
    print(f"[{'OK' if passed else 'ECHEC'}] {name}")

print("\n=== Volumes bronze -> silver ===")
for t in ["customers", "subscriptions", "orders", "complaints", "interactions"]:
    b = one(f"SELECT count(*) FROM bronze.{t}")
    s = one(f"SELECT count(*) FROM silver.{t}")
    print(f"{t:<15}{b:>10,} -> {s:>10,}   (écart {b - s:,})")
print(f"{'rejected_orders':<15}{one('SELECT count(*) FROM silver.rejected_orders'):>10,}")

print("\n=== Journal de qualité ===")
print(con.execute("SELECT * FROM silver.data_quality_log").df().to_string(index=False))

print("\n=== Types silver.orders ===")
print(con.execute("DESCRIBE silver.orders").df()[["column_name", "column_type"]].to_string(index=False))

print("\nRésultat global :", "TOUT EST OK" if all(tests.values()) else "DES TESTS ONT ECHOUE")