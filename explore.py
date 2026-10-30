import duckdb
con = duckdb.connect()

print("== Doublons dans orders ==")
print(con.sql("""
    SELECT count(*) AS lignes, count(DISTINCT order_id) AS ids_uniques
    FROM 'data/raw/orders.csv'
"""))

print("== Montants négatifs ==")
print(con.sql("SELECT count(*) AS nb FROM 'data/raw/orders.csv' WHERE montant < 0"))

print("== Variantes de casse des villes ==")
print(con.sql("""
    SELECT ville, count(*) AS nb FROM 'data/raw/customers.csv'
    GROUP BY ville ORDER BY lower(ville), ville
"""))

print("== Plans avec espaces parasites ==")
print(con.sql("""
    SELECT '[' || plan || ']' AS plan, count(*) AS nb
    FROM 'data/raw/subscriptions.csv' GROUP BY 1 ORDER BY 1
"""))



print(duckdb.sql("""
    SELECT
        count(*) FILTER (WHERE date_commande <= '2025-06-30') AS avant_ref,
        count(*) FILTER (WHERE date_commande >  '2025-06-30') AS apres_ref,
        max(date_commande) AS derniere_date
    FROM 'data/raw/orders.csv'
"""))