"""check_gold.py : contrôles du schéma en étoile, de la table ml_features et de l'absence de fuite."""
import duckdb

con = duckdb.connect("churn.duckdb", read_only=True)


def one(sql):
    return con.execute(sql).fetchone()[0]


REF = "(SELECT ref_date FROM gold.params)"
n_feat = one("SELECT count(*) FROM gold.ml_features")
rate = one("SELECT avg(churn) FROM gold.ml_features")
cols = [r[0] for r in con.execute("DESCRIBE gold.ml_features").fetchall()]
n_nulls = sum(one(f"SELECT count(*) FROM gold.ml_features WHERE {c} IS NULL") for c in cols)

tests = {
    "ml_features : une ligne par client (clé unique)":
        one("SELECT count(*) - count(DISTINCT customer_id) FROM gold.ml_features") == 0,
    "ml_features : taux de churn entre 15 % et 21 %": 0.15 <= rate <= 0.21,
    "ml_features : aucune valeur NULL": n_nulls == 0,
    "population : uniquement des clients actifs à ref_date":
        one(f"""SELECT count(*) FROM gold.ml_population pop
                JOIN silver.subscriptions s ON s.customer_id = pop.customer_id
                WHERE s.date_fin <= {REF}
                  AND NOT EXISTS (SELECT 1 FROM silver.subscriptions s2
                                  WHERE s2.customer_id = pop.customer_id
                                    AND s2.date_debut <= {REF}
                                    AND (s2.date_fin IS NULL OR s2.date_fin > {REF}))""") == 0,
    "anti-fuite : nb_orders_90d = recomptage manuel (<= ref_date)":
        one("SELECT sum(nb_orders_90d) FROM gold.ml_features") ==
        one(f"""SELECT count(*) FROM silver.orders o
                WHERE o.date_commande > {REF} - 90 AND o.date_commande <= {REF}
                  AND o.customer_id IN (SELECT customer_id FROM gold.ml_features)"""),
    "anti-fuite : des commandes futures existent bien dans silver (et sont ignorées)":
        one(f"SELECT count(*) FROM silver.orders WHERE date_commande > {REF}") > 0,
    "étoile : fact_orders -> dim_customer":
        one("""SELECT count(*) FROM gold.fact_orders f
               LEFT JOIN gold.dim_customer d USING (customer_id) WHERE d.customer_id IS NULL""") == 0,
    "étoile : fact_orders -> dim_date":
        one("""SELECT count(*) FROM gold.fact_orders f
               LEFT JOIN gold.dim_date d USING (date_key) WHERE d.date_key IS NULL""") == 0,
    "étoile : fact_interactions -> dim_channel":
        one("""SELECT count(*) FROM gold.fact_interactions f
               LEFT JOIN gold.dim_channel d USING (channel_key) WHERE d.channel_key IS NULL""") == 0,
    "étoile : fact_subscriptions -> dim_plan":
        one("""SELECT count(*) FROM gold.fact_subscriptions f
               LEFT JOIN gold.dim_plan d USING (plan_key) WHERE d.plan_key IS NULL""") == 0,
}

# Détecteur de fuite : une feature seule ne doit pas "prédire" quasi parfaitement le churn
num_cols = [r[0] for r in con.execute("DESCRIBE gold.ml_features").fetchall()
            if r[1] in ("BIGINT", "INTEGER", "DOUBLE", "DECIMAL(38,2)", "DECIMAL(18,3)") or r[1].startswith("DECIMAL")]
num_cols = [c for c in num_cols if c != "churn"]
corrs = {c: one(f"SELECT corr({c}, churn) FROM gold.ml_features") for c in num_cols}
max_corr = max(abs(v) for v in corrs.values() if v is not None)
tests["anti-fuite : aucune feature n'a |corrélation| > 0.6 avec le churn"] = max_corr < 0.6

print("=== Tests ===")
for name, passed in tests.items():
    print(f"[{'OK' if passed else 'ECHEC'}] {name}")

print("\n=== Volumes gold ===")
for t in ["dim_date", "dim_customer", "dim_plan", "dim_channel", "fact_orders", "fact_complaints",
          "fact_interactions", "fact_subscriptions", "ml_population", "ml_features"]:
    print(f"{t:<20}{one(f'SELECT count(*) FROM gold.{t}'):>10,}")

print(f"\nPopulation ML : {n_feat:,} clients | churn = {rate:.1%}")
print("\n=== Corrélation de chaque feature avec le churn ===")
for c, v in sorted(corrs.items(), key=lambda kv: -abs(kv[1] or 0)):
    print(f"{c:<28}{v:>8.3f}")

print("\nRésultat global :", "TOUT EST OK" if all(tests.values()) else "DES TESTS ONT ECHOUE")