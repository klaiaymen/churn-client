"""
ml/score.py : calcule le risque de churn de chaque client actif et l'explique.

Lancer depuis la racine du projet (après train.py) :   python ml/score.py

Sorties :
  gold.churn_scores              1 ligne par client : probabilité, niveau de risque, 3 facteurs en texte
  gold.churn_feature_importance  importance globale de chaque facteur (pour la page "Facteurs" de Power BI)
  exports/*.csv                  fichiers à importer dans Power BI Desktop
"""
import json
import warnings
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
SEED = 42
MIN_SHAP = 0.02            # un facteur n'est affiché que s'il pousse réellement vers le churn
MEDIUM_RATIO = 0.6         # "Moyen" = entre 60 % du seuil et le seuil

# ----------------------------- Modèle et données -----------------------------
meta = json.loads((ROOT / "models" / "model_meta.json").read_text(encoding="utf-8"))
pipe = joblib.load(ROOT / "models" / "churn_model.joblib")
NUM, CAT, THR = meta["num_features"], meta["cat_features"], float(meta["threshold"])

con = duckdb.connect(str(ROOT / "churn.duckdb"))
df = con.execute("SELECT * FROM gold.ml_features").df()
df[NUM] = df[NUM].astype(float)
X = df[NUM + CAT]
print(f"Scoring de {len(df):,} clients actifs | modèle : {meta['model']} | seuil : {THR:.2f}")

# ----------------------------- Probabilités et niveaux de risque -----------------------------
proba = pipe.predict_proba(X)[:, 1]
level = np.where(proba >= THR, "Élevé", np.where(proba >= MEDIUM_RATIO * THR, "Moyen", "Faible"))
level_order = pd.Series(level).map({"Faible": 1, "Moyen": 2, "Élevé": 3}).to_numpy()

# ----------------------------- SHAP par client -----------------------------
prep, mdl = pipe.named_steps["prep"], pipe.named_steps["model"]
Xt = prep.transform(X)
names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
if isinstance(mdl, LogisticRegression):
    rng = np.random.default_rng(SEED)
    background = Xt[rng.choice(len(Xt), size=min(100, len(Xt)), replace=False)]
    sv = shap.LinearExplainer(mdl, background).shap_values(Xt)
else:
    sv = shap.TreeExplainer(mdl).shap_values(Xt)
if isinstance(sv, list):
    sv = sv[1]
if getattr(sv, "ndim", 2) == 3:
    sv = sv[:, :, 1]

# On regroupe les colonnes one-hot (plan_Basic, plan_Premium...) sous leur variable d'origine (plan)
orig = np.array([next((c for c in CAT if n.startswith(c + "_")), n) for n in names])
feats = list(dict.fromkeys(orig))
sv_agg = np.column_stack([sv[:, orig == f].sum(axis=1) for f in feats])

# Phrases lisibles (formulation neutre : n'est affichée que si la valeur pousse vers le churn)
DESCRIBE = {
    "recency_days":             lambda v: f"{v:.0f} jours depuis le dernier achat",
    "nb_orders_90d":            lambda v: f"{v:.0f} commande(s) sur 90 jours",
    "nb_orders_prev_90d":       lambda v: f"{v:.0f} commande(s) sur les 90 jours précédents",
    "orders_trend":             lambda v: f"évolution des commandes de {v:+.0f} vs période précédente",
    "total_spent_180d":         lambda v: f"{v:.0f} € dépensés sur 6 mois",
    "avg_order_amount_180d":    lambda v: f"panier moyen de {v:.0f} €",
    "nb_complaints_180d":       lambda v: f"{v:.0f} réclamation(s) sur 6 mois",
    "nb_unresolved_180d":       lambda v: f"{v:.0f} réclamation(s) non résolue(s)",
    "avg_resolution_days":      lambda v: f"délai moyen de résolution de {v:.0f} jours",
    "nb_interactions_30d":      lambda v: f"{v:.0f} interaction(s) sur 30 jours",
    "nb_interactions_prev_30d": lambda v: f"{v:.0f} interaction(s) le mois précédent",
    "interactions_trend":       lambda v: f"évolution des interactions de {v:+.0f} vs mois précédent",
    "support_share_90d":        lambda v: f"{v:.0%} d'interactions avec le support",
    "tenure_months":            lambda v: f"ancienneté de {v:.0f} mois",
    "prix_mensuel":             lambda v: f"abonnement à {v:.2f} €/mois",
    "nb_plan_changes":          lambda v: f"{v:.0f} changement(s) de plan",
    "age":                      lambda v: f"âge de {v:.0f} ans",
    "plan":                     lambda v: f"plan {v}",
    "segment":                  lambda v: f"segment {v}",
    "ville":                    lambda v: f"ville {v}",
    "genre":                    lambda v: f"genre {v}",
}


def describe(feat, value):
    fn = DESCRIBE.get(feat)
    return fn(value) if fn else f"{feat} = {value}"


top_idx = np.argsort(-sv_agg, axis=1)[:, :3]
factor_cols = {1: [], 2: [], 3: []}
for i in range(len(df)):
    for k in range(3):
        j = top_idx[i, k]
        factor_cols[k + 1].append(describe(feats[j], X.iloc[i][feats[j]]) if sv_agg[i, j] > MIN_SHAP else None)

# ----------------------------- Table de scores -----------------------------
test_file = ROOT / "models" / "test_customer_ids.csv"
test_ids = set(pd.read_csv(test_file)["customer_id"]) if test_file.exists() else set()
split = np.where(df["customer_id"].isin(test_ids), "test", "entrainement") if test_ids else "inconnu"

scores = pd.DataFrame({
    "customer_id": df["customer_id"],
    "churn_probability": proba.round(4),
    "risk_level": level,
    "risk_order": level_order,
    "risk_rank": pd.Series(proba).rank(ascending=False, method="first").astype(int),
    "top_factor_1": factor_cols[1],
    "top_factor_2": factor_cols[2],
    "top_factor_3": factor_cols[3],
})
scores["top_factors"] = scores[["top_factor_1", "top_factor_2", "top_factor_3"]].apply(
    lambda r: " ; ".join(v for v in r if isinstance(v, str)), axis=1)
scores["plan"] = df["plan"]
scores["prix_mensuel"] = df["prix_mensuel"]
scores["ville"] = df["ville"]
scores["segment"] = df["segment"]
scores["revenu_mensuel_en_danger"] = np.where(level == "Élevé", df["prix_mensuel"], 0.0)
scores["churn_observe"] = df["churn"]          # connu ici car les données sont historiques ; inconnu en production
scores["jeu_de_donnees"] = split               # "test" = scores jamais vus par le modèle à l'entraînement
scores["model_name"] = meta["model"]
scores["reference_date"] = meta["ref_date"]
scores = scores.sort_values("risk_rank").reset_index(drop=True)

importance = (pd.DataFrame({"feature": feats, "importance": np.abs(sv_agg).mean(axis=0).round(4)})
              .sort_values("importance", ascending=False).reset_index(drop=True))
importance["rank"] = importance.index + 1

# ----------------------------- Écriture dans DuckDB + exports Power BI -----------------------------
con.register("scores_df", scores)
con.execute("CREATE OR REPLACE TABLE gold.churn_scores AS SELECT * FROM scores_df")
con.register("importance_df", importance)
con.execute("CREATE OR REPLACE TABLE gold.churn_feature_importance AS SELECT * FROM importance_df")

(ROOT / "exports").mkdir(exist_ok=True)
for table in ["churn_scores", "churn_feature_importance", "dim_customer", "dim_plan", "dim_date",
              "fact_subscriptions", "fact_orders", "fact_complaints"]:
    con.execute(f"SELECT * FROM gold.{table}").df().to_csv(
        ROOT / "exports" / f"{table}.csv", index=False, encoding="utf-8-sig")
con.close()

# ----------------------------- Résumé -----------------------------
print("\nRépartition des niveaux de risque :")
summary = scores.groupby("risk_level").agg(clients=("customer_id", "count"),
                                           churn_observe=("churn_observe", "mean")).reindex(["Élevé", "Moyen", "Faible"])
summary["churn_observe"] = (summary["churn_observe"] * 100).round(1).astype(str) + " %"
print(summary.to_string())
print(f"\nRevenu mensuel en danger (risque Élevé) : {scores['revenu_mensuel_en_danger'].sum():,.0f} €")

if test_ids:
    t = scores[scores["jeu_de_donnees"] == "test"]
    pred, real = t["risk_level"] == "Élevé", t["churn_observe"] == 1
    print(f"Sur les clients du TEST ({len(t):,}) : rappel {(pred & real).sum() / real.sum():.2f} | "
          f"précision {(pred & real).sum() / max(pred.sum(), 1):.2f}")

print("\nTop 5 clients à risque :")
print(scores.head(5)[["customer_id", "churn_probability", "risk_level", "top_factors"]].to_string(index=False))
print(f"\nTable gold.churn_scores écrite. CSV exportés dans {ROOT / 'exports'}")