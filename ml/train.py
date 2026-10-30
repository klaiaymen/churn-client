"""
ml/train.py : entraîne 3 modèles de churn, les compare, choisit un seuil métier et explique le meilleur.

Lancer depuis la racine du projet :   python ml/train.py
Voir les expériences :                 mlflow ui --backend-store-uri sqlite:///mlflow.db

Sorties :
  models/churn_model.joblib      pipeline complet (prétraitement + modèle)
  models/model_meta.json         seuil, features, métriques (utilisé par score.py)
  reports/*.png / *.csv          courbes, comparaison, SHAP (pour le README)
  mlflow.db                      suivi des expériences + registre de modèles
"""
import json
import warnings
from pathlib import Path

import duckdb
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ----------------------------- Paramètres -----------------------------
ROOT = Path(__file__).resolve().parents[1]
SEED = 42
TARGET = "churn"
ID_COL = "customer_id"
CAT = ["segment", "ville", "genre", "plan"]

# Hypothèses métier pour choisir le seuil (à adapter et à justifier dans le README)
VALEUR_CLIENT = 150.0   # marge annuelle sauvée si le client reste (€)
TAUX_SUCCES = 0.30      # part des clients à risque contactés qui acceptent de rester
COUT_ACTION = 10.0      # coût d'une action de rétention par client contacté (€)

(ROOT / "models").mkdir(exist_ok=True)
(ROOT / "reports").mkdir(exist_ok=True)

# ----------------------------- Données -----------------------------
con = duckdb.connect(str(ROOT / "churn.duckdb"), read_only=True)
df = con.execute("SELECT * FROM gold.ml_features").df()
ref_date = str(con.execute("SELECT ref_date FROM gold.params").fetchone()[0])
con.close()

NUM = [c for c in df.columns if c not in CAT + [TARGET, ID_COL]]
df[NUM] = df[NUM].astype(float)
X, y = df[NUM + CAT], df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)
print(f"Train : {len(X_train):,} | Test : {len(X_test):,} | churn = {y.mean():.1%}")
print(f"PR-AUC d'un modèle au hasard = taux de churn = {y_test.mean():.3f}\n")


# ----------------------------- Modèles -----------------------------
def make_pipeline(model):
    prep = ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])
    return Pipeline([("prep", prep), ("model", model)])


spw = (y_train == 0).sum() / (y_train == 1).sum()
MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED),
    "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced_subsample",
                                            n_jobs=-1, random_state=SEED),
    "xgboost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss",
                             n_jobs=-1, random_state=SEED),
}


# ----------------------------- Aide : gain métier et seuil -----------------------------
def business_gain(y_true, proba, threshold):
    """Gain net (€) si on contacte tous les clients dont proba >= seuil."""
    contacted = proba >= threshold
    tp = int((contacted & (np.asarray(y_true) == 1)).sum())
    return tp * TAUX_SUCCES * VALEUR_CLIENT - int(contacted.sum()) * COUT_ACTION


def best_threshold(y_true, proba):
    grid = np.linspace(0.05, 0.95, 91)
    return float(grid[int(np.argmax([business_gain(y_true, proba, t) for t in grid]))])


def log_model(pipe):
    # cloudpickle : MLflow 3.x utilise skops par défaut, qui refuse les arbres (Random Forest).
    # Sans risque ici : c'est votre propre modèle, pas un fichier téléchargé.
    try:
        mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")
    except TypeError:                                   # anciennes versions de MLflow
        mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")


# ----------------------------- MLflow -----------------------------
mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
mlflow.set_experiment("churn")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
results, fitted, test_proba, run_ids = [], {}, {}, {}

for name, model in MODELS.items():
    pipe = make_pipeline(model)

    # 1) Validation croisée sur le TRAIN : sert à comparer les modèles et à choisir le seuil
    oof = cross_val_predict(pipe, X_train, y_train, cv=skf, method="predict_proba", n_jobs=1)[:, 1]
    cv_auc, cv_pr = roc_auc_score(y_train, oof), average_precision_score(y_train, oof)
    thr = best_threshold(y_train, oof)

    # 2) Entraînement final et évaluation sur le TEST (jamais vu)
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    m = {
        "cv_roc_auc": cv_auc, "cv_pr_auc": cv_pr,
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_recall": recall_score(y_test, pred),
        "test_precision": precision_score(y_test, pred, zero_division=0),
        "test_f1": f1_score(y_test, pred),
        "test_gain_eur": business_gain(y_test, proba, thr),
    }

    with mlflow.start_run(run_name=name) as run:
        params = {k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None}
        mlflow.log_params(params)
        mlflow.log_param("threshold", round(thr, 2))
        mlflow.log_metrics({k: float(v) for k, v in m.items()})
        log_model(pipe)
        run_ids[name] = run.info.run_id

    results.append({"model": name, "threshold": round(thr, 2), **m})
    fitted[name], test_proba[name] = pipe, proba
    print(f"{name:<22} CV PR-AUC {cv_pr:.3f} | TEST ROC-AUC {m['test_roc_auc']:.3f} "
          f"PR-AUC {m['test_pr_auc']:.3f} | rappel {m['test_recall']:.2f} préc. {m['test_precision']:.2f} "
          f"| seuil {thr:.2f} | gain {m['test_gain_eur']:,.0f} €")

res = pd.DataFrame(results).round(4)
res.to_csv(ROOT / "reports" / "model_comparison.csv", index=False)

# Choix du meilleur modèle sur la validation croisée (et non sur le test, pour ne pas "tricher")
best_name = res.sort_values("cv_pr_auc", ascending=False).iloc[0]["model"]
best_pipe, best_thr = fitted[best_name], float(res.set_index("model").loc[best_name, "threshold"])
best_proba = test_proba[best_name]
print(f"\n>>> Meilleur modèle (CV PR-AUC) : {best_name} | seuil = {best_thr:.2f}")

# Référence métier : que gagne-t-on par rapport à "contacter tout le monde" ?
print(f"Gain test : modèle {business_gain(y_test, best_proba, best_thr):,.0f} € | "
      f"contacter tout le monde {business_gain(y_test, best_proba, 0.0):,.0f} € | ne rien faire 0 €")

cm = confusion_matrix(y_test, (best_proba >= best_thr).astype(int))
print(f"Matrice de confusion (test) :\n  vrais négatifs {cm[0, 0]} | faux positifs {cm[0, 1]}\n"
      f"  faux négatifs  {cm[1, 0]} | vrais positifs {cm[1, 1]}")

# ----------------------------- Registre de modèles + sauvegarde -----------------------------
try:
    mlflow.register_model(f"runs:/{run_ids[best_name]}/model", "churn_model")
    print("Modèle enregistré dans le registre MLflow : churn_model")
except Exception as e:                                  # le registre ne doit pas bloquer l'entraînement
    print(f"(Registre MLflow ignoré : {e})")

joblib.dump(best_pipe, ROOT / "models" / "churn_model.joblib")
meta = {
    "model": best_name, "threshold": best_thr, "ref_date": ref_date,
    "num_features": NUM, "cat_features": CAT,
    "metrics": res.set_index("model").loc[best_name].to_dict(),
    "business_assumptions": {"valeur_client": VALEUR_CLIENT, "taux_succes": TAUX_SUCCES, "cout_action": COUT_ACTION},
}
(ROOT / "models" / "model_meta.json").write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")

# ----------------------------- Graphiques : ROC, PR, gain -----------------------------
fig, ax = plt.subplots(1, 3, figsize=(17, 4.8))
for name, p in test_proba.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    ax[0].plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_test, p)
    ax[1].plot(rc, pr, label=f"{name} ({average_precision_score(y_test, p):.3f})")
ax[0].plot([0, 1], [0, 1], "k--", lw=0.8); ax[0].set(title="Courbe ROC", xlabel="Taux de faux positifs", ylabel="Rappel")
ax[1].axhline(y_test.mean(), color="k", ls="--", lw=0.8); ax[1].set(title="Courbe Précision-Rappel", xlabel="Rappel", ylabel="Précision")
grid = np.linspace(0.05, 0.95, 91)
ax[2].plot(grid, [business_gain(y_test, best_proba, t) for t in grid])
ax[2].axvline(best_thr, color="r", ls="--", label=f"seuil retenu = {best_thr:.2f}")
ax[2].set(title=f"Gain métier selon le seuil ({best_name})", xlabel="Seuil", ylabel="Gain net (€)")
for a in ax: a.legend(); a.grid(alpha=0.3)
plt.tight_layout(); plt.savefig(ROOT / "reports" / "model_curves.png", dpi=130); plt.close()

# ----------------------------- Explicabilité SHAP (modèle retenu) -----------------------------
prep, mdl = best_pipe.named_steps["prep"], best_pipe.named_steps["model"]
names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
sample = prep.transform(X_test.sample(n=min(1000, len(X_test)), random_state=SEED))
if isinstance(mdl, LogisticRegression):
    explainer = shap.LinearExplainer(mdl, prep.transform(X_train.sample(n=100, random_state=SEED)))
else:
    explainer = shap.TreeExplainer(mdl)
sv = explainer.shap_values(sample)
if isinstance(sv, list):
    sv = sv[1]
if getattr(sv, "ndim", 2) == 3:
    sv = sv[:, :, 1]

for kind, fname in [(None, "shap_summary.png"), ("bar", "shap_importance.png")]:
    plt.figure()
    shap.summary_plot(sv, sample, feature_names=names, plot_type=kind, show=False, max_display=12)
    plt.tight_layout(); plt.savefig(ROOT / "reports" / fname, dpi=130, bbox_inches="tight"); plt.close()

imp = pd.Series(np.abs(sv).mean(axis=0), index=names).sort_values(ascending=False).head(8)
print("\nTop 8 facteurs de churn (importance SHAP moyenne) :")
print(imp.round(3).to_string())
print("\nFichiers écrits dans models/ et reports/ . Lancez : mlflow ui --backend-store-uri sqlite:///mlflow.db")