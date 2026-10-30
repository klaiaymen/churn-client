"""
generate_data.py : génère 5 tables CSV synthétiques pour le projet de prédiction du churn.

Principe anti-fuite de données (data leakage) :
  - REF_DATE = date de référence. Les features ML doivent être calculées AVANT cette date.
  - Label churn = 1 si la date_fin de l'abonnement tombe dans les 90 jours APRÈS REF_DATE.
  - Les événements postérieurs à REF_DATE existent dans les données brutes (comme dans la vraie vie)
    et c'est à vous de les exclure lors du calcul des features.
  - Le label n'est PAS exporté : vous le calculerez dans la couche Gold à partir de subscriptions.

Les données contiennent volontairement quelques défauts (doublons, casse incohérente, valeurs
manquantes, montants négatifs) à corriger dans la couche Silver.

Usage :  python generate_data.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------- Paramètres -----------------------------
SEED = 42
N_CUSTOMERS = 10_000
REF_DATE = pd.Timestamp("2025-06-30")
HORIZON_DAYS = 90
END_DATE = REF_DATE + pd.Timedelta(days=HORIZON_DAYS)
START_DATE = pd.Timestamp("2020-01-01")
LAST_SIGNUP = pd.Timestamp("2025-03-31")
HISTORY_DAYS = 540            # historique d'événements généré avant REF_DATE
TARGET_CHURN_RATE = 0.18      # taux de churn visé parmi les clients actifs à REF_DATE
OUT_DIR = Path("data/raw")

PLANS = {"Basic": 9.99, "Standard": 19.99, "Premium": 34.99}
VILLES = ["Tunis", "Sfax", "Sousse", "Nabeul", "Bizerte", "Gabes", "Monastir", "Kairouan"]
VILLE_P = [0.30, 0.20, 0.15, 0.08, 0.07, 0.07, 0.07, 0.06]

rng = np.random.default_rng(SEED)
n = N_CUSTOMERS


def d64(s):
    """Convertit une Series/Index de dates en tableau numpy datetime64[D]."""
    return pd.to_datetime(s).to_numpy().astype("datetime64[D]")


REF = np.datetime64(REF_DATE.date(), "D")
END = np.datetime64(END_DATE.date(), "D")


def gen_events(rate_per_month, start, end):
    """Tire des événements (processus de Poisson) entre start et end pour chaque client."""
    days = np.clip((end - start).astype("int64"), 0, None)
    counts = rng.poisson(rate_per_month * days / 30.0)
    idx = np.repeat(np.arange(len(counts)), counts)
    offs = (rng.random(len(idx)) * days[idx]).astype("int64")
    dates = start[idx] + offs.astype("timedelta64[D]")
    return idx, dates


# ----------------------------- Clients -----------------------------
cust = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(1, n + 1)]})
span = (LAST_SIGNUP - START_DATE).days
cust["date_inscription"] = START_DATE + pd.to_timedelta(rng.integers(0, span + 1, n), unit="D")
age = np.clip(rng.normal(38, 12, n), 18, 75).astype(int)
cust["date_naissance"] = cust["date_inscription"] - pd.to_timedelta(
    age * 365 + rng.integers(0, 365, n), unit="D"
)
cust["genre"] = rng.choice(["F", "M"], n)
cust["ville"] = rng.choice(VILLES, n, p=VILLE_P)
cust["segment"] = rng.choice(["Particulier", "Etudiant", "Pro"], n, p=[0.60, 0.15, 0.25])
cust["plan"] = rng.choice(list(PLANS), n, p=[0.45, 0.35, 0.20])

# Paramètres de comportement (cachés, non exportés)
disengaged = rng.random(n) < 0.20                 # client en train de se désengager
order_rate = rng.lognormal(0.0, 0.5, n)           # commandes / mois
complaint_rate = rng.gamma(2.0, 0.04, n)          # réclamations / mois
inter_rate = rng.lognormal(0.5, 0.5, n)           # interactions / mois

# Clients déjà partis AVANT la date de référence (exclus de la population d'entraînement)
days_to_ref = (REF_DATE - cust["date_inscription"]).dt.days.to_numpy()
past_churn = rng.random(n) < 0.12
end_past = cust["date_inscription"] + pd.to_timedelta(rng.integers(30, days_to_ref), unit="D")
cust["date_fin"] = end_past.where(past_churn)
active = ~past_churn                                # actifs à la date de référence

# ----------------------------- Historique (avant REF_DATE) -----------------------------
signup = d64(cust["date_inscription"])
end_hist = np.minimum(REF, d64(cust["date_fin"].fillna(REF_DATE)))
ws = np.maximum(signup, REF - np.timedelta64(HISTORY_DAYS, "D"))
mid = np.minimum(np.maximum(ws, REF - np.timedelta64(90, "D")), end_hist)
mult_recent = np.where(disengaged, 0.3, 1.0)       # baisse d'activité sur les 90 derniers jours


def history(rate, mult):
    iA, dA = gen_events(rate, ws, mid)
    iB, dB = gen_events(rate * mult, mid, end_hist)
    return np.concatenate([iA, iB]), np.concatenate([dA, dB])


o_idx, o_dates = history(order_rate, mult_recent)
i_idx, i_dates = history(inter_rate, mult_recent)
c_idx, c_dates = gen_events(complaint_rate * np.where(disengaged, 3.0, 1.0), ws, end_hist)
c_resolu = rng.random(len(c_idx)) < np.where(disengaged[c_idx], 0.50, 0.88)


def ago(dates):
    return (REF - dates).astype("int64")


def count_in(idx, ago_days, lo, hi, mask=None):
    m = (ago_days >= lo) & (ago_days < hi)
    if mask is not None:
        m &= mask
    return np.bincount(idx[m], minlength=n)


o_ago, i_ago, c_ago = ago(o_dates), ago(i_dates), ago(c_dates)
orders_90 = count_in(o_idx, o_ago, 0, 90)
inter_30 = count_in(i_idx, i_ago, 0, 30)
comp_180 = count_in(c_idx, c_ago, 0, 180)
unres_180 = count_in(c_idx, c_ago, 0, 180, mask=~c_resolu)
recency = np.full(n, 365.0)
s = pd.Series(o_ago).groupby(o_idx).min()
recency[s.index.to_numpy()] = s.to_numpy()
tenure = np.minimum(days_to_ref / 30, 36)

# ----------------------------- Règle de churn (observable dans les features) -----------------------------
z0 = (
    0.012 * np.minimum(recency, 180)
    + 0.80 * unres_180
    + 0.25 * comp_180
    - 0.20 * np.minimum(inter_30, 8)
    - 0.25 * np.minimum(orders_90, 6)
    + 0.60 * (cust["plan"].to_numpy() == "Basic")
    - 0.50 * (cust["plan"].to_numpy() == "Premium")
    - 0.02 * tenure
    + rng.normal(0, 0.5, n)                         # bruit : le churn n'est jamais parfaitement prévisible
)
lo, hi = -15.0, 5.0
for _ in range(60):                                  # calibration de l'intercept pour viser TARGET_CHURN_RATE
    b = (lo + hi) / 2
    p = (1 / (1 + np.exp(-(z0[active] + b)))).mean()
    lo, hi = (b, hi) if p < TARGET_CHURN_RATE else (lo, b)
churn = active & (rng.random(n) < 1 / (1 + np.exp(-(z0 + b))))

churn_date = REF_DATE + pd.to_timedelta(rng.integers(1, HORIZON_DAYS + 1, n), unit="D")
cust.loc[churn, "date_fin"] = churn_date[churn]

# ----------------------------- Événements après REF_DATE -----------------------------
fin = d64(cust["date_fin"].fillna(END_DATE))
end_post = np.minimum(END, fin)
start_post = np.full(n, REF)
act = active.astype(float)
mult_post = np.where(churn, 0.4, mult_recent)

o2_idx, o2_dates = gen_events(order_rate * mult_post * act, start_post, end_post)
i2_idx, i2_dates = gen_events(inter_rate * mult_post * act, start_post, end_post)
c2_idx, c2_dates = gen_events(
    complaint_rate * np.where(disengaged, 3.0, 1.0) * np.where(churn, 2.0, 1.0) * act,
    start_post, end_post,
)
c2_resolu = rng.random(len(c2_idx)) < np.where(churn[c2_idx], 0.55, 0.88)

o_idx, o_dates = np.concatenate([o_idx, o2_idx]), np.concatenate([o_dates, o2_dates])
i_idx, i_dates = np.concatenate([i_idx, i2_idx]), np.concatenate([i_dates, i2_dates])
c_idx, c_dates = np.concatenate([c_idx, c2_idx]), np.concatenate([c_dates, c2_dates])
c_resolu = np.concatenate([c_resolu, c2_resolu])

ids = cust["customer_id"].to_numpy()

# ----------------------------- Tables de sortie -----------------------------
orders = pd.DataFrame({
    "customer_id": ids[o_idx],
    "date_commande": pd.to_datetime(o_dates),
    "montant": np.round(rng.lognormal(3.3, 0.6, len(o_idx)), 2),
    "categorie": rng.choice(["Electronique", "Maison", "Mode", "Alimentation", "Loisirs"], len(o_idx)),
}).sort_values("date_commande", ignore_index=True)
orders.insert(0, "order_id", [f"O{i:07d}" for i in range(1, len(orders) + 1)])

complaints = pd.DataFrame({
    "customer_id": ids[c_idx],
    "date": pd.to_datetime(c_dates),
    "type": rng.choice(["Facturation", "Livraison", "Qualite", "Service client", "Technique"], len(c_idx)),
    "resolu": c_resolu,
}).sort_values("date", ignore_index=True)
complaints["delai_resolution"] = np.where(
    complaints["resolu"], np.round(rng.gamma(2.0, 2.5, len(complaints)) + 1), np.nan
)
complaints.insert(0, "complaint_id", [f"R{i:06d}" for i in range(1, len(complaints) + 1)])

interactions = pd.DataFrame({
    "customer_id": ids[i_idx],
    "date": pd.to_datetime(i_dates),
    "canal": rng.choice(["app", "mail", "appel", "chat"], len(i_idx), p=[0.5, 0.2, 0.1, 0.2]),
    "type": rng.choice(["connexion", "consultation", "support", "offre"], len(i_idx), p=[0.5, 0.25, 0.15, 0.10]),
}).sort_values("date", ignore_index=True)
interactions.insert(0, "interaction_id", [f"I{i:07d}" for i in range(1, len(interactions) + 1)])

# Abonnements : ~25 % des clients ont changé de plan une fois (historique de 2 lignes)
plans_now = cust["plan"].to_numpy()
elig = (days_to_ref > 240) & active & (rng.random(n) < 0.25)
change_date = cust["date_inscription"] + pd.to_timedelta(
    rng.integers(90, np.maximum(days_to_ref - 30, 91)), unit="D"
)
others = {p: [q for q in PLANS if q != p] for p in PLANS}
old_plan = np.array([rng.choice(others[p]) if e else "" for p, e in zip(plans_now, elig)])

cur = pd.DataFrame({
    "customer_id": ids,
    "plan": plans_now,
    "date_debut": np.where(elig, change_date, cust["date_inscription"]),
    "date_fin": cust["date_fin"],
})
cur["statut"] = np.where(cur["date_fin"].notna(), "resilie", "active")
old = pd.DataFrame({
    "customer_id": ids[elig],
    "plan": old_plan[elig],
    "date_debut": cust.loc[elig, "date_inscription"].to_numpy(),
    "date_fin": change_date[elig].to_numpy(),
    "statut": "remplace",
})
subs = pd.concat([old, cur], ignore_index=True).sort_values(["customer_id", "date_debut"], ignore_index=True)
subs["prix_mensuel"] = subs["plan"].map(PLANS)
subs.insert(0, "subscription_id", [f"S{i:06d}" for i in range(1, len(subs) + 1)])
subs = subs[["subscription_id", "customer_id", "plan", "prix_mensuel", "date_debut", "date_fin", "statut"]]

customers = cust[["customer_id", "date_naissance", "genre", "ville", "date_inscription", "segment"]].copy()

# ----------------------------- Défauts volontaires (pour la couche Silver) -----------------------------
m = rng.random(n)
customers.loc[m < 0.04, "ville"] = customers.loc[m < 0.04, "ville"].str.upper()
customers.loc[(m >= 0.04) & (m < 0.07), "ville"] = customers.loc[(m >= 0.04) & (m < 0.07), "ville"].str.lower()
customers.loc[rng.random(n) < 0.02, "ville"] = np.nan
customers.loc[rng.random(n) < 0.01, "genre"] = np.nan
subs.loc[rng.random(len(subs)) < 0.03, "plan"] = subs["plan"] + " "          # espaces parasites
dups = orders.sample(frac=0.005, random_state=SEED)
orders = pd.concat([orders, dups], ignore_index=True)                          # doublons exacts
neg = rng.random(len(orders)) < 0.002
orders.loc[neg, "montant"] = -orders.loc[neg, "montant"]                       # montants négatifs

# ----------------------------- Export -----------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
for name, df in [("customers", customers), ("subscriptions", subs), ("orders", orders),
                 ("complaints", complaints), ("interactions", interactions)]:
    df.to_csv(OUT_DIR / f"{name}.csv", index=False)
    print(f"{name:<14} {len(df):>9,} lignes")

print(f"\nClients actifs à {REF_DATE.date()} : {active.sum():,}")
print(f"Churn dans les {HORIZON_DAYS} jours suivants : {churn.sum():,} ({churn.sum() / active.sum():.1%})")
print(f"Fichiers écrits dans {OUT_DIR.resolve()}")