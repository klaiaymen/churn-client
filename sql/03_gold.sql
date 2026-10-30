-- =====================================================================
-- COUCHE GOLD : modèle dimensionnel (schéma en étoile) + table de features ML.
--
-- RÈGLE ANTI-FUITE DE DONNÉES :
--   * ref_date = date de "prédiction". Tout ce qui sert de feature date d'AVANT (<=) ref_date.
--   * Le label churn = 1 si l'abonnement se termine dans les horizon_days jours APRÈS ref_date.
-- Pour changer de date de référence, modifiez UNIQUEMENT gold.params.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS gold;

CREATE OR REPLACE TABLE gold.params AS
SELECT DATE '2025-06-30' AS ref_date, 90 AS horizon_days;

-- ---------------------------------------------------------------------
-- DIMENSIONS
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE gold.dim_date AS
SELECT
    CAST(strftime(d, '%Y%m%d') AS INTEGER) AS date_key,
    CAST(d AS DATE)                        AS date,
    year(d)                                AS annee,
    quarter(d)                             AS trimestre,
    month(d)                               AS mois,
    strftime(d, '%Y-%m')                   AS annee_mois,
    monthname(d)                           AS nom_mois,
    isodow(d)                              AS jour_semaine,
    isodow(d) IN (6, 7)                    AS est_weekend
FROM generate_series(TIMESTAMP '2020-01-01', TIMESTAMP '2025-12-31', INTERVAL 1 DAY) AS t(d);

CREATE OR REPLACE TABLE gold.dim_customer AS
SELECT
    c.customer_id,
    c.genre,
    c.ville,
    c.segment,
    c.date_naissance,
    date_diff('year', c.date_naissance, (SELECT ref_date FROM gold.params)) AS age,
    CASE
        WHEN date_diff('year', c.date_naissance, (SELECT ref_date FROM gold.params)) < 25 THEN '18-24'
        WHEN date_diff('year', c.date_naissance, (SELECT ref_date FROM gold.params)) < 35 THEN '25-34'
        WHEN date_diff('year', c.date_naissance, (SELECT ref_date FROM gold.params)) < 50 THEN '35-49'
        ELSE '50+'
    END AS tranche_age,
    c.date_inscription
FROM silver.customers c;

CREATE OR REPLACE TABLE gold.dim_plan AS
SELECT
    row_number() OVER (ORDER BY max(prix_mensuel)) AS plan_key,
    plan,
    max(prix_mensuel) AS prix_mensuel
FROM silver.subscriptions
GROUP BY plan;

CREATE OR REPLACE TABLE gold.dim_channel AS
SELECT row_number() OVER (ORDER BY canal) AS channel_key, canal
FROM (SELECT DISTINCT canal FROM silver.interactions);

-- ---------------------------------------------------------------------
-- TABLES DE FAITS
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE gold.fact_orders AS
SELECT
    order_id,
    customer_id,
    CAST(strftime(date_commande, '%Y%m%d') AS INTEGER) AS date_key,
    montant,
    categorie
FROM silver.orders;

CREATE OR REPLACE TABLE gold.fact_complaints AS
SELECT
    complaint_id,
    customer_id,
    CAST(strftime(date, '%Y%m%d') AS INTEGER) AS date_key,
    type,
    resolu,
    delai_resolution
FROM silver.complaints;

CREATE OR REPLACE TABLE gold.fact_interactions AS
SELECT
    i.interaction_id,
    i.customer_id,
    CAST(strftime(i.date, '%Y%m%d') AS INTEGER) AS date_key,
    ch.channel_key,
    i.type
FROM silver.interactions i
JOIN gold.dim_channel ch ON ch.canal = i.canal;

CREATE OR REPLACE TABLE gold.fact_subscriptions AS
SELECT
    s.subscription_id,
    s.customer_id,
    p.plan_key,
    CAST(strftime(s.date_debut, '%Y%m%d') AS INTEGER) AS date_debut_key,
    CAST(strftime(s.date_fin,   '%Y%m%d') AS INTEGER) AS date_fin_key,
    s.date_debut,
    s.date_fin,
    s.prix_mensuel,
    s.statut
FROM silver.subscriptions s
JOIN gold.dim_plan p ON p.plan = s.plan;

-- ---------------------------------------------------------------------
-- POPULATION ML + LABEL : clients dont l'abonnement est actif à ref_date
-- (les clients déjà partis avant ref_date sont exclus : on ne prédit pas le passé)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE gold.ml_population AS
SELECT
    s.customer_id,
    s.plan,
    s.prix_mensuel,
    CASE WHEN s.date_fin > p.ref_date
          AND s.date_fin <= p.ref_date + p.horizon_days THEN 1 ELSE 0 END AS churn
FROM silver.subscriptions s
CROSS JOIN gold.params p
WHERE s.date_debut <= p.ref_date
  AND (s.date_fin IS NULL OR s.date_fin > p.ref_date);

-- ---------------------------------------------------------------------
-- FEATURES ML : une ligne par client, calculées uniquement avec des données <= ref_date
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE gold.ml_features AS
WITH p AS (SELECT ref_date FROM gold.params),
ord AS (
    SELECT o.customer_id,
        count(*) FILTER (WHERE o.date_commande > p.ref_date - 90)                                   AS nb_orders_90d,
        count(*) FILTER (WHERE o.date_commande <= p.ref_date - 90
                           AND o.date_commande >  p.ref_date - 180)                                 AS nb_orders_prev_90d,
        avg(o.montant) FILTER (WHERE o.date_commande > p.ref_date - 180)                            AS avg_order_amount_180d,
        sum(o.montant) FILTER (WHERE o.date_commande > p.ref_date - 180)                            AS total_spent_180d,
        max(o.date_commande)                                                                        AS last_order_date
    FROM silver.orders o CROSS JOIN p
    WHERE o.date_commande <= p.ref_date                       -- << anti-fuite
    GROUP BY o.customer_id
),
cmp AS (
    SELECT c.customer_id,
        count(*) AS nb_complaints_180d,
        -- une réclamation est "non résolue" si elle ne l'était pas ENCORE à ref_date
        count(*) FILTER (WHERE NOT coalesce(c.resolu, false)
                            OR c.date + c.delai_resolution > p.ref_date) AS nb_unresolved_180d,
        avg(c.delai_resolution) FILTER (WHERE c.resolu
                            AND c.date + c.delai_resolution <= p.ref_date) AS avg_resolution_days
    FROM silver.complaints c CROSS JOIN p
    WHERE c.date <= p.ref_date AND c.date > p.ref_date - 180  -- << anti-fuite
    GROUP BY c.customer_id
),
itr AS (
    SELECT i.customer_id,
        count(*) FILTER (WHERE i.date > p.ref_date - 30)                                   AS nb_interactions_30d,
        count(*) FILTER (WHERE i.date <= p.ref_date - 30 AND i.date > p.ref_date - 60)     AS nb_interactions_prev_30d,
        count(*)                                                                           AS nb_interactions_90d,
        count(*) FILTER (WHERE i.type = 'support')                                         AS nb_support_90d
    FROM silver.interactions i CROSS JOIN p
    WHERE i.date <= p.ref_date AND i.date > p.ref_date - 90   -- << anti-fuite
    GROUP BY i.customer_id
),
chg AS (
    SELECT s.customer_id, count(*) - 1 AS nb_plan_changes
    FROM silver.subscriptions s CROSS JOIN p
    WHERE s.date_debut <= p.ref_date                          -- << anti-fuite
    GROUP BY s.customer_id
)
SELECT
    pop.customer_id,
    -- profil
    c.segment,
    c.ville,
    c.genre,
    date_diff('year',  c.date_naissance,   p.ref_date)                       AS age,
    date_diff('month', c.date_inscription, p.ref_date)                       AS tenure_months,
    -- abonnement
    pop.plan,
    pop.prix_mensuel,
    coalesce(chg.nb_plan_changes, 0)                                         AS nb_plan_changes,
    -- achats
    coalesce(ord.nb_orders_90d, 0)                                           AS nb_orders_90d,
    coalesce(ord.nb_orders_prev_90d, 0)                                      AS nb_orders_prev_90d,
    coalesce(ord.nb_orders_90d, 0) - coalesce(ord.nb_orders_prev_90d, 0)     AS orders_trend,
    coalesce(ord.avg_order_amount_180d, 0)                                   AS avg_order_amount_180d,
    coalesce(ord.total_spent_180d, 0)                                        AS total_spent_180d,
    least(coalesce(date_diff('day', ord.last_order_date, p.ref_date), 365), 365) AS recency_days,
    -- réclamations
    coalesce(cmp.nb_complaints_180d, 0)                                      AS nb_complaints_180d,
    coalesce(cmp.nb_unresolved_180d, 0)                                      AS nb_unresolved_180d,
    coalesce(cmp.avg_resolution_days, 0)                                     AS avg_resolution_days,
    -- interactions
    coalesce(itr.nb_interactions_30d, 0)                                     AS nb_interactions_30d,
    coalesce(itr.nb_interactions_prev_30d, 0)                                AS nb_interactions_prev_30d,
    coalesce(itr.nb_interactions_30d, 0) - coalesce(itr.nb_interactions_prev_30d, 0) AS interactions_trend,
    CASE WHEN coalesce(itr.nb_interactions_90d, 0) = 0 THEN 0
         ELSE itr.nb_support_90d * 1.0 / itr.nb_interactions_90d END         AS support_share_90d,
    -- cible
    pop.churn
FROM gold.ml_population pop
JOIN silver.customers c ON c.customer_id = pop.customer_id
CROSS JOIN p
LEFT JOIN ord ON ord.customer_id = pop.customer_id
LEFT JOIN cmp ON cmp.customer_id = pop.customer_id
LEFT JOIN itr ON itr.customer_id = pop.customer_id
LEFT JOIN chg ON chg.customer_id = pop.customer_id;