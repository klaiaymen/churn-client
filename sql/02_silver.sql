-- =====================================================================
-- COUCHE SILVER : données typées, dédoublonnées, standardisées.
-- Règle : on ne supprime jamais en silence. Les lignes invalides vont dans
-- silver.rejected_orders et chaque correction est comptée dans silver.data_quality_log.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS silver;

-- ---------------------------------------------------------------------
-- customers : typage, standardisation de la casse, valeurs manquantes
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE silver.customers AS
SELECT
    customer_id,
    TRY_CAST(date_naissance   AS DATE)                                  AS date_naissance,
    coalesce(nullif(trim(genre), ''), 'Inconnu')                        AS genre,
    coalesce(upper(left(trim(ville), 1)) || lower(substr(trim(ville), 2)), 'Inconnue') AS ville,
    TRY_CAST(date_inscription AS DATE)                                  AS date_inscription,
    trim(segment)                                                       AS segment
FROM bronze.customers
QUALIFY row_number() OVER (PARTITION BY customer_id ORDER BY _ingestion_ts) = 1;

-- ---------------------------------------------------------------------
-- subscriptions : espaces parasites, types, statut en minuscules
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE silver.subscriptions AS
SELECT
    subscription_id,
    customer_id,
    trim(plan)                          AS plan,
    TRY_CAST(prix_mensuel AS DECIMAL(6,2)) AS prix_mensuel,
    TRY_CAST(date_debut   AS DATE)      AS date_debut,
    TRY_CAST(date_fin     AS DATE)      AS date_fin,
    lower(trim(statut))                 AS statut
FROM bronze.subscriptions
QUALIFY row_number() OVER (PARTITION BY subscription_id ORDER BY _ingestion_ts) = 1;

-- ---------------------------------------------------------------------
-- orders : typage, puis séparation lignes valides / rejetées, puis déduplication
-- ---------------------------------------------------------------------
CREATE OR REPLACE TEMP TABLE tmp_orders_typed AS
SELECT
    o.order_id,
    o.customer_id,
    TRY_CAST(o.date_commande AS DATE)         AS date_commande,
    TRY_CAST(o.montant AS DECIMAL(10,2))      AS montant,
    trim(o.categorie)                         AS categorie,
    CASE
        WHEN TRY_CAST(o.date_commande AS DATE) IS NULL        THEN 'date_invalide'
        WHEN TRY_CAST(o.montant AS DECIMAL(10,2)) IS NULL
          OR TRY_CAST(o.montant AS DECIMAL(10,2)) <= 0        THEN 'montant_invalide'
        WHEN c.customer_id IS NULL                            THEN 'client_inconnu'
    END                                       AS motif_rejet
FROM bronze.orders o
LEFT JOIN silver.customers c USING (customer_id);

CREATE OR REPLACE TABLE silver.rejected_orders AS
SELECT order_id, customer_id, date_commande, montant, categorie, motif_rejet
FROM tmp_orders_typed
WHERE motif_rejet IS NOT NULL;

CREATE OR REPLACE TABLE silver.orders AS
SELECT order_id, customer_id, date_commande, montant, categorie
FROM tmp_orders_typed
WHERE motif_rejet IS NULL
QUALIFY row_number() OVER (PARTITION BY order_id ORDER BY date_commande) = 1;

-- ---------------------------------------------------------------------
-- complaints : booléen, délai entier (NULL si non résolue : c'est normal)
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE silver.complaints AS
SELECT
    complaint_id,
    customer_id,
    TRY_CAST(date AS DATE)                                  AS date,
    trim(type)                                              AS type,
    TRY_CAST(resolu AS BOOLEAN)                             AS resolu,
    CAST(TRY_CAST(delai_resolution AS DOUBLE) AS INTEGER)   AS delai_resolution
FROM bronze.complaints
QUALIFY row_number() OVER (PARTITION BY complaint_id ORDER BY _ingestion_ts) = 1;

-- ---------------------------------------------------------------------
-- interactions
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE silver.interactions AS
SELECT
    interaction_id,
    customer_id,
    TRY_CAST(date AS DATE) AS date,
    trim(canal)            AS canal,
    trim(type)             AS type
FROM bronze.interactions
QUALIFY row_number() OVER (PARTITION BY interaction_id ORDER BY _ingestion_ts) = 1;

-- ---------------------------------------------------------------------
-- Journal de qualité : une ligne par règle, avec le nombre de lignes concernées
-- ---------------------------------------------------------------------
CREATE OR REPLACE TABLE silver.data_quality_log AS
SELECT 'orders' AS table_name, 'doublons order_id' AS regle,
       (SELECT count(*) FROM tmp_orders_typed WHERE motif_rejet IS NULL)
       - (SELECT count(*) FROM silver.orders) AS nb_lignes, 'supprimes' AS action
UNION ALL
SELECT 'orders', 'montant nul, negatif ou illisible', count(*), 'rejetes (silver.rejected_orders)'
FROM tmp_orders_typed WHERE motif_rejet = 'montant_invalide'
UNION ALL
SELECT 'orders', 'date invalide', count(*), 'rejetes (silver.rejected_orders)'
FROM tmp_orders_typed WHERE motif_rejet = 'date_invalide'
UNION ALL
SELECT 'orders', 'client inconnu', count(*), 'rejetes (silver.rejected_orders)'
FROM tmp_orders_typed WHERE motif_rejet = 'client_inconnu'
UNION ALL
SELECT 'customers', 'ville manquante', count(*), 'remplacee par Inconnue'
FROM bronze.customers WHERE ville IS NULL OR trim(ville) = ''
UNION ALL
SELECT 'customers', 'ville avec casse incorrecte', count(*), 'standardisee'
FROM bronze.customers
WHERE ville IS NOT NULL AND trim(ville) <> ''
  AND ville <> upper(left(trim(ville), 1)) || lower(substr(trim(ville), 2))
UNION ALL
SELECT 'customers', 'genre manquant', count(*), 'remplace par Inconnu'
FROM bronze.customers WHERE genre IS NULL OR trim(genre) = ''
UNION ALL
SELECT 'subscriptions', 'plan avec espaces parasites', count(*), 'nettoye (trim)'
FROM bronze.subscriptions WHERE plan <> trim(plan)
UNION ALL
SELECT 'subscriptions', 'date_fin avant date_debut', count(*), 'a surveiller (non corrige)'
FROM silver.subscriptions WHERE date_fin < date_debut;

DROP TABLE IF EXISTS tmp_orders_typed;