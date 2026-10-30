-- =====================================================================
-- COUCHE BRONZE : copie brute des CSV, aucune transformation.
--  - all_varchar = true : tout est chargé en texte (le typage se fait en Silver)
--  - _ingestion_ts      : date/heure du chargement
--  - _source_file       : fichier d'origine (traçabilité)
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS bronze;

CREATE OR REPLACE TABLE bronze.customers AS
SELECT * EXCLUDE (filename),
       current_timestamp AS _ingestion_ts,
       filename          AS _source_file
FROM read_csv('data/raw/customers.csv', header = true, all_varchar = true, filename = true);

CREATE OR REPLACE TABLE bronze.subscriptions AS
SELECT * EXCLUDE (filename),
       current_timestamp AS _ingestion_ts,
       filename          AS _source_file
FROM read_csv('data/raw/subscriptions.csv', header = true, all_varchar = true, filename = true);

CREATE OR REPLACE TABLE bronze.orders AS
SELECT * EXCLUDE (filename),
       current_timestamp AS _ingestion_ts,
       filename          AS _source_file
FROM read_csv('data/raw/orders.csv', header = true, all_varchar = true, filename = true);

CREATE OR REPLACE TABLE bronze.complaints AS
SELECT * EXCLUDE (filename),
       current_timestamp AS _ingestion_ts,
       filename          AS _source_file
FROM read_csv('data/raw/complaints.csv', header = true, all_varchar = true, filename = true);

CREATE OR REPLACE TABLE bronze.interactions AS
SELECT * EXCLUDE (filename),
       current_timestamp AS _ingestion_ts,
       filename          AS _source_file
FROM read_csv('data/raw/interactions.csv', header = true, all_varchar = true, filename = true);