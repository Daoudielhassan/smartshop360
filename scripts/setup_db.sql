-- =============================================================================
-- SmartShop 360 — Script DDL Production PostgreSQL
-- =============================================================================
-- Usage direct :
--   psql -U admin -d smartshop_db -f scripts/setup_db.sql
--
-- Variables requises dans la DB cible :
--   POSTGRES_USER     = admin
--   POSTGRES_DB       = smartshop_db
-- =============================================================================

\set ON_ERROR_STOP on
\echo '[setup_db] == SmartShop 360 — Initialisation PostgreSQL =='

-- ---------------------------------------------------------------------------
-- 0. Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pg_trgm;       -- recherche floue sur VARCHAR
CREATE EXTENSION IF NOT EXISTS unaccent;      -- normalisation accents (optionnel)

\echo '[setup_db] Extensions OK'

-- ---------------------------------------------------------------------------
-- 1. Suppression des objets existants (ordre inverse des dépendances)
-- ---------------------------------------------------------------------------
DROP VIEW  IF EXISTS v_alerts        CASCADE;
DROP VIEW  IF EXISTS v_customer_kpi  CASCADE;
DROP VIEW  IF EXISTS v_product_kpi   CASCADE;
DROP VIEW  IF EXISTS v_data_quality  CASCADE;

DROP TABLE IF EXISTS review_facts    CASCADE;
DROP TABLE IF EXISTS sales_facts     CASCADE;
DROP TABLE IF EXISTS product_mapping CASCADE;
DROP TABLE IF EXISTS customers       CASCADE;
DROP TABLE IF EXISTS products        CASCADE;

\echo '[setup_db] Nettoyage des objets précédents OK'

-- ---------------------------------------------------------------------------
-- 2. Tables
-- ---------------------------------------------------------------------------

-- Catalogue produits ERP
CREATE TABLE products (
    "ProductID"   VARCHAR(50)  PRIMARY KEY,
    "ProductName" VARCHAR(255) NOT NULL,
    "Category"    VARCHAR(100)
);
COMMENT ON TABLE  products              IS 'Catalogue produits issu de l''ERP (Online Retail II)';
COMMENT ON COLUMN products."ProductID" IS 'StockCode ERP normalisé';

-- Clients
CREATE TABLE customers (
    "ClientID" VARCHAR(50)  PRIMARY KEY,
    "Nom"      VARCHAR(100),
    "Pays"     VARCHAR(100)           -- valeurs en ANGLAIS (source CSV)
);
COMMENT ON TABLE  customers          IS 'Référentiel clients';
COMMENT ON COLUMN customers."Pays"  IS 'Pays en anglais : United Kingdom, France, Germany...';

-- Mapping MDM (Golden Records)
CREATE TABLE product_mapping (
    "MappingID"          SERIAL       PRIMARY KEY,
    "ERP_StockCode"      VARCHAR(50)  NOT NULL,
    "ERP_ProductName"    VARCHAR(255),
    "Review_ProductCode" VARCHAR(50),
    "Review_ProductName" VARCHAR(255),
    "Category"           VARCHAR(100),
    "GoldenRecordName"   VARCHAR(255),
    "MatchScore"         FLOAT        NOT NULL DEFAULT 0.0,
    "MatchStrategy"      VARCHAR(20)  NOT NULL DEFAULT 'rank',
    "CreatedAt"          TIMESTAMP    NOT NULL DEFAULT NOW()
);
COMMENT ON TABLE product_mapping IS 'Mapping MDM ERP ↔ avis clients (P1-hard, P2-fuzzy, P3-sbert, P4-tfidf, P5-rank)';

-- Faits de ventes
CREATE TABLE sales_facts (
    "FactID"      SERIAL        PRIMARY KEY,
    "InvoiceNo"   VARCHAR(20)   NOT NULL,
    "StockCode"   VARCHAR(50)   NOT NULL,
    "Quantity"    INTEGER       NOT NULL DEFAULT 0,
    "Revenue"     NUMERIC(12,2) NOT NULL DEFAULT 0,
    "Margin"      NUMERIC(12,2) NOT NULL DEFAULT 0,
    "InvoiceDate" TIMESTAMP,
    "CustomerID"  VARCHAR(50),
    CONSTRAINT fk_sales_customer FOREIGN KEY ("CustomerID") REFERENCES customers("ClientID") ON DELETE SET NULL
);
COMMENT ON TABLE sales_facts IS 'Transactions de ventes (granularité ligne de facture)';

-- Faits d'avis clients
CREATE TABLE review_facts (
    "ReviewID"   SERIAL       PRIMARY KEY,
    "ProductID"  VARCHAR(50),
    "Rating"     NUMERIC(3,1) CHECK ("Rating" BETWEEN 1.0 AND 5.0),
    "ReviewText" TEXT,
    "Sentiment"  VARCHAR(20)  CHECK ("Sentiment" IN ('positive', 'negative', 'neutral')),
    "ReviewDate" TIMESTAMP
);
COMMENT ON TABLE review_facts IS 'Avis clients avec sentiment analysé';

\echo '[setup_db] Tables créées OK'

-- ---------------------------------------------------------------------------
-- 3. Index de performance
-- ---------------------------------------------------------------------------

-- product_mapping
CREATE INDEX idx_pm_erp_code    ON product_mapping ("ERP_StockCode");
CREATE INDEX idx_pm_review_code ON product_mapping ("Review_ProductCode");
CREATE INDEX idx_pm_strategy    ON product_mapping ("MatchStrategy");

-- sales_facts
CREATE INDEX idx_sf_stockcode   ON sales_facts ("StockCode");
CREATE INDEX idx_sf_customerid  ON sales_facts ("CustomerID");
CREATE INDEX idx_sf_invoicedate ON sales_facts ("InvoiceDate");
CREATE INDEX idx_sf_invoiceno   ON sales_facts ("InvoiceNo");

-- review_facts
CREATE INDEX idx_rf_productid   ON review_facts ("ProductID");
CREATE INDEX idx_rf_sentiment   ON review_facts ("Sentiment");
CREATE INDEX idx_rf_rating      ON review_facts ("Rating");

-- customers
CREATE INDEX idx_cust_pays      ON customers ("Pays");

-- Recherche floue sur noms (pg_trgm)
CREATE INDEX idx_pm_golden_trgm ON product_mapping USING gin ("GoldenRecordName" gin_trgm_ops);
CREATE INDEX idx_pm_erp_name_trgm ON product_mapping USING gin ("ERP_ProductName" gin_trgm_ops);

\echo '[setup_db] Index créés OK'

-- ---------------------------------------------------------------------------
-- 4. Vues analytiques
-- ---------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_product_kpi AS
SELECT
    pm."ERP_StockCode"                                        AS "ProductID",
    pm."GoldenRecordName"                                     AS "ProductName",
    pm."Category",
    ROUND(COALESCE(SUM(sf."Revenue"), 0)::NUMERIC, 2)         AS "CA",
    ROUND(COALESCE(SUM(sf."Margin"),  0)::NUMERIC, 2)         AS "Marge",
    COALESCE(SUM(sf."Quantity"), 0)                           AS "QuantiteVendue",
    ROUND(AVG(rf."Rating")::NUMERIC, 2)                       AS "Notemoyenne",
    COUNT(rf."ReviewID")                                      AS "NbAvis",
    SUM(CASE WHEN rf."Sentiment" = 'positive' THEN 1 ELSE 0 END) AS "AvisPositifs",
    SUM(CASE WHEN rf."Sentiment" = 'negative' THEN 1 ELSE 0 END) AS "AvisNegatifs",
    SUM(CASE WHEN rf."Sentiment" = 'neutral'  THEN 1 ELSE 0 END) AS "AvisNeutres"
FROM product_mapping pm
LEFT JOIN sales_facts  sf ON sf."StockCode"  = pm."ERP_StockCode"
LEFT JOIN review_facts rf ON rf."ProductID"  = pm."Review_ProductCode"
GROUP BY pm."ERP_StockCode", pm."GoldenRecordName", pm."Category";

COMMENT ON VIEW v_product_kpi IS 'KPIs produits : CA, marge, ventes, notes, sentiments';


CREATE OR REPLACE VIEW v_customer_kpi AS
SELECT
    c."ClientID",
    c."Nom",
    c."Pays",
    COUNT(DISTINCT sf."InvoiceNo")                           AS "NbCommandes",
    COUNT(DISTINCT sf."StockCode")                           AS "NbProduits",
    ROUND(SUM(sf."Revenue")::NUMERIC, 2)                     AS "CA_Total",
    ROUND(AVG(sf."Revenue")::NUMERIC, 2)                     AS "PanierMoyen",
    ROUND(AVG(rf."Rating")::NUMERIC, 2)                      AS "Notemoyenne",
    COUNT(rf."ReviewID")                                     AS "NbAvis"
FROM customers c
LEFT JOIN sales_facts    sf ON sf."CustomerID"   = c."ClientID"
LEFT JOIN product_mapping pm ON pm."ERP_StockCode" = sf."StockCode"
LEFT JOIN review_facts   rf ON rf."ProductID"    = pm."Review_ProductCode"
GROUP BY c."ClientID", c."Nom", c."Pays";

COMMENT ON VIEW v_customer_kpi IS 'KPIs clients : commandes, CA, panier moyen, note moyenne';


CREATE OR REPLACE VIEW v_alerts AS
SELECT
    p."ProductID",
    p."ProductName",
    p."Category",
    p."CA",
    p."Notemoyenne",
    p."NbAvis",
    p."AvisNegatifs",
    p."QuantiteVendue",
    CASE
        WHEN p."Notemoyenne" < 3.0 AND p."QuantiteVendue" > 50 THEN 'CRITIQUE'
        WHEN p."Notemoyenne" < 3.5                              THEN 'A_SURVEILLER'
        ELSE 'OK'
    END AS "Statut"
FROM v_product_kpi p
WHERE p."NbAvis" > 0
ORDER BY p."Notemoyenne" ASC;

COMMENT ON VIEW v_alerts IS 'Alertes produits : CRITIQUE / A_SURVEILLER / OK';


CREATE OR REPLACE VIEW v_data_quality AS
SELECT
    (SELECT COUNT(*)                    FROM products)                                                AS "Nb_Produits_ERP",
    (SELECT COUNT(DISTINCT "ProductID") FROM review_facts)                                            AS "Nb_Produits_Avis",
    (SELECT COUNT(*)                    FROM product_mapping)                                         AS "Nb_Mappings",
    (SELECT COUNT(*)                    FROM product_mapping WHERE "MatchStrategy" = 'hard')          AS "Nb_Hard_Matches",
    (SELECT COUNT(*)                    FROM product_mapping WHERE "MatchStrategy" = 'fuzzy')         AS "Nb_Fuzzy_Matches",
    (SELECT COUNT(*)                    FROM product_mapping WHERE "MatchStrategy" = 'tfidf')         AS "Nb_TFIDF_Matches",
    (SELECT COUNT(*)                    FROM product_mapping WHERE "MatchStrategy" = 'sbert')         AS "Nb_SBERT_Matches",
    (SELECT COUNT(*)                    FROM review_facts)                                            AS "Nb_Avis_Total",
    (SELECT COUNT(*)                    FROM review_facts    WHERE "ProductID" IS NOT NULL)           AS "Nb_Avis_Lies",
    (SELECT COUNT(DISTINCT "InvoiceNo") FROM sales_facts)                                             AS "Nb_Factures",
    (SELECT COUNT(*)                    FROM customers)                                               AS "Nb_Clients",
    ROUND(
        100.0
        * (SELECT COUNT(DISTINCT rf."ProductID")
           FROM review_facts rf
           WHERE rf."ProductID" IN (SELECT "Review_ProductCode" FROM product_mapping))
        / GREATEST((SELECT COUNT(*) FROM product_mapping), 1),
    1)                                                                                                AS "Taux_Couverture_MDM";

COMMENT ON VIEW v_data_quality IS 'Tableau de bord qualité données MDM';

\echo '[setup_db] Vues analytiques créées OK'

-- ---------------------------------------------------------------------------
-- 5. Rôles & droits (optionnel — décommenter en prod réelle)
-- ---------------------------------------------------------------------------
-- CREATE ROLE smartshop_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO smartshop_readonly;
-- GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO smartshop_readonly;

-- ---------------------------------------------------------------------------
-- 6. Vérification finale
-- ---------------------------------------------------------------------------
\echo ''
\echo '[setup_db] == Vérification =='
SELECT
    schemaname,
    tablename  AS "objet",
    'TABLE'    AS "type"
FROM pg_tables
WHERE schemaname = 'public'
  AND tablename IN ('products','customers','product_mapping','sales_facts','review_facts')
UNION ALL
SELECT
    schemaname,
    viewname,
    'VIEW'
FROM pg_views
WHERE schemaname = 'public'
  AND viewname IN ('v_product_kpi','v_customer_kpi','v_alerts','v_data_quality')
ORDER BY "type", "objet";

\echo ''
\echo '[setup_db] == Initialisation terminée avec succès =='
