"""
src/etl/run_etl.py
===================
Script principal d'ingestion — orchestre le pipeline complet :
  1. Nettoyage des sources (cleaning.py)
  2. Création du mapping MDM (mdm_mapping.py)
  3. Chargement dans PostgreSQL (via SQLAlchemy)
  4. Création des vues analytiques

Usage :
    python -m src.etl.run_etl
"""

import os
import sys
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Compatibilité exécution locale (si src/ pas dans PYTHONPATH)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.etl.cleaning import (
    load_transactions,
    clean_transactions,
    extract_top50_products,
    extract_customers,
    load_reviews,
    CSV_PATH,
    JSON_PATH,
)
from src.etl.mdm_mapping import build_product_mapping, attach_product_to_reviews
from src.etl.incremental import should_run_etl, record_hashes
from src.etl.data_quality import run_all_validations
from src.db_config import get_engine, test_connection

ETL_SOURCE_FILES = [CSV_PATH, JSON_PATH]


# ────────────────────────────────────────────────────────────
#  DDL — Création des tables
# ────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS product_mapping (
    "MappingID"           SERIAL PRIMARY KEY,
    "ERP_StockCode"       VARCHAR(50),
    "ERP_ProductName"     VARCHAR(255),
    "Review_ProductCode"  VARCHAR(50),
    "Review_ProductName"  VARCHAR(255),
    "Category"            VARCHAR(100),
    "GoldenRecordName"    VARCHAR(255),
    "MatchScore"          FLOAT DEFAULT 0.0,
    "MatchStrategy"       VARCHAR(20) DEFAULT 'rank'
);

CREATE TABLE IF NOT EXISTS products (
    "ProductID"    VARCHAR(50) PRIMARY KEY,
    "ProductName"  VARCHAR(255),
    "Category"     VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS customers (
    "ClientID"  VARCHAR(50) PRIMARY KEY,
    "Nom"       VARCHAR(100),
    "Pays"      VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS sales_facts (
    "FactID"       SERIAL PRIMARY KEY,
    "InvoiceNo"    VARCHAR(20),
    "StockCode"    VARCHAR(50),
    "Quantity"     INTEGER,
    "Revenue"      NUMERIC(12,2),
    "Margin"       NUMERIC(12,2),
    "InvoiceDate"  TIMESTAMP,
    "CustomerID"   VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS review_facts (
    "ReviewID"    SERIAL PRIMARY KEY,
    "ProductID"   VARCHAR(50),
    "Rating"      NUMERIC(3,1),
    "ReviewText"  TEXT,
    "Sentiment"   VARCHAR(20),
    "ReviewDate"  TIMESTAMP
);
"""

VIEWS_SQL = [
    """
    CREATE OR REPLACE VIEW v_product_kpi AS
    SELECT
        pm."ERP_StockCode"                                       AS "ProductID",
        pm."GoldenRecordName"                                    AS "ProductName",
        pm."Category",
        ROUND(COALESCE(SUM(sf."Revenue"), 0)::NUMERIC, 2)       AS "CA",
        ROUND(COALESCE(SUM(sf."Margin"),  0)::NUMERIC, 2)       AS "Marge",
        COALESCE(SUM(sf."Quantity"), 0)                         AS "QuantiteVendue",
        ROUND(AVG(rf."Rating")::NUMERIC, 2)                     AS "Notemoyenne",
        COUNT(rf."ReviewID")                                     AS "NbAvis",
        SUM(CASE WHEN rf."Sentiment" = 'positive' THEN 1 ELSE 0 END) AS "AvisPositifs",
        SUM(CASE WHEN rf."Sentiment" = 'negative' THEN 1 ELSE 0 END) AS "AvisNegatifs",
        SUM(CASE WHEN rf."Sentiment" = 'neutral'  THEN 1 ELSE 0 END) AS "AvisNeutres"
    FROM product_mapping pm
    LEFT JOIN sales_facts  sf ON sf."StockCode"  = pm."ERP_StockCode"
    LEFT JOIN review_facts rf ON rf."ProductID"  = pm."Review_ProductCode"
    GROUP BY pm."ERP_StockCode", pm."GoldenRecordName", pm."Category"
    """,
    """
    CREATE OR REPLACE VIEW v_customer_kpi AS
    SELECT
        c."ClientID",
        c."Nom",
        c."Pays",
        COUNT(DISTINCT sf."InvoiceNo")                          AS "NbCommandes",
        COUNT(DISTINCT sf."StockCode")                          AS "NbProduits",
        ROUND(SUM(sf."Revenue")::NUMERIC, 2)                    AS "CA_Total",
        ROUND(AVG(sf."Revenue")::NUMERIC, 2)                    AS "PanierMoyen",
        ROUND(AVG(rf."Rating")::NUMERIC, 2)                     AS "Notemoyenne",
        COUNT(rf."ReviewID")                                     AS "NbAvis"
    FROM customers c
    LEFT JOIN sales_facts    sf ON sf."CustomerID" = c."ClientID"
    LEFT JOIN product_mapping pm ON pm."ERP_StockCode" = sf."StockCode"
    LEFT JOIN review_facts   rf ON rf."ProductID"  = pm."Review_ProductCode"
    GROUP BY c."ClientID", c."Nom", c."Pays"
    """,
    """
    CREATE OR REPLACE VIEW v_alerts AS
    SELECT
        "ProductID",
        "ProductName",
        "Category",
        "CA",
        "Notemoyenne",
        "NbAvis",
        "AvisNegatifs",
        "QuantiteVendue",
        CASE
            WHEN "Notemoyenne" < 3.0 AND "QuantiteVendue" > 50 THEN 'CRITIQUE'
            WHEN "Notemoyenne" < 3.5                            THEN 'A_SURVEILLER'
            ELSE 'OK'
        END AS "Statut"
    FROM v_product_kpi
    WHERE "NbAvis" > 0
    ORDER BY "Notemoyenne" ASC
    """,
    """
    CREATE OR REPLACE VIEW v_data_quality AS
    SELECT
        (SELECT COUNT(*) FROM products)                                                    AS "Nb_Produits_ERP",
        (SELECT COUNT(*) FROM product_mapping)                                             AS "Nb_Golden_Records",
        (SELECT COUNT(DISTINCT "ProductID")       FROM review_facts)                       AS "Nb_Produits_Avis",
        (SELECT COUNT(*)                          FROM product_mapping)                    AS "Nb_Mappings",
        (SELECT COUNT(*)                          FROM review_facts)                       AS "Nb_Avis_Total",
        (SELECT COUNT(*) FROM review_facts        WHERE "ProductID" IS NOT NULL)           AS "Nb_Avis_Lies",
        (SELECT COUNT(DISTINCT "InvoiceNo")       FROM sales_facts)                        AS "Nb_Factures",
        (SELECT COUNT(*)                          FROM customers)                           AS "Nb_Clients",
        ROUND(
            100.0
            * (SELECT COUNT(DISTINCT rf."ProductID")
               FROM review_facts rf
               WHERE rf."ProductID" IN (SELECT "Review_ProductCode" FROM product_mapping))
            / GREATEST((SELECT COUNT(*) FROM product_mapping), 1), 1
        )                                                                                  AS "Taux_Couverture_MDM"
    """,
]


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────

def _create_schema(engine):
    """Supprime et recrée toutes les tables (idempotent)."""
    from sqlalchemy import text
    drop_order = ["review_facts", "sales_facts", "product_mapping", "customers", "products"]
    with engine.begin() as conn:
        for t in drop_order:
            conn.execute(text(f'DROP TABLE IF EXISTS {t} CASCADE'))
        for stmt in DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(text(stmt))
    print("[run_etl] Tables DDL OK (recréées)")


def _create_views(engine):
    """Crée (ou remplace) les vues analytiques PostgreSQL."""
    from sqlalchemy import text
    # DROP préalable pour éviter l'erreur PostgreSQL sur ajout/réordonnancement de colonnes
    views_order = ["v_alerts", "v_customer_kpi", "v_product_kpi", "v_data_quality"]
    with engine.begin() as conn:
        for v in views_order:
            conn.execute(text(f'DROP VIEW IF EXISTS {v} CASCADE'))
        for view_sql in VIEWS_SQL:
            conn.execute(text(view_sql.strip()))
    print("[run_etl] Vues analytiques créées")


def _truncate_tables(engine):
    """Vide les tables avant rechargement (idempotent, désactive FK temporairement)."""
    from sqlalchemy import text
    tables = ["review_facts", "sales_facts", "product_mapping", "customers", "products"]
    with engine.begin() as conn:
        for t in tables:
            conn.execute(text(f'DELETE FROM {t}'))
    print("[run_etl] Tables vidées (DELETE)")


# ────────────────────────────────────────────────────────────
#  Pipeline principal
# ────────────────────────────────────────────────────────────

def run_etl(force: bool = False):
    print(" Démarrage ETL SmartShop 360 → PostgreSQL\n")

    # 0. Vérification connexion BDD
    if not test_connection():
        raise RuntimeError(
            "Impossible de se connecter à PostgreSQL.\n"
            "Vérifiez que le conteneur Docker est bien lancé : docker-compose up -d db"
        )

    # 0b. Détection incrémentale — skip si rien n'a changé
    needs_run, changed = should_run_etl(ETL_SOURCE_FILES, force=force)
    if not needs_run:
        print("  ETL ignoré — données déjà à jour.")
        return

    engine = get_engine()
    _create_schema(engine)
    # _truncate_tables non nécessaire : DROP+CREATE dans _create_schema

    # 1. Source 1 — transactions CSV
    print(" Source 1 — Lecture & nettoyage CSV Online Retail II ...")
    raw_tx = load_transactions()
    tx     = clean_transactions(raw_tx)

    # 2. Top 50 Golden Records ERP (consigne POC : exactement 50)
    print(" Extraction Top 50 produits (Golden Records) ...")
    products_erp_top50 = extract_top50_products(tx, top_n=50)   # MDM : consigne stricte
    products_erp_all   = extract_top50_products(tx, top_n=0)    # Catalogue complet

    # 3. Source 2 — avis JSON réels
    print(" Source 2 — Chargement des avis JSON réels ...")
    reviews_df = load_reviews()

    # 4. MDM — table de mapping (Top 50 ERP × 50 avis les plus commentés)
    print(" Construction PRODUCT_MAPPING ...")
    mapping_df  = build_product_mapping(products_erp_top50, reviews_df)
    reviews_df  = attach_product_to_reviews(reviews_df, mapping_df)

    # 4b. Validation qualité (Great Expectations-style)
    print("\n Validation qualité des données ...")
    products_check_df = pd.DataFrame(
        [(sc, name, cat) for sc, name, cat in products_erp_top50],
        columns=["ProductID", "ProductName", "Category"]
    )
    quality_ok = run_all_validations(tx, reviews_df, products_check_df)
    if not quality_ok:
        if force:
            print("  Erreurs qualité détectées — force=True, ingestion poursuivie.")
        else:
            print("  Des erreurs de qualité ont été détectées — ETL interrompu.")
            print("   Utilisez force=True pour forcer l'ingestion malgré les erreurs.")
            raise ValueError("Validation qualité des données échouée.")

    # 5. Clients
    customers_df = extract_customers(tx)

    # 6. Chargement PostgreSQL ────────────────
    print("\n Chargement dans PostgreSQL ...")

    # Products — catalogue complet (4 630 produits)
    products_pg = pd.DataFrame([
        {"ProductID": sc, "ProductName": name, "Category": cat}
        for sc, name, cat in products_erp_all
    ])
    products_pg.to_sql("products", engine, if_exists="append", index=False)
    print(f"    {len(products_pg)} produits")

    # Customers
    customers_df.to_sql("customers", engine, if_exists="append", index=False)
    print(f"    {len(customers_df):,} clients")

    # Product mapping
    mapping_df.to_sql("product_mapping", engine, if_exists="append", index=False)
    print(f"    {len(mapping_df)} mappings MDM")

    # Sales facts — toutes les transactions nettoyées (aucun filtre)
    sales_pg = tx[["InvoiceNo", "StockCode", "Quantity", "Revenue",
                   "Margin", "InvoiceDate", "CustomerID"]].copy()
    sales_pg.to_sql("sales_facts", engine, if_exists="append", index=False)
    print(f"    {len(sales_pg):,} lignes de ventes")

    # Review facts
    review_pg = reviews_df.rename(columns={
        "Note":               "Rating",
        "Review_ProductCode": "ProductID",
    })[["ProductID", "Rating", "ReviewText", "Sentiment", "ReviewDate"]]
    review_pg.to_sql("review_facts", engine, if_exists="append", index=False)
    print(f"    {len(review_pg):,} avis clients")

    # 7. Vues analytiques
    _create_views(engine)

    # 8. Enregistrement des hashes (marque ce run comme réussi)
    record_hashes(ETL_SOURCE_FILES)

    print("\n ETL PostgreSQL terminé avec succès !")


if __name__ == "__main__":
    run_etl()
