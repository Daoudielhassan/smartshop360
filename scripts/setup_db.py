"""
scripts/setup_db.py
====================
Initialisation PostgreSQL production — SmartShop 360

Ce script crée (ou recrée) :
  • 5 tables     : products, customers, product_mapping, sales_facts, review_facts
  • 11 index     : stockcode, customer, date, trgm…
  • 4 vues       : v_product_kpi, v_customer_kpi, v_alerts, v_data_quality
  • 2 extensions : pg_trgm, unaccent

Options :
    --drop-only   Supprime seulement les objets, sans recréer
    --views-only  Recrée uniquement les vues (tables inchangées)
    --check       Affiche le statut sans modifier la base

Usage :
    python scripts/setup_db.py              # initialisation complète
    python scripts/setup_db.py --views-only # mise à jour des vues uniquement
    python scripts/setup_db.py --check      # état actuel
"""

import os
import sys
import argparse
import textwrap
from pathlib import Path

# ── Chemin racine du projet ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from sqlalchemy import text
from src.db_config import get_engine, test_connection


# ============================================================================
#  DDL — Tables
# ============================================================================

_DROP_VIEWS = [
    "v_alerts",
    "v_customer_kpi",
    "v_product_kpi",
    "v_data_quality",
]

_DROP_TABLES = [
    "review_facts",
    "sales_facts",
    "product_mapping",
    "customers",
    "products",
]

_DDL_EXTENSIONS = """
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS unaccent;
"""

_DDL_TABLES = """
CREATE TABLE IF NOT EXISTS products (
    "ProductID"   VARCHAR(50)  PRIMARY KEY,
    "ProductName" VARCHAR(255) NOT NULL,
    "Category"    VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS customers (
    "ClientID" VARCHAR(50)  PRIMARY KEY,
    "Nom"      VARCHAR(100),
    "Pays"     VARCHAR(100)
);

CREATE TABLE IF NOT EXISTS product_mapping (
    "MappingID"          SERIAL        PRIMARY KEY,
    "ERP_StockCode"      VARCHAR(50)   NOT NULL,
    "ERP_ProductName"    VARCHAR(255),
    "Review_ProductCode" VARCHAR(50),
    "Review_ProductName" VARCHAR(255),
    "Category"           VARCHAR(100),
    "GoldenRecordName"   VARCHAR(255),
    "MatchScore"         FLOAT         NOT NULL DEFAULT 0.0,
    "MatchStrategy"      VARCHAR(20)   NOT NULL DEFAULT 'rank',
    "CreatedAt"          TIMESTAMP     NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sales_facts (
    "FactID"      SERIAL        PRIMARY KEY,
    "InvoiceNo"   VARCHAR(20)   NOT NULL,
    "StockCode"   VARCHAR(50)   NOT NULL,
    "Quantity"    INTEGER       NOT NULL DEFAULT 0,
    "Revenue"     NUMERIC(12,2) NOT NULL DEFAULT 0,
    "Margin"      NUMERIC(12,2) NOT NULL DEFAULT 0,
    "InvoiceDate" TIMESTAMP,
    "CustomerID"  VARCHAR(50),
    CONSTRAINT fk_sales_customer
        FOREIGN KEY ("CustomerID") REFERENCES customers("ClientID") ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS review_facts (
    "ReviewID"   SERIAL       PRIMARY KEY,
    "ProductID"  VARCHAR(50),
    "Rating"     NUMERIC(3,1) CHECK ("Rating" BETWEEN 1.0 AND 5.0),
    "ReviewText" TEXT,
    "Sentiment"  VARCHAR(20)  CHECK ("Sentiment" IN ('positive','negative','neutral')),
    "ReviewDate" TIMESTAMP
);
"""

_DDL_INDEXES = [
    # product_mapping
    ('idx_pm_erp_code',      'product_mapping',  '"ERP_StockCode"'),
    ('idx_pm_review_code',   'product_mapping',  '"Review_ProductCode"'),
    ('idx_pm_strategy',      'product_mapping',  '"MatchStrategy"'),
    # sales_facts
    ('idx_sf_stockcode',     'sales_facts',      '"StockCode"'),
    ('idx_sf_customerid',    'sales_facts',      '"CustomerID"'),
    ('idx_sf_invoicedate',   'sales_facts',      '"InvoiceDate"'),
    ('idx_sf_invoiceno',     'sales_facts',      '"InvoiceNo"'),
    # review_facts
    ('idx_rf_productid',     'review_facts',     '"ProductID"'),
    ('idx_rf_sentiment',     'review_facts',     '"Sentiment"'),
    ('idx_rf_rating',        'review_facts',     '"Rating"'),
    # customers
    ('idx_cust_pays',        'customers',        '"Pays"'),
]

_DDL_TRGM_INDEXES = [
    ('idx_pm_golden_trgm',   'product_mapping', '"GoldenRecordName"'),
    ('idx_pm_erpname_trgm',  'product_mapping', '"ERP_ProductName"'),
]

# ============================================================================
#  DDL — Vues
# ============================================================================

_VIEWS = {
    "v_product_kpi": """
        CREATE OR REPLACE VIEW v_product_kpi AS
        SELECT
            pm."ERP_StockCode"                                        AS "ProductID",
            pm."GoldenRecordName"                                     AS "ProductName",
            pm."Category",
            ROUND(COALESCE(SUM(sf."Revenue"), 0)::NUMERIC, 2)        AS "CA",
            ROUND(COALESCE(SUM(sf."Margin"),  0)::NUMERIC, 2)        AS "Marge",
            COALESCE(SUM(sf."Quantity"), 0)                          AS "QuantiteVendue",
            ROUND(AVG(rf."Rating")::NUMERIC, 2)                      AS "Notemoyenne",
            COUNT(rf."ReviewID")                                     AS "NbAvis",
            SUM(CASE WHEN rf."Sentiment" = 'positive' THEN 1 ELSE 0 END) AS "AvisPositifs",
            SUM(CASE WHEN rf."Sentiment" = 'negative' THEN 1 ELSE 0 END) AS "AvisNegatifs",
            SUM(CASE WHEN rf."Sentiment" = 'neutral'  THEN 1 ELSE 0 END) AS "AvisNeutres"
        FROM product_mapping pm
        LEFT JOIN sales_facts  sf ON sf."StockCode"  = pm."ERP_StockCode"
        LEFT JOIN review_facts rf ON rf."ProductID"  = pm."Review_ProductCode"
        GROUP BY pm."ERP_StockCode", pm."GoldenRecordName", pm."Category"
    """,
    "v_customer_kpi": """
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
        LEFT JOIN sales_facts    sf ON sf."CustomerID"    = c."ClientID"
        LEFT JOIN product_mapping pm ON pm."ERP_StockCode" = sf."StockCode"
        LEFT JOIN review_facts   rf ON rf."ProductID"     = pm."Review_ProductCode"
        GROUP BY c."ClientID", c."Nom", c."Pays"
    """,
    "v_alerts": """
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
        ORDER BY p."Notemoyenne" ASC
    """,
    "v_data_quality": """
        CREATE OR REPLACE VIEW v_data_quality AS
        SELECT
            (SELECT COUNT(*)                    FROM products)                                      AS "Nb_Produits_ERP",
            (SELECT COUNT(DISTINCT "ProductID") FROM review_facts)                                  AS "Nb_Produits_Avis",
            (SELECT COUNT(*)                    FROM product_mapping)                               AS "Nb_Mappings",
            (SELECT COUNT(*) FROM product_mapping WHERE "MatchStrategy" = 'hard')                   AS "Nb_Hard_Matches",
            (SELECT COUNT(*) FROM product_mapping WHERE "MatchStrategy" = 'fuzzy')                  AS "Nb_Fuzzy_Matches",
            (SELECT COUNT(*) FROM product_mapping WHERE "MatchStrategy" = 'tfidf')                  AS "Nb_TFIDF_Matches",
            (SELECT COUNT(*) FROM product_mapping WHERE "MatchStrategy" = 'sbert')                  AS "Nb_SBERT_Matches",
            (SELECT COUNT(*)                    FROM review_facts)                                   AS "Nb_Avis_Total",
            (SELECT COUNT(*) FROM review_facts  WHERE "ProductID" IS NOT NULL)                       AS "Nb_Avis_Lies",
            (SELECT COUNT(DISTINCT "InvoiceNo") FROM sales_facts)                                    AS "Nb_Factures",
            (SELECT COUNT(*)                    FROM customers)                                      AS "Nb_Clients",
            ROUND(
                100.0
                * (SELECT COUNT(DISTINCT rf."ProductID")
                   FROM review_facts rf
                   WHERE rf."ProductID" IN (SELECT "Review_ProductCode" FROM product_mapping))
                / GREATEST((SELECT COUNT(*) FROM product_mapping), 1),
            1)                                                                                       AS "Taux_Couverture_MDM"
    """,
}


# ============================================================================
#  Helpers
# ============================================================================

def _ok(msg: str):  print(f"  \u2713  {msg}")
def _skip(msg: str): print(f"  \u25e6  {msg}")
def _err(msg: str):  print(f"  \u2717  {msg}", file=sys.stderr)


def _exec(conn, sql: str, label: str):
    try:
        conn.execute(text(sql))
        _ok(label)
        return True
    except Exception as e:
        _err(f"{label} : {e}")
        return False


def drop_all(conn):
    print("\n[1/5] Suppression des objets existants")
    for v in _DROP_VIEWS:
        conn.execute(text(f"DROP VIEW IF EXISTS {v} CASCADE"))
        _ok(f"DROP VIEW {v}")
    for t in _DROP_TABLES:
        conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
        _ok(f"DROP TABLE {t}")


def create_extensions(conn):
    print("\n[2/5] Extensions PostgreSQL")
    for stmt in _DDL_EXTENSIONS.strip().split(";"):
        stmt = stmt.strip()
        if stmt:
            _exec(conn, stmt, stmt.replace("\n", " ")[:60])


def create_tables(conn):
    print("\n[3/5] Création des tables")
    for stmt in _DDL_TABLES.strip().split(";"):
        stmt = stmt.strip()
        if not stmt:
            continue
        # Extraire le nom de la table pour le label
        import re
        m = re.search(r"CREATE TABLE IF NOT EXISTS (\w+)", stmt)
        label = m.group(1) if m else stmt[:40]
        _exec(conn, stmt, label)


def create_indexes(conn):
    print("\n[4/5] Création des index")
    for name, table, col in _DDL_INDEXES:
        sql = f'CREATE INDEX IF NOT EXISTS {name} ON {table} ({col})'
        _exec(conn, sql, name)
    for name, table, col in _DDL_TRGM_INDEXES:
        sql = f'CREATE INDEX IF NOT EXISTS {name} ON {table} USING gin ({col} gin_trgm_ops)'
        _exec(conn, sql, f"{name} (gin_trgm)")


def create_views(conn):
    print("\n[5/5] Création des vues analytiques")
    # DROP d'abord pour éviter l'erreur PostgreSQL sur réordonnancement de colonnes
    for v in _DROP_VIEWS:
        conn.execute(text(f"DROP VIEW IF EXISTS {v} CASCADE"))
    for name, sql in _VIEWS.items():
        _exec(conn, sql.strip(), name)


def check_status(engine):
    """Affiche l'état actuel des tables et vues sans modifier la base."""
    print("\n── État de la base de données ──────────────────────────────\n")
    with engine.connect() as conn:
        # Tables
        rows = conn.execute(text("""
            SELECT tablename, pg_size_pretty(pg_total_relation_size(quote_ident(tablename))) AS size,
                   (SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_name = tablename AND table_schema = 'public') AS nb_cols
            FROM   pg_tables
            WHERE  schemaname = 'public'
              AND  tablename IN ('products','customers','product_mapping','sales_facts','review_facts')
            ORDER  BY tablename
        """)).fetchall()

        if rows:
            print(f"  {'Table':<25} {'Taille':>10}  {'Colonnes':>8}")
            print(f"  {'-'*25} {'-'*10}  {'-'*8}")
            for r in rows:
                print(f"  {r[0]:<25} {r[1]:>10}  {r[2]:>8}")
        else:
            print("  Aucune table trouvée.")

        # Vues
        vrows = conn.execute(text("""
            SELECT viewname
            FROM   pg_views
            WHERE  schemaname = 'public'
              AND  viewname IN ('v_product_kpi','v_customer_kpi','v_alerts','v_data_quality')
            ORDER  BY viewname
        """)).fetchall()

        print(f"\n  Vues : {', '.join(r[0] for r in vrows) if vrows else 'aucune'}")

        # Index
        irows = conn.execute(text("""
            SELECT indexname, tablename
            FROM   pg_indexes
            WHERE  schemaname = 'public'
              AND  indexname LIKE 'idx_%'
            ORDER  BY tablename, indexname
        """)).fetchall()
        print(f"  Index : {len(irows)} index applicatifs")

    print()


# ============================================================================
#  Point d'entrée
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="SmartShop 360 — Initialisation PostgreSQL production",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""
            Exemples :
              python scripts/setup_db.py              # initialisation complète
              python scripts/setup_db.py --views-only # recréer les vues uniquement
              python scripts/setup_db.py --drop-only  # tout supprimer
              python scripts/setup_db.py --check      # état sans modification
        """),
    )
    parser.add_argument("--drop-only",   action="store_true", help="Supprime seulement les objets")
    parser.add_argument("--views-only",  action="store_true", help="Recrée uniquement les vues")
    parser.add_argument("--check",       action="store_true", help="Affiche l'état sans modifier")
    args = parser.parse_args()

    print("=" * 60)
    print("  SmartShop 360 — Initialisation PostgreSQL Production")
    print("=" * 60)

    # Vérification connexion
    print("\nConnexion à PostgreSQL...")
    if not test_connection():
        _err("Impossible de se connecter à PostgreSQL.")
        _err("Vérifiez que Docker est lancé : docker-compose up -d db")
        sys.exit(1)
    _ok("Connexion établie")

    engine = get_engine()

    if args.check:
        check_status(engine)
        return

    with engine.begin() as conn:
        if args.drop_only:
            drop_all(conn)
            print("\nSuppression terminée.")
            return

        if args.views_only:
            create_views(conn)
        else:
            drop_all(conn)
            create_extensions(conn)
            create_tables(conn)
            create_indexes(conn)
            create_views(conn)

    print("\n" + "=" * 60)
    print("  Initialisation terminée avec succès !")
    print("=" * 60)

    # Afficher l'état final
    check_status(engine)


if __name__ == "__main__":
    main()
