"""
SmartShop 360 - Pipeline ETL
==============================
Démarche :
1. Source 1 : Lecture du fichier réel Online Retail II (CSV Kaggle)
2. Source 2 : Avis clients simulés (basés sur les produits réels du CSV)
3. Nettoyage & normalisation
4. MDM : Table de mapping produits (Golden Record – Top 50 produits)
5. Chargement dans SQLite (remplace PostgreSQL pour la démo locale)
"""

import sqlite3
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()  # charge .env si présent

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DB_PATH  = os.environ.get("DB_PATH",  os.path.join(os.path.dirname(__file__), "data", "smartshop360.db"))
CSV_PATH = os.environ.get("CSV_PATH", os.path.join(os.path.dirname(__file__), "Data", "online_retail_II.csv"))
SEED     = int(os.environ.get("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
# 1. SOURCE 1 : LECTURE CSV RÉEL (Online Retail II)
# ─────────────────────────────────────────────
def load_real_transactions(csv_path: str, sample_size: int = 20000) -> pd.DataFrame:
    """
    Charge le fichier Online Retail II réel.
    Colonnes attendues : Invoice, StockCode, Description, Quantity,
                         InvoiceDate, Price, Customer ID, Country
    On prend un échantillon pour les performances de démo.
    """
    print(f"   Lecture du fichier : {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype={"Customer ID": str})
    print(f"   {len(df):,} lignes brutes chargées")
    # Normaliser les noms de colonnes pour correspondre au reste du pipeline
    df = df.rename(columns={
        "Invoice":      "InvoiceNo",
        "StockCode":    "StockCode",
        "Description":  "Description",
        "Quantity":     "Quantity",
        "InvoiceDate":  "InvoiceDate",
        "Price":        "UnitPrice",
        "Customer ID":  "CustomerID",
        "Country":      "Country",
    })
    # Echantillonnage reproductible pour la démo
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)
        print(f"   Echantillon : {sample_size:,} lignes retenues")
    return df


# ─────────────────────────────────────────────
# 2. TOP 50 PRODUITS ERP (extrait du CSV réel)
# ─────────────────────────────────────────────
def extract_top50_products(transactions_clean: pd.DataFrame) -> list:
    """
    Retourne la liste des 50 produits les plus vendus (Golden Records ERP).
    Format : [(ProductID, ProductName, Category), ...]
    """
    top = (
        transactions_clean
        .groupby(["StockCode", "Description"])["Quantity"]
        .sum()
        .nlargest(50)
        .reset_index()
    )
    # Catégories déduites par mots-clés simples (simulation MDM)
    def infer_category(name: str) -> str:
        n = str(name).upper()
        if any(w in n for w in ["CANDLE", "LIGHT", "LAMP", "LANTERN", "LED"]):      return "Luminaires"
        if any(w in n for w in ["BAG", "TOTE", "PURSE", "SACK"]):                   return "Accessoires"
        if any(w in n for w in ["MUG", "CUP", "BOWL", "PLATE", "JAR", "BOTTLE"]):  return "Cuisine"
        if any(w in n for w in ["FRAME", "PICTURE", "PHOTO"]):                       return "Cadres"
        if any(w in n for w in ["HEART", "LOVE", "GIFT", "BOX", "WRAP"]):           return "Cadeaux"
        if any(w in n for w in ["PAPER", "CARD", "BOOK", "NOTEBOOK"]):              return "Papeterie"
        if any(w in n for w in ["CUSHION", "PILLOW", "RUG", "CURTAIN"]):            return "Textiles"
        if any(w in n for w in ["MIRROR", "CLOCK", "VASE", "ORNAMENT"]):            return "Décoration"
        if any(w in n for w in ["FLOWER", "PLANT", "POT", "GARDEN"]):               return "Jardin"
        if any(w in n for w in ["SOAP", "OIL", "SPA", "DIFFUSER"]):                 return "Bien-être"
        return "Décoration"

    products = []
    for i, row in enumerate(top.itertuples(), 1):
        products.append((
            row.StockCode,
            str(row.Description).strip().title(),
            infer_category(str(row.Description))
        ))
    return products


# ─────────────────────────────────────────────
# PRODUITS ERP (rempli dynamiquement depuis le CSV)
# ─────────────────────────────────────────────
# Placeholder — sera remplacé par les vraies données CSV dans run_etl()
PRODUCTS_ERP = []

# ─────────────────────────────────────────────
# 2. SOURCE 2 : AVIS CLIENTS SIMULÉS
#    (basés sur les produits réels extraits du CSV)
# ─────────────────────────────────────────────
POSITIVE_REVIEWS = [
    "Absolutely love this product! Great quality.",
    "Beautiful design, exactly as described.",
    "Fast delivery and perfect packaging.",
    "Exceeded my expectations, highly recommend.",
    "Perfect gift, my family loved it.",
    "Great value for money, very satisfied.",
    "Stunning piece, looks amazing in my home.",
    "Very well made, solid and durable.",
    "Lovely colors, matches my decor perfectly.",
    "Will definitely buy again!",
]

NEGATIVE_REVIEWS = [
    "Color was different from the photo, disappointed.",
    "Packaging was damaged on arrival.",
    "Very fragile, broke after one week.",
    "Not worth the price, poor quality.",
    "Customer service was unhelpful.",
    "Product looks nothing like the picture.",
    "Took 3 weeks to arrive, unacceptable.",
    "Missing parts, had to return it.",
    "Cheap material, very disappointed.",
    "Would not recommend to anyone.",
]

NEUTRAL_REVIEWS = [
    "Decent product, nothing special.",
    "Average quality for the price.",
    "It's okay, does the job.",
    "Some minor defects but acceptable.",
    "Delivery was slow but product is fine.",
]

def generate_reviews(products_erp: list, n: int = 1000) -> pd.DataFrame:
    """
    Génère des avis simulés basés sur les vrais produits ERP.
    products_erp : liste de tuples (StockCode, ProductName, Category)
    """
    rows = []
    start = datetime(2023, 1, 1)
    review_products = [(p[0], p[1]) for p in products_erp]

    for i in range(n):
        product = random.choice(review_products)
        sentiment_roll = random.random()
        if sentiment_roll < 0.55:
            sentiment = "positive"
            text = random.choice(POSITIVE_REVIEWS)
            note = round(random.uniform(3.5, 5.0), 1)
        elif sentiment_roll < 0.80:
            sentiment = "negative"
            text = random.choice(NEGATIVE_REVIEWS)
            note = round(random.uniform(1.0, 2.5), 1)
        else:
            sentiment = "neutral"
            text = random.choice(NEUTRAL_REVIEWS)
            note = round(random.uniform(2.5, 3.5), 1)
        review_date = start + timedelta(days=random.randint(0, 730))
        rows.append({
            "ReviewID":   f"REV{str(i+1).zfill(5)}",
            "ProductCode": product[0],
            "ProductName": product[1],
            "ReviewText":  text,
            "Sentiment":   sentiment,
            "Note":        note,
            "ReviewDate":  review_date,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# 3. NETTOYAGE ETL
# ─────────────────────────────────────────────
def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoyage Source 1 (Online Retail II réel) :
    - Supprime les lignes annulées (InvoiceNo commence par 'C')
    - Supprime les quantités négatives et prix nuls
    - Supprime les lignes sans CustomerID
    - Filtre sur le Top 50 produits uniquement
    - Calcule le CA et la marge simulée
    """
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    # Marge simulée entre 25 % et 45 % par ligne
    rng = np.random.default_rng(SEED)
    df["Margin"] = (df["Revenue"] * rng.uniform(0.25, 0.45, len(df))).round(2)
    return df.reset_index(drop=True)


def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoyage Source 2 : supprime doublons."""
    return df.drop_duplicates(subset=["ReviewID"]).reset_index(drop=True)


# ─────────────────────────────────────────────
# 4. MDM — TABLE DE MAPPING PRODUITS (Golden Record)
# ─────────────────────────────────────────────
def build_product_mapping(products_erp: list) -> pd.DataFrame:
    """
    Stratégie MDM simulée :
    - Source 1 (ERP) : Top 50 produits réels du CSV Online Retail II
    - Source 2 (Avis) : Les avis sont générés sur les mêmes produits réels
    - Mapping 1-à-1 : Le StockCode ERP sert de clé commune (Golden Record)

    Dans un contexte réel, on utiliserait :
    - Codes EAN/GTIN/SKU communs entre systèmes
    - Fuzzy Matching (rapidfuzz, Levenshtein) sur les noms de produits
    - Embedding sémantique (sentence-transformers + similarité cosinus)
    """
    mapping = []
    for i, (stock_code, product_name, category) in enumerate(products_erp):
        mapping.append({
            "MappingID":           f"MAP{str(i+1).zfill(3)}",
            "ERP_ProductCode":     stock_code,
            "ERP_ProductName":     product_name,
            "Review_ProductCode":  stock_code,   # même clé — source unifiée
            "Review_ProductName":  product_name,
            "Category":            category,
            "GoldenRecordName":    product_name,
        })
    return pd.DataFrame(mapping)


# ─────────────────────────────────────────────
# 5. EXTRACTION DES CLIENTS RÉELS
# ─────────────────────────────────────────────
def extract_customers(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    """
    Extrait les clients uniques de la source CSV réelle.
    Retourne un DataFrame CUSTOMERS(ClientID, Nom, Pays).
    """
    customers = (
        transactions_clean[["CustomerID", "Country"]]
        .drop_duplicates(subset=["CustomerID"])
        .copy()
    )
    customers["Nom"] = "Customer_" + customers["CustomerID"].astype(str)
    customers = customers.rename(columns={"CustomerID": "ClientID", "Country": "Pays"})
    return customers[["ClientID", "Nom", "Pays"]].reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. CHARGEMENT EN BASE
# ─────────────────────────────────────────────
def load_to_db(conn, transactions_clean: pd.DataFrame,
               reviews_clean: pd.DataFrame,
               mapping_df: pd.DataFrame,
               customers_df: pd.DataFrame):
    """Charge toutes les tables dans SQLite."""

    # Customers (réels depuis le CSV)
    customers_df.to_sql("CUSTOMERS", conn, if_exists="replace", index=False)

    # Products (Golden Record issu du Top 50 ERP)
    products_df = mapping_df[["ERP_ProductCode", "GoldenRecordName", "Category"]].copy()
    products_df.columns = ["ProductID", "ProductName", "Category"]
    products_df.to_sql("PRODUCTS", conn, if_exists="replace", index=False)

    # Mapping MDM
    mapping_df.to_sql("PRODUCT_MAPPING", conn, if_exists="replace", index=False)

    # Filtrer les transactions sur le Top 50 seulement
    top50_codes = set(mapping_df["ERP_ProductCode"].tolist())
    tx = transactions_clean[transactions_clean["StockCode"].isin(top50_codes)].copy()

    # Invoices (agrégées par InvoiceNo)
    invoices = (
        tx.groupby(["InvoiceNo", "CustomerID", "InvoiceDate"])
        .agg(MontantTotal=("Revenue", "sum"))
        .reset_index()
    )
    invoices.columns = ["FactureID", "ClientID", "Date", "MontantTotal"]
    invoices.to_sql("INVOICES", conn, if_exists="replace", index=False)

    # Invoice Lines
    invoice_lines = tx[["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "Revenue", "Margin"]].copy()
    invoice_lines.columns = ["FactureID", "ProduitID", "Quantite", "PrixUnitaire", "Revenue", "Margin"]
    invoice_lines = invoice_lines.reset_index(drop=True)
    invoice_lines.insert(0, "LigneID", invoice_lines.index + 1)
    invoice_lines.to_sql("INVOICE_LINES", conn, if_exists="replace", index=False)

    # Reviews (les avis utilisent déjà le même StockCode → ProduitID direct)
    reviews_db = reviews_clean.rename(columns={"ProductCode": "ProduitID"})
    reviews_db = reviews_db[["ReviewID", "ReviewText", "Sentiment", "Note", "ReviewDate", "ProduitID"]]
    reviews_db.to_sql("REVIEWS", conn, if_exists="replace", index=False)

    conn.commit()
    print(f"✅ Chargement terminé :")
    print(f"   - {len(customers_df):,} clients")
    print(f"   - {len(products_df)} produits (golden records)")
    print(f"   - {len(mapping_df)} mappings MDM")
    print(f"   - {len(invoices):,} factures (Top 50 produits)")
    print(f"   - {len(invoice_lines):,} lignes de facture")
    print(f"   - {len(reviews_db):,} avis clients")


# ─────────────────────────────────────────────
# VUES ANALYTIQUES
# ─────────────────────────────────────────────
def create_analytical_views(conn):
    """Crée (ou recrée) les vues SQL pour le dashboard et l'agent."""
    cur = conn.cursor()
    # Drop and recreate pour éviter les conflits de schéma
    for v in ["V_PRODUCT_KPI", "V_CUSTOMER_KPI", "V_ALERTS", "V_DATA_QUALITY"]:
        cur.execute(f"DROP VIEW IF EXISTS {v}")

    views = [
        """
        CREATE VIEW V_PRODUCT_KPI AS
        SELECT
            p.ProductID,
            p.ProductName,
            p.Category,
            ROUND(COALESCE(SUM(il.Revenue), 0), 2)   AS CA,
            ROUND(COALESCE(SUM(il.Margin), 0), 2)    AS Marge,
            COALESCE(SUM(il.Quantite), 0)             AS QuantiteVendue,
            ROUND(AVG(r.Note), 2)                     AS Notemoyenne,
            COUNT(r.ReviewID)                         AS NbAvis,
            SUM(CASE WHEN r.Sentiment = 'positive' THEN 1 ELSE 0 END) AS AvisPositifs,
            SUM(CASE WHEN r.Sentiment = 'negative' THEN 1 ELSE 0 END) AS AvisNegatifs,
            SUM(CASE WHEN r.Sentiment = 'neutral'  THEN 1 ELSE 0 END) AS AvisNeutres
        FROM PRODUCTS p
        LEFT JOIN INVOICE_LINES il ON il.ProduitID = p.ProductID
        LEFT JOIN REVIEWS r        ON r.ProduitID  = p.ProductID
        GROUP BY p.ProductID, p.ProductName, p.Category
        """,
        """
        CREATE VIEW V_CUSTOMER_KPI AS
        SELECT
            c.ClientID,
            c.Nom,
            c.Pays,
            COUNT(DISTINCT i.FactureID)            AS NbCommandes,
            ROUND(SUM(i.MontantTotal), 2)          AS CA_Total,
            ROUND(AVG(i.MontantTotal), 2)          AS PanierMoyen
        FROM CUSTOMERS c
        LEFT JOIN INVOICES i ON i.ClientID = c.ClientID
        GROUP BY c.ClientID, c.Nom, c.Pays
        """,
        """
        CREATE VIEW V_ALERTS AS
        SELECT
            ProductID,
            ProductName,
            Category,
            CA,
            Notemoyenne,
            NbAvis,
            AvisNegatifs,
            QuantiteVendue,
            CASE
                WHEN Notemoyenne < 3.0 AND QuantiteVendue > 50 THEN 'CRITIQUE'
                WHEN Notemoyenne < 3.5                          THEN 'A_SURVEILLER'
                ELSE 'OK'
            END AS Statut
        FROM V_PRODUCT_KPI
        WHERE NbAvis > 0
        ORDER BY Notemoyenne ASC
        """,
        """
        CREATE VIEW V_DATA_QUALITY AS
        SELECT
            (SELECT COUNT(*) FROM PRODUCTS)                              AS Nb_Produits_ERP,
            (SELECT COUNT(DISTINCT ProduitID) FROM REVIEWS)              AS Nb_Produits_Avis,
            (SELECT COUNT(*) FROM PRODUCT_MAPPING)                       AS Nb_Mappings,
            (SELECT COUNT(*) FROM REVIEWS)                               AS Nb_Avis_Total,
            (SELECT COUNT(*) FROM REVIEWS WHERE ProduitID IS NOT NULL)   AS Nb_Avis_Lies,
            (SELECT COUNT(*) FROM INVOICES)                              AS Nb_Factures,
            (SELECT COUNT(*) FROM CUSTOMERS)                             AS Nb_Clients,
            ROUND(
                100.0 * (SELECT COUNT(DISTINCT ProduitID) FROM REVIEWS)
                / MAX((SELECT COUNT(*) FROM PRODUCTS), 1), 1
            )                                                            AS Taux_Couverture_MDM
        """
    ]
    for v in views:
        cur.execute(v)
    conn.commit()
    print("✅ Vues analytiques créées (dont V_DATA_QUALITY)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_etl():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    print("🚀 Démarrage du pipeline ETL SmartShop 360\n")

    # ── Source 1 : CSV réel ──────────────────
    print("📦 Source 1 — Lecture du fichier Online Retail II (CSV réel)...")
    if os.path.exists(CSV_PATH):
        raw_transactions = load_real_transactions(CSV_PATH, sample_size=30000)
    else:
        raise FileNotFoundError(
            f"Fichier introuvable : {CSV_PATH}\n"
            "Placez online_retail_II.csv dans le répertoire Data/."
        )

    transactions_clean = clean_transactions(raw_transactions)
    print(f"   {len(raw_transactions):,} lignes brutes → {len(transactions_clean):,} après nettoyage")

    # ── Top 50 produits ───────────────────────
    print("🔝 Extraction du Top 50 produits (Golden Records ERP)...")
    products_erp = extract_top50_products(transactions_clean)
    print(f"   {len(products_erp)} produits extraits")

    # ── Source 2 : avis simulés ───────────────
    print("📦 Source 2 — Génération des avis clients (basés sur produits réels)...")
    raw_reviews    = generate_reviews(products_erp, n=1000)
    reviews_clean  = clean_reviews(raw_reviews)
    print(f"   {len(reviews_clean):,} avis générés")

    # ── MDM ────────────────────────────────────
    print("🔗 Construction de la table MDM (Mapping Produits)...")
    mapping_df = build_product_mapping(products_erp)
    print(f"   {len(mapping_df)} correspondances ERP ↔ Avis créées")

    # ── Clients réels ─────────────────────────
    customers_df = extract_customers(transactions_clean)
    print(f"   {len(customers_df):,} clients uniques extraits")

    # ── Chargement SQLite ─────────────────────
    print(f"\n💾 Chargement dans la base de données : {DB_PATH}")
    conn = sqlite3.connect(DB_PATH)
    load_to_db(conn, transactions_clean, reviews_clean, mapping_df, customers_df)
    create_analytical_views(conn)
    conn.close()

    print("\n✅ ETL terminé avec succès !")
    return DB_PATH


if __name__ == "__main__":
    run_etl()
