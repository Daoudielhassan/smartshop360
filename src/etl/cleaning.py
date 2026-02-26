"""
src/etl/cleaning.py
====================
Fonctions de nettoyage et normalisation des deux sources :
  - Source 1 : transactions ERP (Online Retail II CSV)
  - Source 2 : avis clients (labeledReview.datasetFix.json)
"""

import os
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SEED             = int(os.environ.get("SEED", 42))
SAMPLE_SIZE      = int(os.environ.get("SAMPLE_SIZE", 50000))  # 0 = toutes les lignes
TOP_N_PRODUCTS   = int(os.environ.get("TOP_N_PRODUCTS", 70))  # 0 = tous les produits
CSV_PATH    = os.environ.get(
    "CSV_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "Data", "online_retail_II.csv"),
)
JSON_PATH   = os.environ.get(
    "REVIEWS_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "Data", "labeledReview.datasetFix.json"),
)

random.seed(SEED)
np.random.seed(SEED)
rng = np.random.default_rng(SEED)


# ────────────────────────────────────────────────────────────
#  SOURCE 1 — ERP / Transactions (CSV)
# ────────────────────────────────────────────────────────────

def load_transactions(csv_path: str = CSV_PATH, sample_size: int = SAMPLE_SIZE) -> pd.DataFrame:
    """
    Charge le CSV Online Retail II et en prend un échantillon reproductible.
    Colonnes source : Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country
    """
    print(f"[cleaning] Lecture CSV : {csv_path}")
    df = pd.read_csv(csv_path, encoding="utf-8", dtype={"Customer ID": str})
    print(f"[cleaning] {len(df):,} lignes brutes")

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

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=SEED).reset_index(drop=True)
        print(f"[cleaning] Échantillon : {sample_size:,} lignes retenues")
    else:
        print(f"[cleaning] Toutes les lignes conservées : {len(df):,}")

    return df


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les transactions :
      - Supprime les factures annulées (InvoiceNo commence par 'C')
      - Supprime Quantity <= 0 et UnitPrice <= 0
      - Supprime les CustomerID manquants
      - Calcule Revenue et Margin par ligne
    """
    print(f"[cleaning] Nettoyage — {len(df):,} lignes initiales")

    # Commandes annulées
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

    # Valeurs négatives / nulles
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

    # CustomerID manquant
    df = df[df["CustomerID"].notna() & (df["CustomerID"].str.strip() != "")]

    # Conversion date
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    df = df[df["InvoiceDate"].notna()]

    # Calcul Revenue & Margin (marge simulée 20-45 %)
    df = df.copy()
    df["Revenue"] = (df["Quantity"] * df["UnitPrice"]).round(2)
    df = df[df["Revenue"] > 0]          # supprime les lignes Revenue nul/négatif résiduel
    margin_rates  = rng.uniform(0.20, 0.45, size=len(df))
    df["Margin"]  = (df["Revenue"] * margin_rates).round(2)

    df = df.reset_index(drop=True)
    print(f"[cleaning] {len(df):,} lignes après nettoyage")
    return df


def extract_top50_products(transactions_clean: pd.DataFrame, top_n: int = TOP_N_PRODUCTS) -> list:
    """
    Extrait les N produits les plus vendus (Golden Records ERP).
    top_n=0 ou top_n=None → tous les produits uniques (aucune limite).
    Retourne [(StockCode, Description, Category), ...]
    """
    grouped = (
        transactions_clean
        .groupby(["StockCode", "Description"])["Quantity"]
        .sum()
        .reset_index()
        .drop_duplicates(subset=["StockCode"])  # garantit l'unicité de la PK
        .sort_values("Quantity", ascending=False)
    )
    if top_n:
        top = grouped.head(top_n)
        print(f"[cleaning] Sélection des {top_n} produits les plus vendus")
    else:
        top = grouped
        print(f"[cleaning] Tous les produits retenus : {len(top):,} produits uniques")

    def infer_category(name: str) -> str:
        n = str(name).upper()
        if any(w in n for w in ["CANDLE", "LIGHT", "LAMP", "LANTERN", "LED"]):   return "Luminaires"
        if any(w in n for w in ["BAG", "TOTE", "PURSE", "SATCHEL"]):              return "Sacs & Accessoires"
        if any(w in n for w in ["MUG", "CUP", "TEA", "COFFEE", "PLATE", "BOWL"]): return "Art de la Table"
        if any(w in n for w in ["FRAME", "SIGN", "CLOCK", "MIRROR"]):             return "Décoration Murale"
        if any(w in n for w in ["CARD", "WRAP", "RIBBON", "GIFT", "BOX"]):        return "Emballages & Cadeaux"
        if any(w in n for w in ["HEART", "ROSE", "FLOWER", "BIRD", "BUTTERFLY"]): return "Nature & Romantique"
        return "Divers"

    products = [
        (row["StockCode"], row["Description"], infer_category(row["Description"]))
        for _, row in top.iterrows()
    ]
    print(f"[cleaning] {len(products)} produits Golden Record extraits")
    return products


def extract_customers(transactions_clean: pd.DataFrame) -> pd.DataFrame:
    """Extrait les clients uniques du dataset nettoyé."""
    customers = (
        transactions_clean[["CustomerID", "Country"]]
        .drop_duplicates(subset=["CustomerID"])
        .rename(columns={"CustomerID": "ClientID", "Country": "Pays"})
        .reset_index(drop=True)
    )
    customers["Nom"] = customers["ClientID"].apply(lambda x: f"Client_{x}")
    print(f"[cleaning] {len(customers):,} clients uniques extraits")
    return customers


# ────────────────────────────────────────────────────────────
#  SOURCE 2 — Avis clients (JSON réel)
# ────────────────────────────────────────────────────────────

def load_reviews(json_path: str = JSON_PATH, n: int = 1000) -> pd.DataFrame:
    """
    Charge le fichier labeledReview.datasetFix.json.
    Champs : review (str), sentimen (int 0/1), translate (str)
    Retourne un DataFrame avec ReviewText, Sentiment, Note.
    """
    print(f"[cleaning] Lecture avis JSON : {json_path}")
    with open(json_path, encoding="utf-8") as f:
        raw = json.load(f)

    sample = random.sample(raw, min(n, len(raw)))

    rows = []
    for item in sample:
        sentiment_val = int(item.get("sentimen", item.get("sentiment", 1)))
        label = "positive" if sentiment_val == 1 else "negative"
        # Note simulée cohérente avec le sentiment
        note = round(random.uniform(3.5, 5.0), 1) if label == "positive" \
               else round(random.uniform(1.0, 2.5), 1)
        rows.append({
            "ReviewText": item.get("translate", item.get("review", ""))[:500],
            "Sentiment":  label,
            "Note":       note,
        })

    df = pd.DataFrame(rows)
    print(f"[cleaning] {len(df):,} avis chargés depuis JSON")
    return df
