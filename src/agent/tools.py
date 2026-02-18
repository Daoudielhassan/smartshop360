"""
src/agent/tools.py
===================
Outils (Tools) disponibles pour l'agent IA :

  Tool 1 — SQL Executor  (obligatoire)
      Exécute une requête SQL en lecture seule sur PostgreSQL.

  Tool 2 — Python Analysis  (optionnel)
      Statistiques descriptives et calculs avancés via Pandas
      sur les données retournées par Tool 1.

Sécurité : seul le DQL (SELECT) est autorisé.
"""

import re
import os
import sys
import statistics
import pandas as pd
from sqlalchemy import text

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.db_config import get_engine

# ────────────────────────────────────────────────────────────
#  Schéma injecté dans le System Prompt de l'agent
# ────────────────────────────────────────────────────────────

DB_SCHEMA = """
Base de données PostgreSQL — SmartShop 360

TABLES PRINCIPALES :
--------------------
customers(ClientID, Nom, Pays)
products(ProductID, ProductName, Category)
sales_facts(FactID, InvoiceNo, StockCode, Quantity, Revenue, Margin, InvoiceDate, CustomerID)
review_facts(ReviewID, ProductID, Rating, ReviewText, Sentiment, ReviewDate)
product_mapping(MappingID, ERP_StockCode, ERP_ProductName, Review_ProductCode,
                Review_ProductName, Category, GoldenRecordName)

VUES ANALYTIQUES (préférer pour les KPIs) :
--------------------------------------------
v_product_kpi(ProductID, ProductName, Category, CA, Marge, QuantiteVendue,
              Notemoyenne, NbAvis, AvisPositifs, AvisNegatifs, AvisNeutres)
v_customer_kpi(ClientID, Nom, Pays, NbCommandes, CA_Total, PanierMoyen)
v_alerts(ProductID, ProductName, Category, CA, Notemoyenne, NbAvis,
         AvisNegatifs, QuantiteVendue, Statut)
  → Statut : 'CRITIQUE' | 'A_SURVEILLER' | 'OK'
v_data_quality(Nb_Produits_ERP, Nb_Produits_Avis, Nb_Mappings, Nb_Avis_Total,
               Nb_Avis_Lies, Nb_Factures, Nb_Clients, Taux_Couverture_MDM)

NOTES :
- Sentiment dans review_facts : 'positive' | 'negative' | 'neutral'
- Rating dans review_facts : 1.0 à 5.0
- CA = Chiffre d'Affaires (Revenue total)
- Base PostgreSQL → utiliser ILIKE pour les recherches insensibles à la casse
- Pour joindre ventes et avis, toujours passer par product_mapping
- IMPORTANT : les noms de colonnes sont entre guillemets doubles (identifiants PostgreSQL)
  Ex : SELECT "ProductName", "CA" FROM v_product_kpi
"""

SYSTEM_PROMPT = f"""Tu es un Data Analyst expert pour SmartShop 360, un e-commerçant B2C spécialisé en Décoration & Cadeaux.

Tu as accès à une base de données PostgreSQL avec le schéma suivant :
{DB_SCHEMA}

Ta mission :
1. Comprendre la question métier posée en français
2. Générer une requête SQL valide et optimisée pour PostgreSQL
3. Analyser les résultats et formuler une réponse claire en français

Format de réponse OBLIGATOIRE (JSON strict) :
{{
  "sql": "SELECT ...",
  "reasoning": "Explication courte de l'approche SQL choisie",
  "answer_template": "Template de réponse à compléter avec les données"
}}

Règles SQL :
- Utiliser uniquement des SELECT (pas de INSERT/UPDATE/DELETE)
- Préférer les vues analytiques (v_product_kpi, v_customer_kpi, v_alerts)
- Limiter les résultats à LIMIT 20
- Arrondir les montants avec ROUND(x, 2)
- Joindre ventes et avis via product_mapping
- Mettre les noms de colonnes entre guillemets doubles dans les ORDER BY / HAVING
"""


# ────────────────────────────────────────────────────────────
#  Tool 1 : SQL Executor (READ ONLY)
# ────────────────────────────────────────────────────────────

_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def execute_sql(query: str) -> dict:
    """
    Exécute une requête SQL en lecture seule sur PostgreSQL.
    Bloque toute instruction de mutation (DML/DDL).
    """
    if _FORBIDDEN.search(query):
        return {
            "success": False,
            "error":   "Requête non autorisée (écriture bloquée — READ ONLY).",
            "data":    [],
            "columns": [],
            "row_count": 0,
        }
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(query))
            columns = list(result.keys())
            rows    = [dict(zip(columns, row)) for row in result.fetchmany(200)]
        return {
            "success":   True,
            "data":      rows,
            "columns":   columns,
            "row_count": len(rows),
        }
    except Exception as e:
        return {
            "success":   False,
            "error":     str(e),
            "data":      [],
            "columns":   [],
            "row_count": 0,
        }


# ────────────────────────────────────────────────────────────
#  Tool 2 : Python Analysis (calculs avancés)
# ────────────────────────────────────────────────────────────

def python_analysis(data: list, analysis_type: str = "summary") -> dict:
    """
    Calculs statistiques complémentaires sur les données renvoyées par SQL.

    analysis_type :
      - "summary"     : min / max / moyenne / médiane par colonne numérique
      - "correlation" : corrélation linéaire entre toutes les colonnes numériques
      - "trend"       : (si colonnes Date + Revenue) détection de tendance
    """
    if not data:
        return {"result": "Aucune donnée à analyser"}

    df = pd.DataFrame(data)
    numeric_df = df.select_dtypes(include="number")

    if analysis_type == "summary":
        summary = {}
        for col in numeric_df.columns:
            vals = numeric_df[col].dropna().tolist()
            if vals:
                summary[col] = {
                    "min":     round(min(vals), 2),
                    "max":     round(max(vals), 2),
                    "moyenne": round(statistics.mean(vals), 2),
                    "mediane": round(statistics.median(vals), 2),
                }
        return {"result": summary}

    if analysis_type == "correlation":
        if len(numeric_df.columns) >= 2:
            corr = numeric_df.corr().round(3).to_dict()
            return {"result": corr}
        return {"result": "Pas assez de colonnes numériques pour la corrélation"}

    if analysis_type == "trend":
        date_cols = [c for c in df.columns if "date" in c.lower()]
        rev_cols  = [c for c in df.columns if c.lower() in ("revenue", "ca", "ca_total")]
        if date_cols and rev_cols:
            df[date_cols[0]] = pd.to_datetime(df[date_cols[0]], errors="coerce")
            monthly = (
                df.set_index(date_cols[0])[rev_cols[0]]
                .resample("ME").sum().round(2)
            )
            return {"result": monthly.to_dict()}
        return {"result": "Colonnes Date/Revenue non trouvées pour l'analyse de tendance"}

    return {"result": f"Type d'analyse '{analysis_type}' non reconnu"}
