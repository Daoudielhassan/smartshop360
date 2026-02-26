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

    if analysis_type == "segmentation":
        """
        Segmentation clients rentables × satisfaits.
        Divise les clients en 4 quadrants selon la médiane de CA_Total et NoteMoyenne :
          Champions          — CA élevé  + Note élevée
          Déçus Rentables    — CA élevé  + Note faible
          Fans Peu Dépensiers— CA faible + Note élevée
          Inactifs           — CA faible + Note faible
        """
        # Détection automatique des colonnes CA et Note
        ca_col   = next((c for c in df.columns if c.lower() in
                         ("ca_total", "ca", "revenue", "chiffre_affaires", "montant")), None)
        note_col = next((c for c in df.columns if c.lower() in
                         ("notemoyenne", "note_moyenne", "notmoyenne", "rating",
                          "rating_moyen", "note")), None)
        # Fallback : première colonne numérique contenant "ca" ou "note"
        if not ca_col:
            ca_col = next((c for c in df.columns if "ca" in c.lower() or "revenu" in c.lower()), None)
        if not note_col:
            note_col = next((c for c in df.columns if "note" in c.lower() or "rating" in c.lower()), None)

        if not ca_col or not note_col:
            return {"result": f"Colonnes CA ({ca_col}) et/ou Note ({note_col}) introuvables pour la segmentation. Colonnes disponibles : {list(df.columns)}"}

        df[ca_col]   = pd.to_numeric(df[ca_col],   errors="coerce")
        df[note_col] = pd.to_numeric(df[note_col], errors="coerce")
        df = df.dropna(subset=[ca_col, note_col])

        med_ca   = df[ca_col].median()
        med_note = df[note_col].median()

        _LABELS = {
            (True,  True):  "Champions",
            (True,  False): "Déçus Rentables",
            (False, True):  "Fans Peu Dépensiers",
            (False, False): "Inactifs",
        }
        df["_Segment"] = df.apply(
            lambda r: _LABELS[(r[ca_col] >= med_ca, r[note_col] >= med_note)], axis=1
        )

        # Colonne label (Nom client, ou première colonne string)
        label_col = next((c for c in df.columns if c.lower() in ("nom", "name", "clientid")), None)
        if not label_col:
            label_col = next((c for c in df.select_dtypes(include="object").columns
                              if c not in ("_Segment",)), None)

        segments: dict = {}
        for seg_name in ["Champions", "Déçus Rentables", "Fans Peu Dépensiers", "Inactifs"]:
            sub = df[df["_Segment"] == seg_name]
            top: list = []
            if label_col and len(sub) > 0:
                top = (sub.nlargest(3, ca_col)[[label_col, ca_col, note_col]]
                         .round(2).to_dict("records"))
            segments[seg_name] = {
                "count":       int(len(sub)),
                "CA_moyen":    round(float(sub[ca_col].mean()),   2) if len(sub) else 0.0,
                "Note_moyenne": round(float(sub[note_col].mean()), 2) if len(sub) else 0.0,
                "top_clients": top,
            }

        return {
            "result": {
                "type":       "segmentation",
                "segments":   segments,
                "seuil_CA":   round(float(med_ca),   2),
                "seuil_note": round(float(med_note), 2),
                "total":      int(len(df)),
            }
        }

    if analysis_type == "rfm":
        """
        Scoring RFM simplifié (Récence / Fréquence / Montant).
        Attend des colonnes proches de : NbCommandes, CA_Total, (date optionnelle).
        """
        freq_col  = next((c for c in df.columns if "commande" in c.lower() or "orders" in c.lower()), None)
        mon_col   = next((c for c in df.columns if c.lower() in ("ca_total", "ca", "revenue")), None)
        if not freq_col or not mon_col:
            return {"result": "Colonnes NbCommandes et CA_Total requises pour RFM"}

        df[freq_col] = pd.to_numeric(df[freq_col], errors="coerce")
        df[mon_col]  = pd.to_numeric(df[mon_col],  errors="coerce")
        df = df.dropna(subset=[freq_col, mon_col])

        # Quintiles 1-5
        df["F_score"] = pd.qcut(df[freq_col], 5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
        df["M_score"] = pd.qcut(df[mon_col],  5, labels=[1,2,3,4,5], duplicates="drop").astype(float)
        df["RFM"]     = (df["F_score"] + df["M_score"]) / 2

        def _rfm_label(score):
            if score >= 4.5: return "VIP"
            if score >= 3.5: return "Fidèle"
            if score >= 2.5: return "Actif"
            return "À risque"

        df["Profil"] = df["RFM"].apply(_rfm_label)
        rfm_summary = df.groupby("Profil").agg(
            count=(mon_col, "count"),
            CA_moyen=(mon_col, "mean"),
        ).round(2).to_dict("index")

        return {"result": {"type": "rfm", "profils": rfm_summary, "total": int(len(df))}}

    if analysis_type == "clustering":
        """
        Segmentation automatique par K-Means (clustering non supervisé).
        Détecte des groupes naturels de clients selon variables numériques.
        """
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
        except ImportError:
            return {"result": "scikit-learn requis pour le clustering (pip install scikit-learn)"}

        numeric_df = df.select_dtypes(include="number").dropna()

        if len(numeric_df) < 5:
            return {"result": "Dataset insuffisant pour clustering fiable"}

        features = numeric_df.copy()

        scaler   = StandardScaler()
        X_scaled = scaler.fit_transform(features)

        # Nombre de clusters adaptatif (2 à 4)
        k     = min(4, max(2, len(features) // 20))
        model = KMeans(n_clusters=k, random_state=42, n_init="auto")
        labels = model.fit_predict(X_scaled)

        features["_Cluster"] = labels
        df = df.loc[features.index].copy()
        df["_Cluster"] = labels

        # Profilage des clusters
        clusters: dict = {}
        for c in sorted(df["_Cluster"].unique()):
            sub = df[df["_Cluster"] == c]
            clusters[f"Cluster {c}"] = {
                "count":    int(len(sub)),
                "moyennes": (
                    sub.select_dtypes(include="number")
                    .drop(columns=["_Cluster"], errors="ignore")
                    .mean().round(2).to_dict()
                ),
            }

        # Importance des variables (dispersion inter-cluster)
        centers    = scaler.inverse_transform(model.cluster_centers_)
        feat_cols  = [c for c in features.columns if c != "_Cluster"]
        centers_df = pd.DataFrame(centers, columns=feat_cols)
        dispersion = centers_df.std().sort_values(ascending=False).round(2).to_dict()

        return {
            "result": {
                "type":               "clustering",
                "k":                  k,
                "clusters":           clusters,
                "variable_importance": dispersion,
                "total":              int(len(df)),
            }
        }

    return {"result": f"Type d'analyse '{analysis_type}' non reconnu"}
