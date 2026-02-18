"""
src/etl/mdm_mapping.py
========================
Stratégie MDM — création de la table pivot PRODUCT_MAPPING.

Niveaux de matching (par ordre de priorité) :
  1. Hard match   : code EAN/GTIN commun (déterministe)
  2. Fuzzy match  : TF-IDF + similarité cosinus sur les noms (seuil configurable)
  3. IA match     : Embeddings Sentence-BERT (RAG — voir rag.py)
  4. Rank-based   : Fallback POC si aucun match trouvé
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)


# ─────────────────────────────────────────────────────────────
#  FUZZY MATCHING — TF-IDF + Similarité Cosinus
# ─────────────────────────────────────────────────────────────

def fuzzy_match_tfidf(
    erp_names: list[str],
    review_names: list[str],
    threshold: float = 0.35,
) -> dict[int, tuple[int, float]]:
    """
    Matching flou basé sur TF-IDF + similarité cosinus.

    Parameters
    ----------
    erp_names     : liste des noms produits ERP
    review_names  : liste des noms produits de la source avis
    threshold     : score minimum pour accepter un match (0–1)

    Returns
    -------
    dict { erp_idx → (review_idx, score) }
    Seuls les matches avec score >= threshold sont inclus.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("[mdm] scikit-learn non disponible — fuzzy matching ignoré")
        return {}

    all_names = erp_names + review_names
    vectorizer = TfidfVectorizer(
        analyzer    = "char_wb",   # n-grammes de caractères → robuste aux fautes
        ngram_range = (2, 4),
        lowercase   = True,
    )
    tfidf_matrix = vectorizer.fit_transform(all_names)

    n_erp    = len(erp_names)
    erp_vec  = tfidf_matrix[:n_erp]
    rev_vec  = tfidf_matrix[n_erp:]

    sim_matrix = cosine_similarity(erp_vec, rev_vec)   # shape (n_erp, n_review)

    matches = {}
    for erp_idx in range(n_erp):
        best_rev_idx = int(np.argmax(sim_matrix[erp_idx]))
        best_score   = float(sim_matrix[erp_idx, best_rev_idx])
        if best_score >= threshold:
            matches[erp_idx] = (best_rev_idx, best_score)

    n_matched = len(matches)
    print(
        f"[mdm] TF-IDF fuzzy matching : {n_matched}/{n_erp} matchés "
        f"(seuil={threshold}, {n_erp - n_matched} sans match → fallback rank)"
    )
    return matches


def build_product_mapping(
    products_erp: list,
    reviews_df: pd.DataFrame,
    use_fuzzy: bool = True,
    fuzzy_threshold: float = 0.35,
) -> pd.DataFrame:
    """
    Crée la table PRODUCT_MAPPING (Golden Records).

    Parameters
    ----------
    products_erp     : list of (StockCode, Description, Category)
    reviews_df       : DataFrame avec colonnes ReviewText, Sentiment, Note
    use_fuzzy        : active le matching TF-IDF (désactiver pour tests rapides)
    fuzzy_threshold  : score minimum TF-IDF pour valider un match

    Returns
    -------
    DataFrame avec colonnes :
        MappingID, ERP_StockCode, ERP_ProductName, Review_ProductCode,
        Review_ProductName, Category, GoldenRecordName, MatchScore, MatchStrategy
    """
    n = min(len(products_erp), 50)
    erp_names = [desc for _, desc, _ in products_erp[:n]]
    rev_names = [f"Product {i+1}" for i in range(n)]

    fuzzy_results: dict = {}
    if use_fuzzy and n > 1:
        fuzzy_results = fuzzy_match_tfidf(erp_names, rev_names, threshold=fuzzy_threshold)

    rows = []
    for i, (stock_code, description, category) in enumerate(products_erp[:n]):
        if i in fuzzy_results:
            rev_idx, score = fuzzy_results[i]
            match_strategy = "tfidf"
        else:
            rev_idx = i
            score   = 0.0
            match_strategy = "rank"

        review_pid = f"REV_{rev_idx + 1:03d}"
        rows.append({
            "MappingID":            i + 1,
            "ERP_StockCode":        stock_code,
            "ERP_ProductName":      description,
            "Review_ProductCode":   review_pid,
            "Review_ProductName":   description,
            "Category":             category,
            "GoldenRecordName":     description,
            "MatchScore":           round(score, 4),
            "MatchStrategy":        match_strategy,
        })

    df = pd.DataFrame(rows)
    n_tfidf = (df["MatchStrategy"] == "tfidf").sum()
    n_rank  = (df["MatchStrategy"] == "rank").sum()
    print(f"[mdm] {len(df)} Golden Records — TF-IDF: {n_tfidf}, rank-based: {n_rank}")
    return df


def attach_product_to_reviews(reviews_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    """
    Associe chaque avis à un produit via le Review_ProductCode.
    Distribue les avis uniformément sur les 50 produits mappés.
    """
    review_codes = mapping_df["Review_ProductCode"].tolist()
    reviews_df = reviews_df.copy()
    reviews_df["Review_ProductCode"] = [
        review_codes[i % len(review_codes)] for i in range(len(reviews_df))
    ]

    # Date d'avis simulée sur les 18 derniers mois
    base_date = datetime(2024, 1, 1)
    reviews_df["ReviewDate"] = [
        base_date + timedelta(days=random.randint(0, 548))
        for _ in range(len(reviews_df))
    ]

    # ReviewID unique
    reviews_df.insert(0, "ReviewID", range(1, len(reviews_df) + 1))
    return reviews_df


def mdm_strategy_description() -> dict:
    """
    Retourne la description textuelle de la stratégie MDM
    (pour affichage dans le dashboard Data Quality).
    """
    return {
        "poc": {
            "name": "Rank-based Matching (POC)",
            "description": "Le Top-50 ERP est associé rang-par-rang aux 50 avis les plus récents.",
            "fiabilite": "⚠️ Artificielle — démo uniquement",
        },
        "production_1": {
            "name": "Hard Match — Code EAN/GTIN",
            "description": "Réconciliation déterministe via code-barres universel. Fiabilité 100%.",
            "fiabilite": "✅ Production — Priorité 1",
        },
        "production_2": {
            "name": "Fuzzy Matching (thefuzz)",
            "description": "Distance de Levenshtein sur noms produits. Seuil > 90% = lien auto.",
            "fiabilite": "🟡 Production — Priorité 2",
        },
        "production_3": {
            "name": "Semantic Matching (Embeddings)",
            "description": "Sentence-BERT + similarité cosinus pour noms très différents.",
            "fiabilite": "🔵 Production — Priorité 3",
        },
    }
