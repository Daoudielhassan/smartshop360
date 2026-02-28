"""
src/etl/mdm_mapping.py
========================
Stratégie MDM — création de la table pivot PRODUCT_MAPPING.

Pipeline de matching par priorité décroissante :
  P1  Hard match   : normalisation StockCode → déduplication variantes (A/B/C)
  P2  Fuzzy match  : rapidfuzz token_sort_ratio sur descriptions ERP
  P3  Semantic     : SBERT (sentence-transformers) ReviewText → description produit
  P4  TF-IDF       : cosinus sklearn sur n-grammes de caractères (fallback)
  P5  Rank-based   : assignation cyclique déterministe (fallback final)

Contrôle via variable d'environnement :
  MDM_STRATEGY=poc       → P5 seulement (comportement POC historique)
  MDM_STRATEGY=fuzzy     → P1 + P2 + P4 + P5
  MDM_STRATEGY=semantic  → P1 + P2 + P3 + P4 + P5 (production, défaut)
"""

import os
import re
import random
import collections
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ─── Configuration ────────────────────────────────────────────────────────────
MDM_STRATEGY: str = os.getenv("MDM_STRATEGY", "semantic").lower()

FUZZY_THRESHOLD  = float(os.getenv("MDM_FUZZY_THRESHOLD",  "80"))    # 0–100 (rapidfuzz)
TFIDF_THRESHOLD  = float(os.getenv("MDM_TFIDF_THRESHOLD",  "0.25"))  # 0–1
SBERT_THRESHOLD  = float(os.getenv("MDM_SBERT_THRESHOLD",  "0.10"))  # 0–1
SBERT_MODEL      = os.getenv("MDM_SBERT_MODEL", "all-MiniLM-L6-v2")

# Mots-clés → catégorie (ordre de priorité, insensible à la casse)
_CATEGORY_RULES: list[tuple[list[str], str]] = [
    (["bag", "tote", "purse", "handbag"],                   "Sacs & Accessoires"),
    (["lamp", "light", "lantern", "candle", "holder"],       "Luminaires"),
    (["mug", "cup", "plate", "bowl", "glass", "jar"],        "Art de la Table"),
    (["frame", "sign", "plaque", "poster", "wall"],          "Decoration Murale"),
    (["box", "wrap", "ribbon", "gift", "tag", "basket"],     "Emballages & Cadeaux"),
    (["flower", "heart", "rose", "romantic", "love"],        "Nature & Romantique"),
]


def infer_category(product_name: str) -> str:
    """Infère la catégorie d'un produit à partir de son nom (règles mots-clés)."""
    name_lower = product_name.lower()
    for keywords, category in _CATEGORY_RULES:
        if any(kw in name_lower for kw in keywords):
            return category
    return "Divers"


# ─────────────────────────────────────────────────────────────
#  P1 — HARD MATCH : normalisation StockCode
# ─────────────────────────────────────────────────────────────

def normalize_stockcode(code: str) -> str:
    """Normalise un StockCode pour regrouper les variantes (85123A, 85123B → 85123)."""
    return re.sub(r"[A-Za-z]+$", "", str(code).strip().upper())


def hard_match_groups(products_erp: list) -> dict[str, list[int]]:
    """
    P1 — Regroupe les indices ERP par StockCode normalisé.

    Returns
    -------
    dict { normalized_code → [erp_idx, ...] }
    """
    groups: dict[str, list[int]] = {}
    for idx, (stock_code, _, _) in enumerate(products_erp):
        key = normalize_stockcode(stock_code)
        groups.setdefault(key, []).append(idx)
    n_variants = sum(1 for g in groups.values() if len(g) > 1)
    print(f"[mdm-P1] Hard match : {len(products_erp)} produits → {len(groups)} groupes "
          f"({n_variants} groupes avec variantes)")
    return groups


# ─────────────────────────────────────────────────────────────
#  P2 — FUZZY MATCH : rapidfuzz Levenshtein
# ─────────────────────────────────────────────────────────────

def fuzzy_match_rapidfuzz(
    erp_names: list[str],
    threshold: float = FUZZY_THRESHOLD,
) -> dict[int, tuple[int, float]]:
    """
    P2 — Détecte les doublons sémantiques dans le catalogue ERP.

    Returns
    -------
    dict { erp_idx → (golden_idx, score_0_1) }
    """
    try:
        from rapidfuzz import fuzz
    except ImportError:
        print("[mdm-P2] rapidfuzz non disponible — pip install rapidfuzz")
        return {}

    matches: dict[int, tuple[int, float]] = {}
    n = len(erp_names)
    for i in range(n):
        best_score, best_j = 0.0, -1
        for j in range(n):
            if i == j:
                continue
            score = fuzz.token_sort_ratio(erp_names[i], erp_names[j])
            if score > best_score:
                best_score, best_j = score, j
        if best_score >= threshold and best_j >= 0 and best_j < i:
            matches[i] = (best_j, round(best_score / 100.0, 4))

    print(f"[mdm-P2] rapidfuzz : {len(matches)}/{n} doublons détectés (seuil={threshold}%)")
    return matches


# ─────────────────────────────────────────────────────────────
#  P3 — SEMANTIC MATCH : SBERT (sentence-transformers)
# ─────────────────────────────────────────────────────────────

def semantic_match_sbert(
    review_texts: list[str],
    product_names: list[str],
    threshold: float = SBERT_THRESHOLD,
    model_name: str = SBERT_MODEL,
    batch_size: int = 128,
) -> list[int]:
    """
    P3 — Assigne chaque avis au produit sémantiquement le plus proche.
    Utilise sentence-transformers + cosinus sur embeddings normalisés.

    Returns
    -------
    list[int] — indice produit par avis (-1 si sous le seuil → fallback).
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity as cos_sim
    except ImportError:
        print("[mdm-P3] sentence-transformers / sklearn non disponible — P3 sauté")
        return [-1] * len(review_texts)

    if not review_texts or not product_names:
        return [-1] * len(review_texts)

    print(f"[mdm-P3] SBERT '{model_name}' — encodage {len(product_names)} produits "
          f"+ {len(review_texts)} avis ...")
    model      = SentenceTransformer(model_name)
    prod_emb   = model.encode(product_names, batch_size=batch_size,
                               show_progress_bar=False, normalize_embeddings=True)
    review_emb = model.encode(review_texts,  batch_size=batch_size,
                               show_progress_bar=False, normalize_embeddings=True)

    scores = cos_sim(review_emb, prod_emb)  # (n_reviews, n_products)
    assignments: list[int] = []
    n_matched = 0
    for row in scores:
        best_idx   = int(np.argmax(row))
        best_score = float(row[best_idx])
        if best_score >= threshold:
            assignments.append(best_idx)
            n_matched += 1
        else:
            assignments.append(-1)

    coverage = n_matched / max(len(review_texts), 1) * 100
    print(f"[mdm-P3] SBERT : {n_matched}/{len(review_texts)} avis assignés "
          f"(couverture {coverage:.1f}%, seuil={threshold})")
    return assignments


# ─────────────────────────────────────────────────────────────
#  P4 — TF-IDF COSINUS (fallback sklearn)
# ─────────────────────────────────────────────────────────────

def fuzzy_match_tfidf(
    erp_names: list[str],
    review_names: list[str],
    threshold: float = TFIDF_THRESHOLD,
) -> dict[int, tuple[int, float]]:
    """
    P4 — Matching TF-IDF + cosinus sur n-grammes de caractères.

    Returns
    -------
    dict { erp_idx → (review_idx, score_0_1) }
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        print("[mdm-P4] scikit-learn non disponible — TF-IDF ignoré")
        return {}

    if not erp_names or not review_names:
        return {}

    all_names    = erp_names + review_names
    vectorizer   = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), lowercase=True)
    tfidf_matrix = vectorizer.fit_transform(all_names).toarray()  # type: ignore

    n_erp      = len(erp_names)
    erp_vec    = tfidf_matrix[:n_erp]
    rev_vec    = tfidf_matrix[n_erp:]
    sim_matrix = cosine_similarity(erp_vec, rev_vec)

    matches: dict[int, tuple[int, float]] = {}
    for erp_idx in range(n_erp):
        best_rev_idx = int(np.argmax(sim_matrix[erp_idx]))
        best_score   = float(sim_matrix[erp_idx, best_rev_idx])
        if best_score >= threshold:
            matches[erp_idx] = (best_rev_idx, round(best_score, 4))

    n_matched = len(matches)
    print(f"[mdm-P4] TF-IDF : {n_matched}/{n_erp} matchés (seuil={threshold})")

    return matches


# ─────────────────────────────────────────────────────────────
#  BUILD PRODUCT MAPPING — pipeline P1 → P2 → P4 → P5
# ─────────────────────────────────────────────────────────────

def build_product_mapping(
    products_erp: list,
    reviews_df: pd.DataFrame,
    use_fuzzy: bool = True,
    fuzzy_threshold: float | None = None,
) -> pd.DataFrame:
    """
    Crée la table PRODUCT_MAPPING (Golden Records) via le pipeline MDM.

    Stratégie contrôlée par MDM_STRATEGY (env) :
      poc      → P5 rank seulement
      fuzzy    → P1 + P2 + P4 + P5
      semantic → P1 + P2 + P3 + P4 + P5 (P3 appliqué dans attach_product_to_reviews)

    Returns
    -------
    DataFrame : MappingID, ERP_StockCode, ERP_ProductName, Review_ProductCode,
                Review_ProductName, Category, GoldenRecordName, MatchScore, MatchStrategy
    """
    strategy  = MDM_STRATEGY
    n_erp     = len(products_erp)
    threshold = fuzzy_threshold if fuzzy_threshold is not None else FUZZY_THRESHOLD
    print(f"[mdm] Pipeline MDM — stratégie : {strategy.upper()} ({n_erp} produits ERP)")

    # ── P1 : déduplication variantes StockCode ────────────────────────────────
    p1_golden: dict[int, int] = {}   # erp_idx → canonical_idx
    if strategy in ("fuzzy", "semantic"):
        groups = hard_match_groups(products_erp)
        for indices in groups.values():
            canonical = indices[0]
            for idx in indices[1:]:
                p1_golden[idx] = canonical

    # ── P2 : doublons rapidfuzz ──────────────────────────────────────────────
    p2_matches: dict[int, tuple[int, float]] = {}
    if strategy in ("fuzzy", "semantic") and use_fuzzy and n_erp > 1:
        erp_names  = [desc for _, desc, _ in products_erp]
        p2_matches = fuzzy_match_rapidfuzz(erp_names, threshold=threshold)

    # ── P4 : TF-IDF cosinus sur noms ERP ─────────────────────────────────────
    p4_matches: dict[int, tuple[int, float]] = {}
    if strategy != "poc" and use_fuzzy and n_erp > 1:
        erp_names   = [desc for _, desc, _ in products_erp]
        p4_matches  = fuzzy_match_tfidf(erp_names, erp_names, threshold=TFIDF_THRESHOLD)

    # ── Construction Golden Records ───────────────────────────────────────────
    rows  = []
    stats = {"hard": 0, "fuzzy": 0, "tfidf": 0, "rank": 0}

    for i, (stock_code, description, category) in enumerate(products_erp):
        review_pid  = f"REV_{(i % 1000) + 1:04d}"
        score       = 0.0
        match_strat = "rank"

        if i in p1_golden:
            canonical_idx = p1_golden[i]
            review_pid    = f"REV_{(canonical_idx % 1000) + 1:04d}"
            score         = 1.0
            match_strat   = "hard"
            stats["hard"] += 1
        elif i in p2_matches:
            twin_idx, p2_score = p2_matches[i]
            review_pid   = f"REV_{(twin_idx % 1000) + 1:04d}"
            score        = p2_score
            match_strat  = "fuzzy"
            stats["fuzzy"] += 1
        elif i in p4_matches:
            rev_idx, tf_score = p4_matches[i]
            review_pid   = f"REV_{(rev_idx % 1000) + 1:04d}"
            score        = tf_score
            match_strat  = "tfidf"
            stats["tfidf"] += 1
        else:
            stats["rank"] += 1

        rows.append({
            "MappingID":           i + 1,
            "ERP_StockCode":       stock_code,
            "ERP_ProductName":     description,
            "Review_ProductCode":  review_pid,
            "Review_ProductName":  description,
            "Category":            category,
            "GoldenRecordName":    description,
            "MatchScore":          round(score, 4),
            "MatchStrategy":       match_strat,
        })

    df = pd.DataFrame(rows)
    print(f"[mdm] {len(df)} Golden Records — "
          f"hard: {stats['hard']}, fuzzy: {stats['fuzzy']}, "
          f"tfidf: {stats['tfidf']}, rank: {stats['rank']}")
    return df


# ─────────────────────────────────────────────────────────────
#  ATTACH REVIEWS — P3 SBERT ou TF-IDF word ou round-robin
# ─────────────────────────────────────────────────────────────

def attach_product_to_reviews(
    reviews_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assigne chaque avis client au bon Review_ProductCode.

    semantic → SBERT cosinus (ReviewText → description produit ERP)
    fuzzy    → TF-IDF word bigrammes (ReviewText → description produit ERP)
    poc      → round-robin cyclique (comportement historique)
    """
    review_codes  = mapping_df["Review_ProductCode"].tolist()
    product_names = mapping_df["GoldenRecordName"].tolist()
    reviews_df    = reviews_df.copy()
    strategy      = MDM_STRATEGY
    assigned      = False

    # ── P3 : SBERT ─────────────────────────────────────────────────────────
    if strategy == "semantic":
        review_texts = reviews_df["ReviewText"].fillna("").tolist()
        sbert_assign = semantic_match_sbert(
            review_texts  = review_texts,
            product_names = product_names,
            threshold     = SBERT_THRESHOLD,
            model_name    = SBERT_MODEL,
        )
        if any(idx >= 0 for idx in sbert_assign):
            valid           = [idx for idx in sbert_assign if idx >= 0]
            most_common_idx = collections.Counter(valid).most_common(1)[0][0]
            codes = [
                review_codes[min(prod_idx if prod_idx >= 0 else most_common_idx,
                                 len(review_codes) - 1)]
                for prod_idx in sbert_assign
            ]
            reviews_df["Review_ProductCode"] = codes
            assigned = True

    # ── P4 : TF-IDF word (fallback SBERT) ───────────────────────────────────
    if not assigned and strategy in ("fuzzy", "semantic"):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            review_texts = reviews_df["ReviewText"].fillna("").tolist()
            vectorizer   = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                           lowercase=True, max_features=5000)
            all_texts    = product_names + review_texts
            tfidf_all    = vectorizer.fit_transform(all_texts).toarray()
            prod_vec     = tfidf_all[:len(product_names)]
            rev_vec      = tfidf_all[len(product_names):]
            sim          = cosine_similarity(rev_vec, prod_vec)
            tfidf_assign = [int(np.argmax(row)) for row in sim]
            reviews_df["Review_ProductCode"] = [
                review_codes[min(idx, len(review_codes) - 1)] for idx in tfidf_assign
            ]
            assigned = True
            print(f"[mdm-P4] {len(tfidf_assign)} avis assignés via TF-IDF word")
        except Exception as e:
            print(f"[mdm-P4] TF-IDF word matching échoué : {e}")

    # ── P5 : Round-robin (fallback final) ───────────────────────────────────
    if not assigned:
        reviews_df["Review_ProductCode"] = [
            review_codes[i % len(review_codes)] for i in range(len(reviews_df))
        ]
        print(f"[mdm-P5] {len(reviews_df)} avis assignés via round-robin")

    # Dates d'avis simulées sur les 18 derniers mois
    base_date = datetime(2024, 1, 1)
    reviews_df["ReviewDate"] = [
        base_date + timedelta(days=random.randint(0, 548))
        for _ in range(len(reviews_df))
    ]
    reviews_df.insert(0, "ReviewID", range(1, len(reviews_df) + 1))
    return reviews_df


# ─────────────────────────────────────────────────────────────
#  DESCRIPTION DE LA STRATÉGIE (dashboard)
# ─────────────────────────────────────────────────────────────

def mdm_strategy_description() -> dict:
    """Description de la stratégie MDM active (dashboard Data Quality)."""
    active = MDM_STRATEGY
    return {
        "poc": {
            "name":        "P5 — Rank-based Matching (POC)",
            "description": "Le Top-50 ERP est associé rang-par-rang aux 50 avis les plus récents.",
            "fiabilite":   "⚠ Artificielle — démo uniquement",
            "active":      active == "poc",
        },
        "hard": {
            "name":        "P1 — Hard Match — Normalisation StockCode",
            "description": (
                "Regroupement déterministe des variantes produit (85123A, 85123B → 85123). "
                "Fiabilité 100% pour les variantes de taille/couleur."
            ),
            "fiabilite":   "✅ Production — Priorité 1",
            "active":      active in ("fuzzy", "semantic"),
        },
        "fuzzy": {
            "name":        "P2 — Fuzzy Matching (rapidfuzz)",
            "description": (
                "Levenshtein token_sort_ratio sur descriptions ERP. "
                f"Seuil = {FUZZY_THRESHOLD}% — détecte les doublons entre produits similaires."
            ),
            "fiabilite":   "✅ Production — Priorité 2",
            "active":      active in ("fuzzy", "semantic"),
        },
        "semantic": {
            "name":        "P3 — Semantic Matching (SBERT)",
            "description": (
                f"Sentence-BERT ({SBERT_MODEL}) — cosinus entre texte d'avis et "
                f"description produit ERP. Seuil = {SBERT_THRESHOLD}. "
                "Chaque avis est assigné au produit sémantiquement le plus proche."
            ),
            "fiabilite":   "✅ Production — Priorité 3" if active == "semantic"
                           else "⚙ Désactivé (MDM_STRATEGY != semantic)",
            "active":      active == "semantic",
        },
        "tfidf": {
            "name":        "P4 — TF-IDF Cosinus (sklearn, fallback)",
            "description": (
                f"TF-IDF char n-grams (2-4) mapping ERP, word bigrammes assignation avis. "
                f"Seuil = {TFIDF_THRESHOLD}."
            ),
            "fiabilite":   "✅ Production — Priorité 4 (fallback P3)",
            "active":      active != "poc",
        },
    }
