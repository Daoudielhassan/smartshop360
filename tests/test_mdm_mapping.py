"""
tests/test_mdm_mapping.py
===========================
Tests pour src/etl/mdm_mapping.py
"""

import pytest
import pandas as pd
import numpy as np
from src.etl.mdm_mapping import (
    fuzzy_match_tfidf,
    build_product_mapping,
    infer_category,
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def erp_names():
    return [
        "WHITE HANGING HEART T-LIGHT HOLDER",
        "JUMBO BAG RED RETROSPOT",
        "CREAM CUPID HEARTS COAT HANGER",
        "KNITTED UNION FLAG HOT WATER BOTTLE",
        "RED WOOLLY HOTTIE WHITE HEART",
    ]


@pytest.fixture
def review_names():
    return [
        "White Hanging Heart Tlight Holder",  # quasi-identique #0
        "Jumbo Bag Red Retro Spot",            # quasi-identique #1
        "Cupid Hearts Coat",                   # partiel #2
        "Knitted Union Flag",                  # partiel #3
        "Something Completely Different Item", # bruit
    ]


@pytest.fixture
def sample_products_erp(erp_names):
    return [
        (f"SC{i+1:03d}", name, "Category")
        for i, name in enumerate(erp_names)
    ]


@pytest.fixture
def sample_reviews_df(review_names):
    return pd.DataFrame({
        "ProductName": review_names,
        "Score": [4.5, 3.8, 4.0, 4.2, 3.1],
    })


# ─────────────────────────────────────────────────────────────
#  Tests infer_category
# ─────────────────────────────────────────────────────────────

class TestInferCategory:

    def test_kitchen_category(self):
        cat = infer_category("GLASS STAR TABLE LAMP")
        assert isinstance(cat, str) and len(cat) > 0

    def test_returns_string_for_any_input(self):
        for name in ["unknown xyz 1234", "", "bag red", "mug cup"]:
            assert isinstance(infer_category(name), str)

    def test_consistent_on_same_input(self):
        assert infer_category("RED MUG") == infer_category("RED MUG")


# ─────────────────────────────────────────────────────────────
#  Tests fuzzy_match_tfidf
# ─────────────────────────────────────────────────────────────

class TestFuzzyMatchTfidf:

    def test_returns_dict(self, erp_names, review_names):
        result = fuzzy_match_tfidf(erp_names, review_names)
        assert isinstance(result, dict)

    def test_keys_are_valid_indices(self, erp_names, review_names):
        result = fuzzy_match_tfidf(erp_names, review_names)
        for k in result:
            assert 0 <= k < len(erp_names)

    def test_values_are_tuples(self, erp_names, review_names):
        result = fuzzy_match_tfidf(erp_names, review_names)
        for v in result.values():
            assert isinstance(v, tuple) and len(v) == 2

    def test_scores_between_0_and_1(self, erp_names, review_names):
        result = fuzzy_match_tfidf(erp_names, review_names)
        for _, score in result.values():
            assert 0.0 <= score <= 1.0

    def test_obvious_match_detected(self, erp_names, review_names):
        """Le premier nom ERP et le premier nom review sont quasi-identiques."""
        result = fuzzy_match_tfidf(erp_names, review_names, threshold=0.3)
        assert 0 in result, "Le match évident (index 0) devrait être trouvé"
        matched_idx, score = result[0]
        assert matched_idx == 0
        assert score > 0.5

    def test_threshold_filter(self, erp_names, review_names):
        result_low  = fuzzy_match_tfidf(erp_names, review_names, threshold=0.01)
        result_high = fuzzy_match_tfidf(erp_names, review_names, threshold=0.99)
        assert len(result_low) >= len(result_high)

    def test_empty_inputs(self):
        result = fuzzy_match_tfidf([], [])
        assert result == {}


# ─────────────────────────────────────────────────────────────
#  Tests build_product_mapping
# ─────────────────────────────────────────────────────────────

class TestBuildProductMapping:

    def test_returns_dataframe(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(sample_products_erp, sample_reviews_df)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(sample_products_erp, sample_reviews_df)
        for col in ["ERP_StockCode", "ERP_ProductName", "Category"]:
            assert col in df.columns, f"Colonne manquante: {col}"

    def test_row_count_matches_erp(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(sample_products_erp, sample_reviews_df)
        assert len(df) == len(sample_products_erp)

    def test_fuzzy_columns_present_when_enabled(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(
            sample_products_erp, sample_reviews_df,
            use_fuzzy=True
        )
        assert "MatchScore" in df.columns
        assert "MatchStrategy" in df.columns

    def test_no_duplicate_stockcodes(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(sample_products_erp, sample_reviews_df)
        assert not df["ERP_StockCode"].duplicated().any()

    def test_without_fuzzy(self, sample_products_erp, sample_reviews_df):
        df = build_product_mapping(
            sample_products_erp, sample_reviews_df,
            use_fuzzy=False
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(sample_products_erp)
