"""
tests/test_cleaning.py
========================
Tests unitaires pour src/etl/cleaning.py
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from src.etl.cleaning import (
    clean_transactions,
    extract_top50_products,
    extract_customers,
)


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_raw_transactions():
    return pd.DataFrame({
        "InvoiceNo":   ["INV001", "INV002", "C00003", "INV004", "INV005"],
        "StockCode":   ["A001",   "B002",   "A001",   "C003",   "B002"],
        "Description": ["Mug A", "Bag B",  "Mug A",  "Lamp C", "Bag B"],
        "Quantity":    [5,        3,        -2,       2,        0],
        "InvoiceDate": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03",
                                        "2023-01-04", "2023-01-05"]),
        "UnitPrice":   [10.0,     20.0,     10.0,     15.0,     20.0],
        "CustomerID":  ["C001",   "C002",   "C001",   None,     "C002"],
        "Country":     ["UK",     "FR",     "UK",     "DE",     "FR"],
    })


@pytest.fixture
def sample_clean_transactions():
    """Transactions propres (après nettoyage)."""
    np.random.seed(42)
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "InvoiceNo":   ["INV001", "INV002", "INV003"] * 20,
        "StockCode":   ["A001", "B002", "C003"] * 20,
        "Description": ["WHITE HANGING HEART T-LIGHT HOLDER",
                         "JUMBO BAG RED RETROSPOT",
                         "CREAM CUPID HEARTS COAT HANGER"] * 20,
        "Quantity":    [5, 3, 2] * 20,
        "InvoiceDate": pd.date_range("2023-01-01", periods=60, freq="D"),
        "UnitPrice":   [10.0, 20.0, 15.0] * 20,
        "CustomerID":  ["C001", "C002", "C003"] * 20,
        "Country":     ["UK", "FR", "DE"] * 20,
        "Revenue":     [50.0, 60.0, 30.0] * 20,
        "Margin":      [15.0, 20.0, 10.0] * 20,
    })
    return df


# ─────────────────────────────────────────────────────────────
#  Tests clean_transactions
# ─────────────────────────────────────────────────────────────

class TestCleanTransactions:

    def test_removes_cancelled_invoices(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert not result["InvoiceNo"].str.startswith("C").any()

    def test_removes_zero_or_negative_quantity(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert (result["Quantity"] > 0).all()

    def test_removes_null_customer_id(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert result["CustomerID"].notna().all()

    def test_computes_revenue(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert "Revenue" in result.columns
        assert (result["Revenue"] > 0).all()

    def test_computes_margin(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert "Margin" in result.columns
        assert (result["Margin"] > 0).all()

    def test_result_has_fewer_rows_than_input(self, sample_raw_transactions):
        result = clean_transactions(sample_raw_transactions)
        assert len(result) < len(sample_raw_transactions)


# ─────────────────────────────────────────────────────────────
#  Tests extract_top50_products
# ─────────────────────────────────────────────────────────────

class TestExtractTop50Products:

    def test_returns_list(self, sample_clean_transactions):
        result = extract_top50_products(sample_clean_transactions)
        assert isinstance(result, list)

    def test_each_item_is_tuple_of_3(self, sample_clean_transactions):
        result = extract_top50_products(sample_clean_transactions)
        for item in result:
            assert len(item) == 3

    def test_stockcodes_are_unique(self, sample_clean_transactions):
        result = extract_top50_products(sample_clean_transactions)
        codes = [r[0] for r in result]
        assert len(codes) == len(set(codes)), "StockCodes doivent être uniques"

    def test_returns_at_most_50(self, sample_clean_transactions):
        result = extract_top50_products(sample_clean_transactions)
        assert len(result) <= 50

    def test_category_is_string(self, sample_clean_transactions):
        result = extract_top50_products(sample_clean_transactions)
        for _, _, cat in result:
            assert isinstance(cat, str) and len(cat) > 0


# ─────────────────────────────────────────────────────────────
#  Tests extract_customers
# ─────────────────────────────────────────────────────────────

class TestExtractCustomers:

    def test_returns_dataframe(self, sample_clean_transactions):
        result = extract_customers(sample_clean_transactions)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns(self, sample_clean_transactions):
        result = extract_customers(sample_clean_transactions)
        assert {"ClientID", "Pays", "Nom"}.issubset(result.columns)

    def test_no_duplicate_client_ids(self, sample_clean_transactions):
        result = extract_customers(sample_clean_transactions)
        assert result["ClientID"].duplicated().sum() == 0

    def test_nom_format(self, sample_clean_transactions):
        result = extract_customers(sample_clean_transactions)
        assert result["Nom"].str.startswith("Client_").all()
