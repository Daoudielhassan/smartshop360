"""
src/etl/data_quality.py
=========================
Règles de validation de la qualité des données inspirées de Great Expectations.
Fonctionne sans dépendance externe — peut être remplacé par great_expectations
en production via : pip install great-expectations

Chaque règle retourne un dict :
  { "rule": str, "passed": bool, "detail": str, "severity": "error"|"warning" }
"""

from __future__ import annotations
import pandas as pd
from typing import Any


# ─────────────────────────────────────────────────────────────
#  Moteur de validation léger
# ─────────────────────────────────────────────────────────────

class ExpectationResult:
    def __init__(self, rule: str, passed: bool, detail: str, severity: str = "error"):
        self.rule     = rule
        self.passed   = passed
        self.detail   = detail
        self.severity = severity

    def __repr__(self):
        icon = "" if self.passed else ("" if self.severity == "error" else "")
        return f"{icon} [{self.severity.upper()}] {self.rule} — {self.detail}"


class DataQualityReport:
    def __init__(self, name: str):
        self.name    = name
        self.results: list[ExpectationResult] = []

    def add(self, result: ExpectationResult):
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == "error")

    @property
    def errors(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed and r.severity == "error"]

    @property
    def warnings(self) -> list[ExpectationResult]:
        return [r for r in self.results if not r.passed and r.severity == "warning"]

    def print_report(self):
        print(f"\n{'='*60}")
        print(f" Data Quality Report — {self.name}")
        print(f"{'='*60}")
        for r in self.results:
            print(f"  {r}")
        status = " PASSED" if self.passed else " FAILED"
        print(f"\n  Résultat global : {status} ({len(self.errors)} erreur(s), {len(self.warnings)} avertissement(s))")
        print(f"{'='*60}\n")

    def to_dict(self) -> dict:
        return {
            "name":     self.name,
            "passed":   self.passed,
            "results":  [{"rule": r.rule, "passed": r.passed, "detail": r.detail, "severity": r.severity} for r in self.results],
            "errors":   len(self.errors),
            "warnings": len(self.warnings),
        }


# ─────────────────────────────────────────────────────────────
#  Fonctions d'expectation
# ─────────────────────────────────────────────────────────────

def expect_no_nulls(df: pd.DataFrame, col: str, severity: str = "error") -> ExpectationResult:
    n = df[col].isna().sum()
    return ExpectationResult(
        rule     = f"expect_no_nulls({col})",
        passed   = n == 0,
        detail   = f"{n} valeur(s) nulle(s) sur {len(df)} lignes",
        severity = severity,
    )


def expect_min_rows(df: pd.DataFrame, minimum: int, severity: str = "error") -> ExpectationResult:
    return ExpectationResult(
        rule     = f"expect_min_rows({minimum})",
        passed   = len(df) >= minimum,
        detail   = f"{len(df)} lignes (min attendu : {minimum})",
        severity = severity,
    )


def expect_unique(df: pd.DataFrame, col: str, severity: str = "error") -> ExpectationResult:
    n_dup = df[col].duplicated().sum()
    return ExpectationResult(
        rule     = f"expect_unique({col})",
        passed   = n_dup == 0,
        detail   = f"{n_dup} doublon(s) détecté(s)",
        severity = severity,
    )


def expect_values_in_set(df: pd.DataFrame, col: str, valid_set: set, severity: str = "error") -> ExpectationResult:
    invalid = ~df[col].isin(valid_set)
    n = invalid.sum()
    samples = df.loc[invalid, col].unique()[:5].tolist()
    return ExpectationResult(
        rule     = f"expect_values_in_set({col})",
        passed   = n == 0,
        detail   = f"{n} valeur(s) hors ensemble valide. Exemples : {samples}",
        severity = severity,
    )


def expect_column_between(df: pd.DataFrame, col: str, min_val: Any, max_val: Any, severity: str = "error") -> ExpectationResult:
    out = df[(df[col] < min_val) | (df[col] > max_val)]
    n   = len(out)
    return ExpectationResult(
        rule     = f"expect_column_between({col}, {min_val}, {max_val})",
        passed   = n == 0,
        detail   = f"{n} valeur(s) hors plage [{min_val}, {max_val}]",
        severity = severity,
    )


def expect_no_future_dates(df: pd.DataFrame, col: str, severity: str = "warning") -> ExpectationResult:
    now   = pd.Timestamp.now()
    future = df[df[col] > now]
    n     = len(future)
    return ExpectationResult(
        rule     = f"expect_no_future_dates({col})",
        passed   = n == 0,
        detail   = f"{n} ligne(s) avec date future",
        severity = severity,
    )


def expect_positive_values(df: pd.DataFrame, col: str, severity: str = "error") -> ExpectationResult:
    n = (df[col] <= 0).sum()
    return ExpectationResult(
        rule     = f"expect_positive_values({col})",
        passed   = n == 0,
        detail   = f"{n} valeur(s) <= 0",
        severity = severity,
    )


# ─────────────────────────────────────────────────────────────
#  Suites de règles métier
# ─────────────────────────────────────────────────────────────

def validate_transactions(df: pd.DataFrame) -> DataQualityReport:
    """Règles de qualité pour les transactions ERP."""
    report = DataQualityReport("transactions_erp")
    report.add(expect_min_rows(df, 1000))
    report.add(expect_no_nulls(df, "CustomerID"))
    report.add(expect_no_nulls(df, "StockCode"))
    report.add(expect_no_nulls(df, "InvoiceDate"))
    report.add(expect_positive_values(df, "Quantity"))
    report.add(expect_positive_values(df, "Revenue", severity="warning"))  # nettoyé en amont, warning seulement
    report.add(expect_no_future_dates(df, "InvoiceDate", severity="warning"))
    report.add(expect_column_between(df, "Revenue", 0.01, 100_000, severity="warning"))
    return report


def validate_reviews(df: pd.DataFrame) -> DataQualityReport:
    """Règles de qualité pour les avis clients."""
    report = DataQualityReport("reviews_json")
    report.add(expect_min_rows(df, 100))
    report.add(expect_no_nulls(df, "ReviewText"))
    report.add(expect_no_nulls(df, "Sentiment"))
    report.add(expect_values_in_set(df, "Sentiment", {"positive", "negative", "neutral"}))
    report.add(expect_column_between(df, "Note", 1.0, 5.0))
    return report


def validate_products(df: pd.DataFrame) -> DataQualityReport:
    """Règles de qualité pour les Golden Records produits."""
    report = DataQualityReport("products_golden_records")
    report.add(expect_min_rows(df, 1))
    report.add(expect_unique(df, "ProductID"))
    report.add(expect_no_nulls(df, "ProductName"))
    report.add(expect_no_nulls(df, "Category"))
    return report


def run_all_validations(transactions_df: pd.DataFrame, reviews_df: pd.DataFrame, products_df: pd.DataFrame) -> bool:
    """
    Exécute toutes les suites de validation et affiche les rapports.
    Retourne True si aucune erreur critique n'est détectée.
    """
    reports = [
        validate_transactions(transactions_df),
        validate_reviews(reviews_df),
        validate_products(products_df),
    ]
    all_passed = True
    for r in reports:
        r.print_report()
        if not r.passed:
            all_passed = False
    return all_passed
