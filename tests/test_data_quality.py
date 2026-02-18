"""
tests/test_data_quality.py
============================
Tests pour src/etl/data_quality.py
"""

import pytest
import pandas as pd
import numpy as np
from src.etl.data_quality import (
    expect_no_nulls,
    expect_min_rows,
    expect_unique,
    expect_values_in_set,
    expect_column_between,
    expect_no_future_dates,
    expect_positive_values,
    DataQualityReport,
    ExpectationResult,
)


# ─────────────────────────────────────────────────────────────
#  Tests des expectations individuelles
# ─────────────────────────────────────────────────────────────

class TestExpectNoNulls:
    def test_passes_clean_column(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        r = expect_no_nulls(df, "a")
        assert r.passed

    def test_fails_with_nulls(self):
        df = pd.DataFrame({"a": [1, None, 3]})
        r = expect_no_nulls(df, "a")
        assert not r.passed

    def test_error_message_contains_column_name(self):
        df = pd.DataFrame({"col_test": [None]})
        r = expect_no_nulls(df, "col_test")
        assert "col_test" in r.message


class TestExpectMinRows:
    def test_passes_with_enough_rows(self):
        df = pd.DataFrame({"x": range(100)})
        r = expect_min_rows(df, 50)
        assert r.passed

    def test_fails_with_too_few_rows(self):
        df = pd.DataFrame({"x": range(10)})
        r = expect_min_rows(df, 100)
        assert not r.passed

    def test_passes_at_exact_threshold(self):
        df = pd.DataFrame({"x": range(50)})
        r = expect_min_rows(df, 50)
        assert r.passed


class TestExpectUnique:
    def test_passes_unique_column(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4]})
        r = expect_unique(df, "id")
        assert r.passed

    def test_fails_duplicates(self):
        df = pd.DataFrame({"id": [1, 2, 2, 3]})
        r = expect_unique(df, "id")
        assert not r.passed


class TestExpectValuesInSet:
    def test_passes_valid_values(self):
        df = pd.DataFrame({"status": ["active", "inactive", "active"]})
        r = expect_values_in_set(df, "status", ["active", "inactive"])
        assert r.passed

    def test_fails_unexpected_values(self):
        df = pd.DataFrame({"status": ["active", "UNKNOWN", "inactive"]})
        r = expect_values_in_set(df, "status", ["active", "inactive"])
        assert not r.passed

    def test_message_contains_invalid_values(self):
        df = pd.DataFrame({"cat": ["A", "Z"]})
        r = expect_values_in_set(df, "cat", ["A", "B"])
        assert "Z" in r.message


class TestExpectColumnBetween:
    def test_passes_in_range(self):
        df = pd.DataFrame({"score": [1, 2, 3, 4, 5]})
        r = expect_column_between(df, "score", 0, 10)
        assert r.passed

    def test_fails_below_min(self):
        df = pd.DataFrame({"score": [-1, 2, 3]})
        r = expect_column_between(df, "score", 0, 10)
        assert not r.passed

    def test_fails_above_max(self):
        df = pd.DataFrame({"score": [1, 2, 15]})
        r = expect_column_between(df, "score", 0, 10)
        assert not r.passed

    def test_passes_at_boundaries(self):
        df = pd.DataFrame({"score": [0, 5, 10]})
        r = expect_column_between(df, "score", 0, 10)
        assert r.passed


class TestExpectNoFutureDates:
    def test_passes_past_dates(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2021-06-15"])
        })
        r = expect_no_future_dates(df, "date")
        assert r.passed

    def test_fails_future_date(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-01", "2099-01-01"])
        })
        r = expect_no_future_dates(df, "date")
        assert not r.passed


class TestExpectPositiveValues:
    def test_passes_positive(self):
        df = pd.DataFrame({"qty": [1, 5, 100]})
        r = expect_positive_values(df, "qty")
        assert r.passed

    def test_fails_zero(self):
        df = pd.DataFrame({"qty": [0, 5]})
        r = expect_positive_values(df, "qty")
        assert not r.passed

    def test_fails_negative(self):
        df = pd.DataFrame({"qty": [-1, 5]})
        r = expect_positive_values(df, "qty")
        assert not r.passed


# ─────────────────────────────────────────────────────────────
#  Tests DataQualityReport
# ─────────────────────────────────────────────────────────────

class TestDataQualityReport:

    def build_report(self, results):
        report = DataQualityReport()
        for r in results:
            report.add(r)
        return report

    def test_passed_when_all_ok(self):
        r1 = ExpectationResult(passed=True, message="ok")
        r2 = ExpectationResult(passed=True, message="ok")
        report = self.build_report([r1, r2])
        assert report.passed

    def test_failed_when_any_error(self):
        r1 = ExpectationResult(passed=True, message="ok")
        r2 = ExpectationResult(passed=False, level="error", message="fail")
        report = self.build_report([r1, r2])
        assert not report.passed

    def test_errors_and_warnings(self):
        report = DataQualityReport()
        report.add(ExpectationResult(passed=False, level="error", message="E"))
        report.add(ExpectationResult(passed=False, level="warning", message="W"))
        assert len(report.errors) == 1
        assert len(report.warnings) == 1

    def test_to_dict_contains_keys(self):
        report = self.build_report([
            ExpectationResult(passed=True, message="ok")
        ])
        d = report.to_dict()
        assert "passed" in d
        assert "errors" in d
        assert "warnings" in d
        assert "total_checks" in d

    def test_total_checks_count(self):
        checks = [ExpectationResult(passed=True, message="ok")] * 5
        report = self.build_report(checks)
        assert report.to_dict()["total_checks"] == 5
