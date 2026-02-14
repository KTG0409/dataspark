"""
Tests for DataProfile — dataset profiling logic.

These tests are fully offline (no API key needed).
"""

import json

import numpy as np
import pandas as pd
import pytest

from dataspark.profiles import DataProfile


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def simple_df():
    """A small, clean DataFrame for basic tests."""
    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
        "score": [88.5, 92.0, 76.3, 95.1, 81.7],
        "grade": ["A", "A", "B", "A", "B"],
    })


@pytest.fixture
def messy_df():
    """A DataFrame with nulls, outliers, duplicates, and mixed types."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "user_id": range(1, n + 1),
        "revenue": np.append(np.random.normal(100, 20, n - 2), [500, -50]),  # outliers
        "category": np.random.choice(["A", "B", "C", None], n),
        "signup_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "notes": [None if i % 5 == 0 else f"note_{i}" for i in range(n)],
    })


@pytest.fixture
def correlated_df():
    """A DataFrame with strongly correlated columns."""
    np.random.seed(42)
    x = np.random.normal(0, 1, 200)
    return pd.DataFrame({
        "x": x,
        "y": x * 2.5 + np.random.normal(0, 0.1, 200),  # r ≈ 0.99
        "z": np.random.normal(0, 1, 200),                # uncorrelated
    })


# ── Profile Build ─────────────────────────────────────────────────


class TestProfileBuild:

    def test_basic_shape(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert profile["shape"] == {"rows": 5, "columns": 4}

    def test_column_count(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert len(profile["columns"]) == 4

    def test_numeric_stats_present(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        score_col = next(c for c in profile["columns"] if c["name"] == "score")
        assert "stats" in score_col
        assert "mean" in score_col["stats"]
        assert "std" in score_col["stats"]
        assert "skewness" in score_col["stats"]

    def test_categorical_detection(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        grade_col = next(c for c in profile["columns"] if c["name"] == "grade")
        assert grade_col.get("is_likely_categorical") is True

    def test_top_values(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        grade_col = next(c for c in profile["columns"] if c["name"] == "grade")
        assert "top_values" in grade_col
        assert "A" in grade_col["top_values"]

    def test_sample_rows(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert len(profile["sample_rows"]) == 3

    def test_memory_usage(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert profile["memory_usage_mb"] >= 0

    def test_caching(self, simple_df):
        dp = DataProfile(simple_df, "test")
        p1 = dp.build()
        p2 = dp.build()
        assert p1 is p2  # same object, cached


# ── Missing Data ──────────────────────────────────────────────────


class TestMissingData:

    def test_no_missing(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert profile["missing_data"]["total_missing_cells"] == 0

    def test_missing_detected(self, messy_df):
        profile = DataProfile(messy_df, "messy").build()
        missing = profile["missing_data"]
        assert missing["total_missing_cells"] > 0
        col_names = [m["column"] for m in missing["columns_with_missing"]]
        assert "notes" in col_names

    def test_null_percentages(self, messy_df):
        profile = DataProfile(messy_df, "messy").build()
        notes_col = next(c for c in profile["columns"] if c["name"] == "notes")
        assert notes_col["null_pct"] > 0


# ── Duplicates ────────────────────────────────────────────────────


class TestDuplicates:

    def test_no_duplicates(self, simple_df):
        profile = DataProfile(simple_df, "test").build()
        assert profile["duplicates"]["total_duplicate_rows"] == 0

    def test_duplicates_detected(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        profile = DataProfile(df, "dupes").build()
        assert profile["duplicates"]["total_duplicate_rows"] == 1


# ── Correlations ──────────────────────────────────────────────────


class TestCorrelations:

    def test_high_correlation_detected(self, correlated_df):
        profile = DataProfile(correlated_df, "corr").build()
        assert "high_correlations" in profile
        assert len(profile["high_correlations"]) > 0
        top = profile["high_correlations"][0]
        assert abs(top["correlation"]) > 0.9

    def test_uncorrelated_excluded(self, correlated_df):
        profile = DataProfile(correlated_df, "corr").build()
        # x and z should NOT appear as highly correlated
        pairs = [(h["col_a"], h["col_b"]) for h in profile["high_correlations"]]
        assert not any(
            ("x" in p and "z" in p) for p in pairs
        )


# ── Outliers ──────────────────────────────────────────────────────


class TestOutliers:

    def test_outliers_detected(self, messy_df):
        profile = DataProfile(messy_df, "messy").build()
        rev_col = next(c for c in profile["columns"] if c["name"] == "revenue")
        assert rev_col["stats"]["n_outliers_iqr"] > 0


# ── Date Columns ──────────────────────────────────────────────────


class TestDates:

    def test_datetime_range(self, messy_df):
        profile = DataProfile(messy_df, "messy").build()
        date_col = next(c for c in profile["columns"] if c["name"] == "signup_date")
        assert "date_range" in date_col

    def test_string_date_detection(self):
        df = pd.DataFrame({"date_str": ["2024-01-01", "2024-02-01", "2024-03-01"]})
        profile = DataProfile(df, "dates").build()
        col = profile["columns"][0]
        assert col.get("might_be_date") is True


# ── Prompt Context ────────────────────────────────────────────────


class TestPromptContext:

    def test_returns_string(self, simple_df):
        ctx = DataProfile(simple_df, "test").to_prompt_context()
        assert isinstance(ctx, str)
        assert len(ctx) > 100

    def test_contains_key_sections(self, simple_df):
        ctx = DataProfile(simple_df, "test").to_prompt_context()
        assert "Dataset Profile" in ctx
        assert "Columns" in ctx
        assert "Sample Rows" in ctx

    def test_contains_column_names(self, simple_df):
        ctx = DataProfile(simple_df, "test").to_prompt_context()
        for col in simple_df.columns:
            assert col in ctx

    def test_messy_data_shows_issues(self, messy_df):
        ctx = DataProfile(messy_df, "messy").to_prompt_context()
        assert "Missing Data" in ctx
        assert "outlier" in ctx.lower() or "Outlier" in ctx or "n_outliers" in ctx.lower()
