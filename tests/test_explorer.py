"""
Tests for DataExplorer — data loading and profiling.

Fully offline (no API key needed).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataspark.explorer import DataExplorer


@pytest.fixture
def explorer():
    return DataExplorer()


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "x", "y"],
        "c": [1.1, 2.2, 3.3, 4.4, 5.5],
    })


@pytest.fixture
def csv_file(sample_df):
    """Write a temp CSV and return the path."""
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        sample_df.to_csv(f, index=False)
        return f.name


@pytest.fixture
def tsv_file(sample_df):
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as f:
        sample_df.to_csv(f, sep="\t", index=False)
        return f.name


@pytest.fixture
def json_file(sample_df):
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        sample_df.to_json(f, orient="records")
        return f.name


@pytest.fixture
def parquet_file(sample_df):
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        sample_df.to_parquet(f.name)
        return f.name


# ── Loading ───────────────────────────────────────────────────────


class TestLoading:

    def test_load_dataframe(self, explorer, sample_df):
        df = explorer.load(sample_df, name="test")
        assert len(df) == 5
        assert "test" in explorer.dataframes
        assert "test" in explorer.profiles

    def test_load_csv(self, explorer, csv_file):
        df = explorer.load(csv_file)
        assert len(df) == 5
        assert len(explorer.dataframes) == 1

    def test_load_tsv(self, explorer, tsv_file):
        df = explorer.load(tsv_file)
        assert len(df) == 5

    def test_load_json(self, explorer, json_file):
        df = explorer.load(json_file)
        assert len(df) == 5

    def test_load_parquet(self, explorer, parquet_file):
        df = explorer.load(parquet_file)
        assert len(df) == 5

    def test_load_with_custom_name(self, explorer, sample_df):
        explorer.load(sample_df, name="my_data")
        assert "my_data" in explorer.dataframes

    def test_unsupported_format(self, explorer):
        with pytest.raises(ValueError, match="Unsupported file type"):
            explorer.load("data.xyz")

    def test_multiple_datasets(self, explorer, sample_df):
        explorer.load(sample_df, name="first")
        explorer.load(sample_df, name="second")
        assert len(explorer.dataframes) == 2


# ── Profiling ─────────────────────────────────────────────────────


class TestProfiling:

    def test_profile_auto_select(self, explorer, sample_df):
        explorer.load(sample_df, name="only")
        profile = explorer.profile()
        assert profile.name == "only"

    def test_profile_by_name(self, explorer, sample_df):
        explorer.load(sample_df, name="first")
        explorer.load(sample_df, name="second")
        profile = explorer.profile("first")
        assert profile.name == "first"

    def test_profile_ambiguous_raises(self, explorer, sample_df):
        explorer.load(sample_df, name="a")
        explorer.load(sample_df, name="b")
        with pytest.raises(ValueError, match="Multiple datasets"):
            explorer.profile()

    def test_profile_missing_raises(self, explorer):
        with pytest.raises(KeyError, match="No dataset named"):
            explorer.profile("nonexistent")


# ── LLM Context ──────────────────────────────────────────────────


class TestLLMContext:

    def test_context_returns_string(self, explorer, sample_df):
        explorer.load(sample_df, name="test")
        ctx = explorer.context_for_llm("test")
        assert isinstance(ctx, str)
        assert "test" in ctx

    def test_context_has_column_info(self, explorer, sample_df):
        explorer.load(sample_df, name="test")
        ctx = explorer.context_for_llm("test")
        for col in sample_df.columns:
            assert col in ctx


# ── Quick Look ────────────────────────────────────────────────────


class TestQuickLook:

    def test_quick_look_runs(self, explorer, sample_df, capsys):
        explorer.load(sample_df, name="test")
        explorer.quick_look("test")
        captured = capsys.readouterr()
        assert "test" in captured.out
        assert "5 rows" in captured.out
