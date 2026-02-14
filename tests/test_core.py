"""
Tests for Spark core — the main interface.

API calls are mocked so no API key is needed to run these.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from dataspark.core import Spark
from dataspark import prompts


# ── Mock Setup ────────────────────────────────────────────────────


@pytest.fixture
def mock_anthropic():
    """Patch the Anthropic client so no real API calls are made."""
    with patch("dataspark.core.anthropic") as mock_mod:
        mock_client = MagicMock()

        # Mock a typical API response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a mock AI response.")]
        mock_client.messages.create.return_value = mock_response

        mock_mod.Anthropic.return_value = mock_client
        mock_mod.APIError = Exception  # for error handling paths

        yield mock_client


@pytest.fixture
def spark(mock_anthropic):
    """A Spark instance with mocked API."""
    return Spark(api_key="sk-ant-test-key")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "x": [1, 2, 3, 4, 5],
        "y": [10, 20, 30, 40, 50],
        "cat": ["a", "b", "a", "b", "a"],
    })


# ── Initialization ────────────────────────────────────────────────


class TestInit:

    def test_default_model(self, spark):
        assert "sonnet" in spark.model

    def test_custom_model_shortcut(self, mock_anthropic):
        s = Spark(api_key="test", model="opus")
        assert "opus" in s.model

    def test_custom_model_full(self, mock_anthropic):
        s = Spark(api_key="test", model="claude-haiku-4-5-20251001")
        assert s.model == "claude-haiku-4-5-20251001"

    def test_no_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="No API key"):
                Spark()

    def test_repr(self, spark):
        r = repr(spark)
        assert "Spark(" in r
        assert "messages=0" in r


# ── Data Loading ──────────────────────────────────────────────────


class TestDataLoading:

    def test_load_dataframe(self, spark, sample_df):
        df = spark.load(sample_df, name="test")
        assert len(df) == 5
        assert spark._data_context != ""

    def test_load_sets_context(self, spark, sample_df):
        spark.load(sample_df, name="test")
        assert "test" in spark._data_context
        assert "x" in spark._data_context


# ── API Calls ─────────────────────────────────────────────────────


class TestAPICalls:

    def test_explore_calls_api(self, spark, sample_df, mock_anthropic):
        result = spark.explore(sample_df, name="test")
        assert mock_anthropic.messages.create.called
        assert result == "This is a mock AI response."

    def test_project_calls_api(self, spark, mock_anthropic):
        result = spark.project("Build a churn model")
        assert mock_anthropic.messages.create.called
        assert result == "This is a mock AI response."

    def test_brainstorm_calls_api(self, spark, mock_anthropic):
        result = spark.brainstorm("I have sales data")
        assert mock_anthropic.messages.create.called
        assert result == "This is a mock AI response."

    def test_code_calls_api(self, spark, mock_anthropic):
        result = spark.code("Write a preprocessing pipeline")
        assert mock_anthropic.messages.create.called
        assert result == "This is a mock AI response."

    def test_ask_calls_api(self, spark, mock_anthropic):
        result = spark.ask("What model should I use?")
        assert mock_anthropic.messages.create.called

    def test_best_practices_calls_api(self, spark, mock_anthropic):
        result = spark.best_practices("Cross-validation")
        assert mock_anthropic.messages.create.called


# ── Conversation History ──────────────────────────────────────────


class TestConversationHistory:

    def test_ask_builds_history(self, spark, mock_anthropic):
        spark.ask("Question 1")
        spark.ask("Question 2")
        assert len(spark.conversation_history) == 4  # 2 user + 2 assistant

    def test_history_has_correct_roles(self, spark, mock_anthropic):
        spark.ask("Hello")
        assert spark.conversation_history[0]["role"] == "user"
        assert spark.conversation_history[1]["role"] == "assistant"

    def test_explore_sets_history(self, spark, sample_df, mock_anthropic):
        spark.explore(sample_df)
        assert len(spark.conversation_history) >= 2

    def test_reset_clears_everything(self, spark, sample_df, mock_anthropic):
        spark.load(sample_df, name="test")
        spark.ask("Hello")
        spark.reset()
        assert len(spark.conversation_history) == 0
        assert spark._data_context == ""


# ── System Prompts ────────────────────────────────────────────────


class TestSystemPrompts:

    def test_explore_uses_explore_prompt(self, spark, sample_df, mock_anthropic):
        spark.explore(sample_df)
        call_kwargs = mock_anthropic.messages.create.call_args
        assert "EXPLORATION" in call_kwargs.kwargs.get("system", "")

    def test_project_uses_project_prompt(self, spark, mock_anthropic):
        spark.project("test project")
        call_kwargs = mock_anthropic.messages.create.call_args
        assert "PROJECT" in call_kwargs.kwargs.get("system", "")

    def test_brainstorm_uses_brainstorm_prompt(self, spark, mock_anthropic):
        spark.brainstorm("test brainstorm")
        call_kwargs = mock_anthropic.messages.create.call_args
        assert "BRAINSTORM" in call_kwargs.kwargs.get("system", "")

    def test_data_context_injected(self, spark, sample_df, mock_anthropic):
        spark.load(sample_df, name="mydata")
        spark.ask("analyze this")
        call_kwargs = mock_anthropic.messages.create.call_args
        system = call_kwargs.kwargs.get("system", "")
        assert "mydata" in system


# ── Slash Commands ────────────────────────────────────────────────


class TestCommands:

    def test_clear_command(self, spark, mock_anthropic):
        spark.ask("Hello")
        result = spark._handle_command("/clear")
        assert result is True
        assert len(spark.conversation_history) == 0

    def test_model_command(self, spark):
        spark._handle_command("/model haiku")
        assert "haiku" in spark.model

    def test_help_command(self, spark):
        result = spark._handle_command("/help")
        assert result is True

    def test_unknown_command(self, spark):
        result = spark._handle_command("/foobar")
        assert result is True  # handled (prints error)

    def test_quit_returns_false(self, spark):
        result = spark._handle_command("/quit")
        assert result is False


# ── Error Handling ────────────────────────────────────────────────


class TestErrorHandling:

    def test_explore_without_data(self, spark):
        result = spark.explore()
        assert "Error" in result

    def test_api_error_handled(self, spark, mock_anthropic):
        mock_anthropic.messages.create.side_effect = Exception("API is down")
        result = spark.ask("test")
        assert "Error" in result
