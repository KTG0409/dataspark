"""
Tests for prompt templates.
"""

from dataspark import prompts


class TestPromptConstants:

    def test_system_base_exists(self):
        assert len(prompts.SYSTEM_BASE) > 100
        assert "data scien" in prompts.SYSTEM_BASE.lower()

    def test_all_modes_contain_base(self):
        modes = [
            prompts.SYSTEM_EXPLORE,
            prompts.SYSTEM_PROJECT,
            prompts.SYSTEM_BRAINSTORM,
            prompts.SYSTEM_CODE,
            prompts.SYSTEM_BEST_PRACTICES,
            prompts.SYSTEM_CHAT,
        ]
        for mode in modes:
            assert prompts.SYSTEM_BASE in mode

    def test_modes_have_unique_content(self):
        modes = [
            prompts.SYSTEM_EXPLORE,
            prompts.SYSTEM_PROJECT,
            prompts.SYSTEM_BRAINSTORM,
            prompts.SYSTEM_CODE,
            prompts.SYSTEM_BEST_PRACTICES,
            prompts.SYSTEM_CHAT,
        ]
        # Each mode should be unique
        assert len(set(modes)) == len(modes)


class TestPromptBuilders:

    def test_explore_prompt_with_focus(self):
        msgs = prompts.build_explore_prompt("profile data here", "look at revenue")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        assert "profile data here" in msgs[0]["content"]
        assert "revenue" in msgs[0]["content"]

    def test_explore_prompt_without_focus(self):
        msgs = prompts.build_explore_prompt("profile data here")
        assert "recommendations" in msgs[0]["content"].lower()

    def test_project_prompt_with_data(self):
        msgs = prompts.build_project_prompt("churn model", "data profile")
        assert "churn model" in msgs[0]["content"]
        assert "data profile" in msgs[0]["content"]

    def test_project_prompt_without_data(self):
        msgs = prompts.build_project_prompt("churn model")
        assert "churn model" in msgs[0]["content"]

    def test_brainstorm_prompt(self):
        msgs = prompts.build_brainstorm_prompt("sales data", "profile")
        assert "sales data" in msgs[0]["content"]
