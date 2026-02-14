"""
Spark — The main interface for DataSpark.

This is the entry point for all interactions. It manages the API connection,
conversation history, dataset context, and dispatches to the right mode.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Union

try:
    import anthropic
except ImportError:
    raise ImportError(
        "dataspark requires the anthropic SDK: pip install anthropic"
    )

try:
    import pandas as pd
except ImportError:
    raise ImportError("dataspark requires pandas: pip install pandas")

from dataspark.explorer import DataExplorer
from dataspark.profiles import DataProfile
from dataspark import prompts


class Spark:
    """
    Your AI-powered data science co-pilot.

    Usage:
        spark = Spark()                          # uses ANTHROPIC_API_KEY env var
        spark = Spark(api_key="sk-ant-...")       # explicit key
        spark.explore("data.csv")                # analyze a dataset
        spark.ask("How should I handle outliers?")  # ask anything
        spark.chat()                             # interactive session
    """

    DEFAULT_MODEL = "claude-sonnet-4-5-20250514"
    AVAILABLE_MODELS = {
        "opus": "claude-opus-4-5-20250514",
        "sonnet": "claude-sonnet-4-5-20250514",
        "haiku": "claude-haiku-4-5-20251001",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        verbose: bool = False,
    ):
        """
        Initialize DataSpark.

        Args:
            api_key: Anthropic API key. Falls back to ANTHROPIC_API_KEY env var.
            model: Model to use. Shortcuts: "opus", "sonnet", "haiku".
                   Or pass a full model string.
            max_tokens: Max response tokens (default 4096).
            verbose: Print debug info.
        """
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "No API key found. Either pass api_key= or set ANTHROPIC_API_KEY env var.\n"
                "Get a key at: https://console.anthropic.com/settings/keys"
            )

        self.client = anthropic.Anthropic(api_key=key)
        self.model = self.AVAILABLE_MODELS.get(model, model) or self.DEFAULT_MODEL
        self.max_tokens = max_tokens
        self.verbose = verbose

        # State
        self.explorer = DataExplorer()
        self.conversation_history: list[dict] = []
        self._current_system: str = prompts.SYSTEM_CHAT
        self._data_context: str = ""

        if self.verbose:
            print(f"[DataSpark] Initialized with model: {self.model}")

    # ─── Core API Call ─────────────────────────────────────────────

    def _call(
        self,
        messages: list[dict],
        system: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Make a call to the Claude API."""
        sys_prompt = system or self._current_system

        # Inject data context into system prompt if available
        if self._data_context:
            sys_prompt += f"\n\n## Currently Loaded Dataset Context:\n{self._data_context}"

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens or self.max_tokens,
                system=sys_prompt,
                messages=messages,
            )
            return response.content[0].text

        except anthropic.APIError as e:
            return f"[API Error] {e}"
        except Exception as e:
            return f"[Error] {e}"

    def _call_with_history(self, user_message: str, system: Optional[str] = None) -> str:
        """Call API with conversation history."""
        self.conversation_history.append({"role": "user", "content": user_message})

        response_text = self._call(
            messages=self.conversation_history,
            system=system,
        )

        self.conversation_history.append({"role": "assistant", "content": response_text})
        return response_text

    # ─── Data Loading ──────────────────────────────────────────────

    def load(
        self,
        source: Union[str, Path, pd.DataFrame],
        name: Optional[str] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Load a dataset and generate its profile.

        Args:
            source: File path, URL, or pandas DataFrame.
            name: Optional name for the dataset.
            **kwargs: Additional args passed to the pandas reader.

        Returns:
            The loaded DataFrame.
        """
        df = self.explorer.load(source, name=name, **kwargs)
        name = name or list(self.explorer.profiles.keys())[-1]
        self._data_context = self.explorer.context_for_llm(name)

        if self.verbose:
            self.explorer.quick_look(name)

        return df

    # ─── Main Modes ────────────────────────────────────────────────

    def explore(
        self,
        source: Optional[Union[str, Path, pd.DataFrame]] = None,
        focus: str = "",
        name: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Explore a dataset with AI-powered analysis.

        Args:
            source: File path, URL, or DataFrame. If None, uses last loaded dataset.
            focus: Optional specific question or area to focus on.
            name: Optional dataset name.

        Returns:
            AI analysis and recommendations as a string.
        """
        if source is not None:
            self.load(source, name=name, **kwargs)

        if not self._data_context:
            return "[Error] No dataset loaded. Pass a file path, URL, or DataFrame."

        self._current_system = prompts.SYSTEM_EXPLORE
        messages = prompts.build_explore_prompt(self._data_context, focus)

        response = self._call(messages, system=prompts.SYSTEM_EXPLORE)

        # Add to history for follow-up
        self.conversation_history = messages + [
            {"role": "assistant", "content": response}
        ]

        self._print_response(response)
        return response

    def project(self, description: str) -> str:
        """
        Design a data science project.

        Args:
            description: What you want to build/solve.

        Returns:
            Project plan and recommendations.
        """
        self._current_system = prompts.SYSTEM_PROJECT
        messages = prompts.build_project_prompt(description, self._data_context)

        response = self._call(messages, system=prompts.SYSTEM_PROJECT)

        self.conversation_history = messages + [
            {"role": "assistant", "content": response}
        ]

        self._print_response(response)
        return response

    def brainstorm(self, context: str) -> str:
        """
        Brainstorm analysis ideas and approaches.

        Args:
            context: Describe your data, domain, or problem.

        Returns:
            Creative ideas ranked by feasibility and impact.
        """
        self._current_system = prompts.SYSTEM_BRAINSTORM
        messages = prompts.build_brainstorm_prompt(context, self._data_context)

        response = self._call(messages, system=prompts.SYSTEM_BRAINSTORM)

        self.conversation_history = messages + [
            {"role": "assistant", "content": response}
        ]

        self._print_response(response)
        return response

    def code(self, request: str) -> str:
        """
        Generate production-quality code.

        Args:
            request: What code do you need?

        Returns:
            Complete, runnable Python code with explanations.
        """
        self._current_system = prompts.SYSTEM_CODE
        return self._ask_internal(request, system=prompts.SYSTEM_CODE)

    def best_practices(self, topic: str) -> str:
        """
        Get expert guidance on data science best practices.

        Args:
            topic: What do you need guidance on?

        Returns:
            Best practices, pitfalls, and recommendations.
        """
        self._current_system = prompts.SYSTEM_BEST_PRACTICES
        return self._ask_internal(topic, system=prompts.SYSTEM_BEST_PRACTICES)

    def ask(self, question: str) -> str:
        """
        Ask any data science question (uses conversation history).

        Args:
            question: Your question.

        Returns:
            AI response.
        """
        return self._ask_internal(question)

    def _ask_internal(self, message: str, system: Optional[str] = None) -> str:
        """Internal ask that preserves history."""
        response = self._call_with_history(message, system=system)
        self._print_response(response)
        return response

    # ─── Interactive Chat ──────────────────────────────────────────

    def chat(self, greeting: bool = True) -> None:
        """
        Start an interactive chat session.

        Commands:
            /explore <file>  — Load and explore a dataset
            /project <desc>  — Design a project
            /brainstorm <ctx> — Brainstorm ideas
            /code <request>  — Generate code
            /clear           — Clear conversation history
            /model <name>    — Switch model (opus/sonnet/haiku)
            /save <file>     — Save conversation to file
            /quit or /exit   — End session
        """
        if greeting:
            self._print_greeting()

        while True:
            try:
                user_input = input("\n\033[36mYou:\033[0m ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye!")
                break

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    continue
                if user_input.lower() in ("/quit", "/exit", "/q"):
                    print("\nGoodbye! Happy analyzing.")
                    break

            # Regular message
            response = self._call_with_history(user_input)
            self._print_response(response)

    def _handle_command(self, cmd: str) -> bool:
        """Handle slash commands. Returns True if handled."""
        parts = cmd.split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        if command == "/explore":
            if arg:
                self.explore(arg)
            else:
                print("Usage: /explore <file_path>")
            return True

        elif command == "/project":
            if arg:
                self.project(arg)
            else:
                print("Usage: /project <description>")
            return True

        elif command == "/brainstorm":
            if arg:
                self.brainstorm(arg)
            else:
                print("Usage: /brainstorm <context>")
            return True

        elif command == "/code":
            if arg:
                self.code(arg)
            else:
                print("Usage: /code <what you need>")
            return True

        elif command == "/clear":
            self.conversation_history = []
            self._data_context = ""
            print("Conversation cleared.")
            return True

        elif command == "/model":
            if arg in self.AVAILABLE_MODELS:
                self.model = self.AVAILABLE_MODELS[arg]
                print(f"Switched to: {self.model}")
            elif arg:
                self.model = arg
                print(f"Set model to: {self.model}")
            else:
                print(f"Current: {self.model}")
                print(f"Shortcuts: {', '.join(self.AVAILABLE_MODELS.keys())}")
            return True

        elif command == "/save":
            self._save_conversation(arg or "dataspark_conversation.md")
            return True

        elif command in ("/quit", "/exit", "/q"):
            return False  # let the main loop handle exit

        elif command == "/help":
            print(self.chat.__doc__)
            return True

        else:
            print(f"Unknown command: {command}. Type /help for available commands.")
            return True

    # ─── Utilities ─────────────────────────────────────────────────

    def _print_greeting(self):
        """Print the welcome message."""
        print("\033[1m")
        print("╔══════════════════════════════════════════════════╗")
        print("║            ✦ DataSpark Co-Pilot ✦               ║")
        print("║     AI-Powered Data Science Assistant            ║")
        print("╠══════════════════════════════════════════════════╣")
        print("║  Commands:                                      ║")
        print("║    /explore <file>  — Analyze a dataset          ║")
        print("║    /project <desc>  — Design a project           ║")
        print("║    /brainstorm      — Generate ideas             ║")
        print("║    /code <request>  — Generate code              ║")
        print("║    /model <name>    — Switch model               ║")
        print("║    /save <file>     — Save conversation          ║")
        print("║    /help            — Show all commands           ║")
        print("║    /quit            — Exit                       ║")
        print("╚══════════════════════════════════════════════════╝")
        print(f"\033[0m  Model: {self.model}")
        if self._data_context:
            print(f"  Dataset loaded: {list(self.explorer.dataframes.keys())}")
        print()

    def _print_response(self, text: str) -> None:
        """Pretty-print the AI response."""
        print(f"\n\033[33m{'─' * 60}\033[0m")
        print(text)
        print(f"\033[33m{'─' * 60}\033[0m")

    def _save_conversation(self, filepath: str) -> None:
        """Save the conversation history to a markdown file."""
        with open(filepath, "w") as f:
            f.write("# DataSpark Conversation\n\n")
            for msg in self.conversation_history:
                role = "**You**" if msg["role"] == "user" else "**DataSpark**"
                f.write(f"## {role}\n\n{msg['content']}\n\n---\n\n")
        print(f"Conversation saved to: {filepath}")

    def reset(self) -> None:
        """Reset all state: history, data, system prompt."""
        self.conversation_history = []
        self._data_context = ""
        self._current_system = prompts.SYSTEM_CHAT
        self.explorer = DataExplorer()
        print("DataSpark reset.")

    @property
    def history(self) -> list[dict]:
        """Access conversation history."""
        return self.conversation_history

    def __repr__(self) -> str:
        n_msgs = len(self.conversation_history)
        n_datasets = len(self.explorer.dataframes)
        return (
            f"Spark(model='{self.model}', "
            f"messages={n_msgs}, "
            f"datasets={n_datasets})"
        )
