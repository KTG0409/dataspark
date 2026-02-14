"""
Command-line interface for DataSpark.

Usage:
    dataspark                      # Start interactive chat
    dataspark explore data.csv     # Explore a dataset
    dataspark project "Build a churn model"  # Design a project
    dataspark ask "What test should I use?"   # Quick question
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="DataSpark — AI-powered data science co-pilot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dataspark                                    Interactive chat
  dataspark explore sales.csv                  Analyze a dataset
  dataspark explore sales.csv -f "focus on seasonality"
  dataspark project "Predict customer churn"   Design a project
  dataspark brainstorm "3 years of sales data" Brainstorm ideas
  dataspark ask "When should I use XGBoost vs Random Forest?"
  dataspark code "Build a preprocessing pipeline"
        """,
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="chat",
        choices=["chat", "explore", "project", "brainstorm", "ask", "code", "practices"],
        help="Operation mode (default: chat)",
    )

    parser.add_argument("input", nargs="?", default="", help="Input: file path, question, or description")
    parser.add_argument("-f", "--focus", default="", help="Focus area for exploration")
    parser.add_argument("-m", "--model", default=None, help="Model: opus, sonnet, haiku, or full name")
    parser.add_argument("-k", "--key", default=None, help="API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show debug info")

    args = parser.parse_args()

    from dataspark import Spark

    try:
        spark = Spark(api_key=args.key, model=args.model, verbose=args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    mode = args.mode

    if mode == "chat":
        spark.chat()

    elif mode == "explore":
        if not args.input:
            print("Error: /explore requires a file path. Usage: dataspark explore <file>", file=sys.stderr)
            sys.exit(1)
        spark.explore(args.input, focus=args.focus)

    elif mode == "project":
        if not args.input:
            print("Error: project requires a description.", file=sys.stderr)
            sys.exit(1)
        spark.project(args.input)

    elif mode == "brainstorm":
        if not args.input:
            print("Error: brainstorm requires context.", file=sys.stderr)
            sys.exit(1)
        spark.brainstorm(args.input)

    elif mode == "ask":
        if not args.input:
            print("Error: ask requires a question.", file=sys.stderr)
            sys.exit(1)
        spark.ask(args.input)

    elif mode == "code":
        if not args.input:
            print("Error: code requires a request.", file=sys.stderr)
            sys.exit(1)
        spark.code(args.input)

    elif mode == "practices":
        if not args.input:
            print("Error: practices requires a topic.", file=sys.stderr)
            sys.exit(1)
        spark.best_practices(args.input)


if __name__ == "__main__":
    main()
