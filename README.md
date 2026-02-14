# DataSpark — AI-Powered Data Science Co-Pilot

**A Python library that brings Claude's data science expertise into your local workflow — Jupyter notebooks, scripts, terminal, anywhere Python runs.**

No browser needed. No logging in. Just `import` and go.

---

## Quick Start

### 1. Install

```bash
pip install dataspark-ai              # from PyPI
# or
pip install dataspark-ai[full]        # includes sklearn, matplotlib, seaborn, plotly, scipy
# or from source
git clone https://github.com/yourusername/dataspark.git
cd dataspark && pip install -e ".[dev]"
```

### 2. Set your API key

```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

Get a key at [console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)

### 3. Use It

```python
from dataspark import Spark

spark = Spark()

# Explore a dataset — get instant analysis, quality checks, and recommendations
spark.explore("sales_data.csv")

# Ask any data science question
spark.ask("Should I use one-hot encoding or target encoding for high-cardinality categoricals?")

# Design a complete project
spark.project("Build a customer churn prediction model for our SaaS platform")

# Brainstorm creative analysis ideas
spark.brainstorm("I have 3 years of e-commerce transaction data with 2M rows")

# Generate production code
spark.code("Build a feature engineering pipeline for time series with lag features and rolling stats")

# Get best practices guidance
spark.best_practices("Cross-validation strategies for time series data")

# Interactive session (like chatting with Claude)
spark.chat()
```

---

## Core Features

### `spark.explore(source)` — Dataset Analysis

Pass a CSV, Excel file, DataFrame, or URL. DataSpark will:
- Profile every column (types, distributions, outliers, correlations)
- Flag data quality issues
- Recommend specific analyses based on what it sees
- Ask you clarifying questions about your goals
- Provide ready-to-run code snippets

```python
spark.explore("customers.csv")
spark.explore("https://data.example.com/dataset.csv")
spark.explore(my_dataframe, name="revenue")

# Focus on something specific
spark.explore("data.csv", focus="I need to predict the 'churned' column")
```

### `spark.project(description)` — Project Design

Describe what you want to build. DataSpark designs the full pipeline:

```python
spark.project("Forecast demand for 500 SKUs across 12 warehouses, daily granularity")
spark.project("Build a recommendation engine for our content platform")
spark.project("Anomaly detection for network traffic logs, ~10M events/day")
```

### `spark.brainstorm(context)` — Idea Generation

Get creative, ranked ideas from quick wins to big bets:

```python
spark.brainstorm("We have clickstream data, purchase history, and customer support tickets")
spark.brainstorm("Our marketing team wants to understand campaign attribution")
```

### `spark.code(request)` — Code Generation

Get complete, production-quality Python code:

```python
spark.code("XGBoost pipeline with Optuna hyperparameter tuning")
spark.code("Automated EDA function that generates a PDF report")
spark.code("FastAPI endpoint that serves predictions from a pickled model")
```

### `spark.ask(question)` — Ask Anything

Maintains conversation history so you can have a back-and-forth:

```python
spark.ask("What's the best way to handle class imbalance?")
spark.ask("Show me how to implement SMOTE with that approach")
spark.ask("Now how do I evaluate it properly?")
```

### `spark.chat()` — Interactive Terminal Session

Full interactive mode with slash commands:

```
/explore data.csv    — Load and analyze a dataset
/project <desc>      — Design a project
/brainstorm <ctx>    — Generate ideas
/code <request>      — Generate code
/model sonnet        — Switch models
/save conversation.md — Save chat history
/clear               — Reset context
/help                — Show commands
/quit                — Exit
```

---

## Configuration

```python
# Model selection (default: Claude Sonnet)
spark = Spark(model="opus")      # Most capable
spark = Spark(model="sonnet")    # Balanced (default)
spark = Spark(model="haiku")     # Fastest / cheapest

# Longer responses
spark = Spark(max_tokens=8192)

# Debug mode
spark = Spark(verbose=True)
```

---

## Command-Line Usage

```bash
# Interactive chat
dataspark

# Explore a dataset
dataspark explore data.csv
dataspark explore data.csv -f "focus on the target variable"

# Quick question
dataspark ask "When should I use Ridge vs Lasso?"

# Project design
dataspark project "Build a fraud detection system"

# Use a specific model
dataspark -m opus explore big_dataset.parquet
```

---

## Jupyter Notebook Tips

```python
from dataspark import Spark
spark = Spark()

# Load data through spark — it profiles automatically
df = spark.load("data.csv")

# Now all questions are context-aware
spark.ask("What feature engineering should I do?")
spark.ask("Write the code for that")

# You can also explore at any point
spark.explore(focus="relationships between price and demand")
```

---

## Architecture

```
dataspark/
├── __init__.py      # Clean exports
├── core.py          # Spark class — main interface & API calls
├── explorer.py      # DataExplorer — load & profile datasets
├── profiles.py      # DataProfile — statistical profiling
├── prompts.py       # System prompts for each mode
└── cli.py           # Command-line interface
```

The library works by:
1. **Profiling your data** locally (pandas — nothing leaves your machine except the summary)
2. **Building rich context** from the profile (statistics, distributions, quality issues)
3. **Sending that context + your question** to Claude via the API
4. **Maintaining conversation history** so follow-ups are contextual

Your raw data never leaves your machine. Only statistical summaries and column metadata are sent to the API.

---

## Extending DataSpark

### Custom System Prompts

```python
from dataspark import Spark

spark = Spark()
spark._current_system = """You are a financial data science expert.
Focus on: regulatory compliance, risk modeling, backtesting.
Always consider: data leakage, survivorship bias, look-ahead bias."""

spark.ask("How should I backtest this trading strategy?")
```

### Adding Data Context Manually

```python
spark._data_context = """
We have a PostgreSQL database with:
- transactions (50M rows, 3 years)
- customers (2M rows)
- products (10K SKUs)
Business: B2B SaaS, $50M ARR, 15% annual churn
"""
spark.ask("What analyses would have the most business impact?")
```

---

## Cost Awareness

API costs per ~1000 tokens (approximate):
| Model  | Input | Output |
|--------|-------|--------|
| Haiku  | $0.001 | $0.005 |
| Sonnet | $0.003 | $0.015 |
| Opus   | $0.015 | $0.075 |

A typical `explore()` call uses ~2-4K tokens. An interactive session might use 10-50K tokens total.
Use `spark = Spark(model="haiku")` for cost-sensitive workloads.

---

## Privacy & Security

- **Your raw data stays local.** Only statistical summaries (means, distributions, column names, 3 sample rows) are sent to the API.
- **API key is yours.** Use a personal key — it's billed to your Anthropic account, not tied to any employer.
- **No logging by default.** Conversations are in-memory only unless you `/save` them.
- **Review what's sent.** Call `spark.explorer.context_for_llm()` to see exactly what goes to the API.

---

## License

MIT — use it however you want.
