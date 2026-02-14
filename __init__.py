"""
DataSpark — Your local AI-powered data science co-pilot.

Usage:
    from dataspark import Spark

    spark = Spark()                          # interactive mode
    spark = Spark(api_key="sk-ant-...")       # explicit key
    spark = Spark(model="claude-sonnet-4-5-20250514")  # choose model

    # Explore a dataset
    spark.explore("data.csv")

    # Ask anything
    spark.ask("What clustering approach should I use for customer segmentation?")

    # Start a guided project
    spark.project("Predict customer churn for a SaaS company")

    # Brainstorm analysis ideas
    spark.brainstorm("I have 3 years of sales data with regional breakdowns")

    # Get code generated
    spark.code("Build a random forest pipeline with cross-validation")

    # Full interactive session
    spark.chat()
"""

__version__ = "0.1.0"

from dataspark.explorer import DataExplorer
from dataspark.profiles import DataProfile


def __getattr__(name):
    if name == "Spark":
        from dataspark.core import Spark
        return Spark
    raise AttributeError(f"module 'dataspark' has no attribute {name}")


__all__ = ["Spark", "DataExplorer", "DataProfile"]
