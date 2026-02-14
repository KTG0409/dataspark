"""
DataProfile — Automatic dataset profiling to feed context to the LLM.
"""

import json
from pathlib import Path
from typing import Any, Optional

try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError("dataspark requires pandas and numpy: pip install pandas numpy")


class DataProfile:
    """Generate a rich, LLM-friendly profile of a DataFrame."""

    def __init__(self, df: pd.DataFrame, name: str = "dataset"):
        self.df = df
        self.name = name
        self._profile: Optional[dict] = None

    def build(self) -> dict:
        """Build the full profile dictionary."""
        if self._profile is not None:
            return self._profile

        df = self.df
        profile = {
            "name": self.name,
            "shape": {"rows": len(df), "columns": len(df.columns)},
            "columns": self._profile_columns(),
            "missing_data": self._missing_summary(),
            "duplicates": {
                "total_duplicate_rows": int(df.duplicated().sum()),
                "pct_duplicate": round(df.duplicated().mean() * 100, 2),
            },
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1e6, 2),
            "sample_rows": df.head(3).to_dict(orient="records"),
        }

        # Correlation highlights for numeric columns
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            high_corr = []
            for i in range(len(num_cols)):
                for j in range(i + 1, len(num_cols)):
                    val = corr.iloc[i, j]
                    if abs(val) > 0.7:
                        high_corr.append({
                            "col_a": num_cols[i],
                            "col_b": num_cols[j],
                            "correlation": round(val, 3),
                        })
            profile["high_correlations"] = sorted(
                high_corr, key=lambda x: abs(x["correlation"]), reverse=True
            )[:10]

        self._profile = profile
        return profile

    def _profile_columns(self) -> list[dict]:
        """Profile each column."""
        df = self.df
        cols = []
        for col in df.columns:
            s = df[col]
            info: dict[str, Any] = {
                "name": col,
                "dtype": str(s.dtype),
                "non_null_count": int(s.notna().sum()),
                "null_count": int(s.isna().sum()),
                "null_pct": round(s.isna().mean() * 100, 2),
                "n_unique": int(s.nunique()),
            }

            if pd.api.types.is_numeric_dtype(s):
                desc = s.describe()
                info["stats"] = {
                    "mean": round(float(desc["mean"]), 4),
                    "std": round(float(desc["std"]), 4),
                    "min": float(desc["min"]),
                    "25%": float(desc["25%"]),
                    "50%": float(desc["50%"]),
                    "75%": float(desc["75%"]),
                    "max": float(desc["max"]),
                    "skewness": round(float(s.skew()), 4),
                    "kurtosis": round(float(s.kurtosis()), 4),
                }
                # Detect potential outliers (IQR method)
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr = q3 - q1
                n_outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                info["stats"]["n_outliers_iqr"] = n_outliers

            elif pd.api.types.is_datetime64_any_dtype(s):
                info["date_range"] = {
                    "min": str(s.min()),
                    "max": str(s.max()),
                }

            elif pd.api.types.is_string_dtype(s) or pd.api.types.is_categorical_dtype(s):
                top = s.value_counts().head(5)
                info["top_values"] = {str(k): int(v) for k, v in top.items()}
                if s.nunique() <= 20:
                    info["is_likely_categorical"] = True
                # Check for potential date strings
                sample = s.dropna().head(20)
                try:
                    pd.to_datetime(sample, format="mixed")
                    info["might_be_date"] = True
                except (ValueError, TypeError):
                    pass

            cols.append(info)
        return cols

    def _missing_summary(self) -> dict:
        """Summarize missing data patterns."""
        df = self.df
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if missing.empty:
            return {"total_missing_cells": 0, "columns_with_missing": []}

        return {
            "total_missing_cells": int(missing.sum()),
            "pct_total_cells": round(missing.sum() / (len(df) * len(df.columns)) * 100, 2),
            "columns_with_missing": [
                {"column": col, "count": int(val), "pct": round(val / len(df) * 100, 2)}
                for col, val in missing.sort_values(ascending=False).items()
            ],
        }

    def to_prompt_context(self) -> str:
        """Convert profile to an LLM-friendly text summary."""
        p = self.build()

        lines = [
            f"## Dataset Profile: {p['name']}",
            f"Shape: {p['shape']['rows']:,} rows × {p['shape']['columns']} columns",
            f"Memory: {p['memory_usage_mb']} MB",
            f"Duplicate rows: {p['duplicates']['total_duplicate_rows']:,} ({p['duplicates']['pct_duplicate']}%)",
            "",
            "### Columns:",
        ]

        for c in p["columns"]:
            line = f"- **{c['name']}** ({c['dtype']}): {c['n_unique']} unique, {c['null_pct']}% null"
            if "stats" in c:
                s = c["stats"]
                line += f" | mean={s['mean']}, std={s['std']}, range=[{s['min']}, {s['max']}]"
                if s["n_outliers_iqr"] > 0:
                    line += f", {s['n_outliers_iqr']} outliers(IQR)"
            if c.get("is_likely_categorical"):
                line += " [CATEGORICAL]"
            if c.get("might_be_date"):
                line += " [POSSIBLE DATE]"
            if "top_values" in c:
                top3 = list(c["top_values"].items())[:3]
                line += f" | top: {', '.join(f'{k}({v})' for k, v in top3)}"
            lines.append(line)

        if p.get("high_correlations"):
            lines.append("\n### Notable Correlations:")
            for hc in p["high_correlations"]:
                lines.append(f"- {hc['col_a']} ↔ {hc['col_b']}: r={hc['correlation']}")

        if p["missing_data"]["total_missing_cells"] > 0:
            lines.append(f"\n### Missing Data: {p['missing_data']['pct_total_cells']}% of all cells")
            for m in p["missing_data"]["columns_with_missing"][:5]:
                lines.append(f"- {m['column']}: {m['count']:,} ({m['pct']}%)")

        lines.append("\n### Sample Rows (first 3):")
        lines.append("```json")
        lines.append(json.dumps(p["sample_rows"], indent=2, default=str)[:2000])
        lines.append("```")

        return "\n".join(lines)
