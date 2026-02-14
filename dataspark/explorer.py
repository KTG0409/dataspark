"""
DataExplorer — Load, profile, and generate analysis context for datasets.
"""

from pathlib import Path
from typing import Optional, Union

try:
    import pandas as pd
except ImportError:
    raise ImportError("dataspark requires pandas: pip install pandas")

from dataspark.profiles import DataProfile


class DataExplorer:
    """Load and explore datasets, generating rich context for AI analysis."""

    def __init__(self):
        self.dataframes: dict[str, pd.DataFrame] = {}
        self.profiles: dict[str, DataProfile] = {}

    def load(
        self,
        source: Union[str, Path, pd.DataFrame],
        name: Optional[str] = None,
        **read_kwargs,
    ) -> pd.DataFrame:
        """
        Load a dataset from file path, URL, or existing DataFrame.

        Supports: .csv, .tsv, .xlsx, .xls, .json, .parquet, .feather, .pkl
        """
        if isinstance(source, pd.DataFrame):
            df = source
            name = name or "dataframe"
        else:
            source = Path(source) if not str(source).startswith("http") else source
            name = name or (Path(source).stem if isinstance(source, Path) else "dataset")

            ext = str(source).rsplit(".", 1)[-1].lower()
            loaders = {
                "csv": pd.read_csv,
                "tsv": lambda p, **kw: pd.read_csv(p, sep="\t", **kw),
                "xlsx": pd.read_excel,
                "xls": pd.read_excel,
                "json": pd.read_json,
                "parquet": pd.read_parquet,
                "feather": pd.read_feather,
                "pkl": pd.read_pickle,
                "pickle": pd.read_pickle,
            }

            loader = loaders.get(ext)
            if loader is None:
                raise ValueError(
                    f"Unsupported file type: .{ext}. "
                    f"Supported: {', '.join(loaders.keys())}"
                )
            df = loader(source, **read_kwargs)

        self.dataframes[name] = df
        self.profiles[name] = DataProfile(df, name)
        return df

    def profile(self, name: Optional[str] = None) -> DataProfile:
        """Get the profile for a loaded dataset."""
        if name is None:
            if len(self.profiles) == 1:
                name = list(self.profiles.keys())[0]
            else:
                raise ValueError(
                    f"Multiple datasets loaded. Specify one: {list(self.profiles.keys())}"
                )
        if name not in self.profiles:
            raise KeyError(f"No dataset named '{name}'. Loaded: {list(self.profiles.keys())}")
        return self.profiles[name]

    def context_for_llm(self, name: Optional[str] = None) -> str:
        """Generate the full LLM-ready context string for a dataset."""
        return self.profile(name).to_prompt_context()

    def quick_look(self, name: Optional[str] = None) -> None:
        """Print a quick summary of a loaded dataset."""
        p = self.profile(name)
        profile = p.build()
        s = profile["shape"]
        print(f"\n{'='*60}")
        print(f"  {p.name}: {s['rows']:,} rows × {s['columns']} columns")
        print(f"  Memory: {profile['memory_usage_mb']} MB")
        print(f"  Duplicates: {profile['duplicates']['total_duplicate_rows']:,}")
        print(f"{'='*60}")
        print("\nColumns:")
        for c in profile["columns"]:
            tag = c["dtype"]
            if c.get("is_likely_categorical"):
                tag += ", categorical"
            if c.get("might_be_date"):
                tag += ", possible date"
            null_info = f", {c['null_pct']}% null" if c["null_pct"] > 0 else ""
            print(f"  • {c['name']} ({tag}{null_info})")

        if profile.get("high_correlations"):
            print("\nStrong Correlations:")
            for hc in profile["high_correlations"][:5]:
                print(f"  • {hc['col_a']} ↔ {hc['col_b']}: r={hc['correlation']}")
        print()
