# -*- coding: utf-8 -*-
"""
Preprocessing Module

Feature normalization, scaling, and pipeline utilities for ML workflows.
All functions work on pandas DataFrames/Series and return the same type.

Usage:
    from feature_engine import preprocessing

    scaled = preprocessing.min_max_scale(df)
    standardized = preprocessing.standard_scale(df)
    winsorized = preprocessing.winsorize(series, lower=0.01, upper=0.99)
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, Any


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def min_max_scale(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    feature_range: Tuple[float, float] = (0.0, 1.0),
) -> pd.DataFrame:
    """Min-max normalization to [feature_range].

    Args:
        df: Input DataFrame.
        columns: Columns to scale. None = all numeric columns.
        feature_range: Output range (default 0-1).

    Returns:
        Scaled DataFrame (original is not modified).
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()
    lo, hi = feature_range

    for col in cols:
        col_min = result[col].min()
        col_max = result[col].max()
        rng = col_max - col_min
        if rng == 0:
            result[col] = lo
        else:
            result[col] = lo + (result[col] - col_min) / rng * (hi - lo)

    return result


def standard_scale(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Z-score standardization: (x - mean) / std.

    Args:
        df: Input DataFrame.
        columns: Columns to scale. None = all numeric columns.

    Returns:
        Standardized DataFrame.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()

    for col in cols:
        mean = result[col].mean()
        std = result[col].std()
        if std == 0 or np.isnan(std):
            result[col] = 0.0
        else:
            result[col] = (result[col] - mean) / std

    return result


def robust_scale(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Robust scaling using median and IQR: (x - median) / IQR.

    More resilient to outliers than standard scaling.

    Args:
        df: Input DataFrame.
        columns: Columns to scale. None = all numeric columns.

    Returns:
        Scaled DataFrame.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()

    for col in cols:
        median = result[col].median()
        q75 = result[col].quantile(0.75)
        q25 = result[col].quantile(0.25)
        iqr = q75 - q25
        if iqr == 0:
            result[col] = 0.0
        else:
            result[col] = (result[col] - median) / iqr

    return result


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def log_normalize(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Log normalization: log(1 + x) for positive values.

    Args:
        df: Input DataFrame.
        columns: Columns to normalize. None = all numeric columns.

    Returns:
        Log-normalized DataFrame.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()

    for col in cols:
        min_val = result[col].min()
        shift = abs(min_val) + 1 if min_val < 0 else 0
        result[col] = np.log1p(result[col] + shift)

    return result


def rank_normalize(
    df: pd.DataFrame,
    columns: Optional[list] = None,
) -> pd.DataFrame:
    """Rank-based normalization: convert to percentile ranks (0-1).

    Args:
        df: Input DataFrame.
        columns: Columns to normalize. None = all numeric columns.

    Returns:
        Rank-normalized DataFrame.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()

    for col in cols:
        result[col] = result[col].rank(pct=True)

    return result


# ---------------------------------------------------------------------------
# Outlier Handling
# ---------------------------------------------------------------------------

def winsorize(
    series: pd.Series,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorize a series: clip values at quantile boundaries.

    Args:
        series: Input series.
        lower: Lower quantile (default 1st percentile).
        upper: Upper quantile (default 99th percentile).

    Returns:
        Winsorized series.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def winsorize_df(
    df: pd.DataFrame,
    columns: Optional[list] = None,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.DataFrame:
    """Winsorize multiple columns of a DataFrame.

    Args:
        df: Input DataFrame.
        columns: Columns to winsorize. None = all numeric columns.
        lower: Lower quantile.
        upper: Upper quantile.

    Returns:
        Winsorized DataFrame.
    """
    cols = columns or df.select_dtypes(include=[np.number]).columns.tolist()
    result = df.copy()
    for col in cols:
        result[col] = winsorize(result[col], lower, upper)
    return result


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------

def fill_missing(
    df: pd.DataFrame,
    strategy: str = "forward_fill",
    columns: Optional[list] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Fill missing values.

    Strategies:
        - forward_fill: propagate last valid value
        - backward_fill: use next valid value
        - mean: fill with column mean
        - median: fill with column median
        - zero: fill with 0

    Args:
        df: Input DataFrame.
        strategy: Filling strategy.
        columns: Columns to fill. None = all columns.
        limit: Maximum consecutive fills.

    Returns:
        DataFrame with filled values.
    """
    cols = columns or df.columns.tolist()
    result = df.copy()

    fill_methods = {
        "forward_fill": lambda c: result[c].ffill(limit=limit),
        "backward_fill": lambda c: result[c].bfill(limit=limit),
        "mean": lambda c: result[c].fillna(result[c].mean()),
        "median": lambda c: result[c].fillna(result[c].median()),
        "zero": lambda c: result[c].fillna(0),
    }

    if strategy not in fill_methods:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose from: {list(fill_methods.keys())}")

    for col in cols:
        result[col] = fill_methods[strategy](col)

    return result


# ---------------------------------------------------------------------------
# Feature Pipeline
# ---------------------------------------------------------------------------

class FeaturePipeline:
    """Sequential feature transformation pipeline.

    Usage:
        pipeline = FeaturePipeline()
        pipeline.add_step("winsorize", lower=0.01, upper=0.99)
        pipeline.add_step("standard_scale")
        transformed = pipeline.fit_transform(df)
    """

    STEPS = {
        "min_max_scale": min_max_scale,
        "standard_scale": standard_scale,
        "robust_scale": robust_scale,
        "log_normalize": log_normalize,
        "rank_normalize": rank_normalize,
        "winsorize": winsorize_df,
        "fill_missing": fill_missing,
    }

    def __init__(self):
        self._steps: list = []
        self._fitted_params: Dict[str, Any] = {}

    def add_step(self, name: str, **kwargs) -> "FeaturePipeline":
        """Add a transformation step."""
        if name not in self.STEPS:
            raise ValueError(f"Unknown step '{name}'. Available: {list(self.STEPS.keys())}")
        self._steps.append((name, kwargs))
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all steps sequentially."""
        result = df.copy()
        for name, kwargs in self._steps:
            fn = self.STEPS[name]
            result = fn(result, **kwargs)
        return result

    def describe(self) -> list:
        """Return list of configured steps."""
        return [{"step": name, "params": params} for name, params in self._steps]
