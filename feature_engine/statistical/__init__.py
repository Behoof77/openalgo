# -*- coding: utf-8 -*-
"""
Statistical Features Module

Pure NumPy/Pandas implementations of statistical features for financial data.
No external dependencies beyond numpy and pandas.

Usage:
    from feature_engine import statistical

    returns = statistical.simple_returns(close)
    log_ret = statistical.log_returns(close)
    vol = statistical.realized_volatility(close, window=20)
    z = statistical.zscore(close, window=20)
"""

import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Return Calculations
# ---------------------------------------------------------------------------

def simple_returns(close: pd.Series) -> pd.Series:
    """Simple percentage returns: (P_t / P_{t-1}) - 1."""
    return close.pct_change()


def log_returns(close: pd.Series) -> pd.Series:
    """Log returns: ln(P_t / P_{t-1})."""
    return np.log(close / close.shift(1))


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Cumulative returns from a return series: (1 + r).cumprod() - 1."""
    return (1 + returns).cumprod() - 1


def forward_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Forward returns: return over the next N periods."""
    return close.shift(-periods) / close - 1


def overnight_returns(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """Overnight returns: open_t / close_{t-1} - 1."""
    return open_price / close.shift(1) - 1


def intraday_returns(open_price: pd.Series, close: pd.Series) -> pd.Series:
    """Intraday returns: close_t / open_t - 1."""
    return close / open_price - 1


# ---------------------------------------------------------------------------
# Rolling Statistics
# ---------------------------------------------------------------------------

def rolling_mean(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling mean (same as SMA)."""
    return close.rolling(window=window).mean()


def rolling_std(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling standard deviation."""
    return close.rolling(window=window).std()


def rolling_skew(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling skewness."""
    return close.rolling(window=window).skew()


def rolling_kurtosis(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling kurtosis (excess)."""
    return close.rolling(window=window).kurt()


def rolling_min(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling minimum."""
    return close.rolling(window=window).min()


def rolling_max(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling maximum."""
    return close.rolling(window=window).max()


def rolling_quantile(close: pd.Series, window: int = 20, q: float = 0.5) -> pd.Series:
    """Rolling quantile."""
    return close.rolling(window=window).quantile(q)


# ---------------------------------------------------------------------------
# Volatility Measures
# ---------------------------------------------------------------------------

def realized_volatility(close: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Realized volatility from log returns.

    Args:
        close: Price series.
        window: Rolling window.
        annualize: If True, multiply by sqrt(252) for daily data.
    """
    log_ret = log_returns(close)
    vol = log_ret.rolling(window=window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """Parkinson volatility estimator (uses high/low range)."""
    log_hl = np.log(high / low)
    variance = (log_hl ** 2).rolling(window=window).mean() / (4 * np.log(2))
    return np.sqrt(variance) * np.sqrt(252)


def garman_klass_volatility(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Garman-Klass volatility estimator."""
    log_hl = np.log(high / low) ** 2
    log_co = np.log(close / open_price) ** 2
    variance = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    vol = variance.rolling(window=window).mean()
    return np.sqrt(vol) * np.sqrt(252)


def yang_zhang_volatility(
    open_price: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Yang-Zhang volatility estimator (unbiased for opening jumps)."""
    log_oc = np.log(open_price / close.shift(1))
    log_co = np.log(close / open_price)
    log_ho = np.log(high / open_price)
    log_lo = np.log(low / open_price)

    rs_open = log_oc.rolling(window=window).var()
    rs_close = log_co.rolling(window=window).var()
    rs_hl = (log_ho ** 2 + log_lo ** 2).rolling(window=window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    variance = rs_open + k * rs_close + (1 - k) * rs_hl
    return np.sqrt(variance) * np.sqrt(252)


# ---------------------------------------------------------------------------
# Z-Score and Standardization
# ---------------------------------------------------------------------------

def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score: (x - mean) / std."""
    mean = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    return (close - mean) / std


def modified_zscore(close: pd.Series, window: int = 20) -> pd.Series:
    """Modified z-score using median and MAD (more robust to outliers)."""
    median = close.rolling(window=window).median()
    mad = (close - median).abs().rolling(window=window).median()
    return 0.6745 * (close - median) / mad


# ---------------------------------------------------------------------------
# Price-Volume Relationship
# ---------------------------------------------------------------------------

def price_volume_correlation(
    close: pd.Series, volume: pd.Series, window: int = 20,
) -> pd.Series:
    """Rolling correlation between price changes and volume."""
    return close.pct_change().rolling(window=window).corr(volume.pct_change())


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume z-score: how unusual is current volume."""
    mean = volume.rolling(window=window).mean()
    std = volume.rolling(window=window).std()
    return (volume - mean) / std


def volume_ratio(volume: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    """Volume ratio: short-term average / long-term average."""
    short_avg = volume.rolling(window=short_window).mean()
    long_avg = volume.rolling(window=long_window).mean()
    return short_avg / long_avg


# ---------------------------------------------------------------------------
# Cross-Asset / Relative Features
# ---------------------------------------------------------------------------

def relative_strength(close_a: pd.Series, close_b: pd.Series, window: int = 20) -> pd.Series:
    """Relative strength: (A / B) normalized over window."""
    ratio = close_a / close_b
    return (ratio - ratio.rolling(window=window).mean()) / ratio.rolling(window=window).std()


def beta(
    asset_returns: pd.Series, benchmark_returns: pd.Series, window: int = 60,
) -> pd.Series:
    """Rolling beta of asset relative to benchmark."""
    cov = asset_returns.rolling(window=window).cov(benchmark_returns)
    var = benchmark_returns.rolling(window=window).var()
    return cov / var


def correlation(
    series_a: pd.Series, series_b: pd.Series, window: int = 20,
) -> pd.Series:
    """Rolling correlation between two series."""
    return series_a.rolling(window=window).corr(series_b)
