# -*- coding: utf-8 -*-
"""
Label Generation Module

Create target variables for supervised ML from price data.
Supports regression targets (forward returns) and classification targets
(trend direction, breakout, etc.).

Usage:
    from feature_engine import labels

    target = labels.trend_target(close, horizon=5, threshold=0.02)
    breakout = labels.breakout_target(close, high, window=20, threshold=0.03)
    cls = labels.classification_labels(close, horizon=5, thresholds=[-0.01, 0.01])
"""

import numpy as np
import pandas as pd
from typing import Optional, List


def trend_target(
    close: pd.Series,
    horizon: int = 5,
    threshold: float = 0.02,
) -> pd.Series:
    """Binary trend target: 1 if price rises > threshold in next N bars.

    Args:
        close: Price series.
        horizon: Number of forward bars to look.
        threshold: Minimum return threshold (e.g., 0.02 = 2%).

    Returns:
        pd.Series: 1 = uptrend, 0 = no trend / downtrend.
    """
    future_return = close.shift(-horizon) / close - 1
    return (future_return > threshold).astype(int)


def breakout_target(
    close: pd.Series,
    high: pd.Series,
    window: int = 20,
    horizon: int = 5,
    threshold: float = 0.03,
) -> pd.Series:
    """Breakout target: 1 if price breaks above N-period high by threshold.

    Args:
        close: Close prices.
        high: High prices.
        window: Lookback period for highest high.
        horizon: Forward bars to check.
        threshold: Minimum breakout magnitude.

    Returns:
        pd.Series: 1 = breakout occurred, 0 = no breakout.
    """
    highest = high.rolling(window=window).max()
    future_high = high.shift(-1).rolling(window=horizon).max()
    breakout = (future_high > highest * (1 + threshold))
    return breakout.astype(int)


def trend_target_multiclass(
    close: pd.Series,
    horizon: int = 5,
    thresholds: Optional[List[float]] = None,
) -> pd.Series:
    """Multi-class trend target.

    Args:
        close: Price series.
        horizon: Forward bars.
        thresholds: List of (lower, upper) boundaries for classes.
            Default: [-0.02, -0.01, 0.01, 0.02] → 5 classes.

    Returns:
        pd.Series: Integer class labels (0 = strong down, N = strong up).
    """
    if thresholds is None:
        thresholds = [-0.02, -0.01, 0.01, 0.02]

    future_return = close.shift(-horizon) / close - 1
    labels = pd.Series(0, index=close.index, dtype=int)

    for i, t in enumerate(thresholds):
        if t < 0:
            labels[future_return < t] = i
        else:
            labels[future_return >= t] = i + 1

    return labels


def classification_labels(
    close: pd.Series,
    horizon: int = 5,
    thresholds: Optional[List[float]] = None,
) -> pd.Series:
    """Generic classification labels from forward returns.

    Args:
        close: Price series.
        horizon: Forward bars.
        thresholds: Return thresholds defining class boundaries.
            Default: [-0.01, 0.01] → 3 classes (down, flat, up).

    Returns:
        pd.Series: Integer class labels.
    """
    if thresholds is None:
        thresholds = [-0.01, 0.01]

    future_return = close.shift(-horizon) / close - 1
    labels = pd.cut(
        future_return,
        bins=[-np.inf] + thresholds + [np.inf],
        labels=list(range(len(thresholds) + 1)),
    )
    return labels.astype(float).fillna(1).astype(int)


def regression_target(
    close: pd.Series,
    horizon: int = 5,
) -> pd.Series:
    """Forward return as regression target.

    Args:
        close: Price series.
        horizon: Forward bars.

    Returns:
        pd.Series: Forward return (raw percentage).
    """
    return close.shift(-horizon) / close - 1


def volatility_target(
    close: pd.Series,
    horizon: int = 20,
) -> pd.Series:
    """Forward volatility as regression target.

    Args:
        close: Price series.
        horizon: Forward bars.

    Returns:
        pd.Series: Realized volatility over the next N bars.
    """
    log_ret = np.log(close / close.shift(1))
    return log_ret.rolling(window=horizon).std().shift(-horizon) * np.sqrt(252)


def max_drawdown_target(
    close: pd.Series,
    horizon: int = 20,
) -> pd.Series:
    """Maximum drawdown over the next N bars (risk target).

    Returns negative values (drawdowns are losses).

    Returns:
        pd.Series: Max drawdown as negative fraction.
    """
    result = pd.Series(0.0, index=close.index)
    for i in range(len(close) - horizon):
        future = close.iloc[i: i + horizon + 1]
        running_max = future.cummax()
        dd = (future - running_max) / running_max
        result.iloc[i] = dd.min()
    return result
