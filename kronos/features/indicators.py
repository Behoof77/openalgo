"""
Numpy-based technical indicators for the multi-factor signal pipeline.

Every function is pure numpy (no TA-Lib dependency) and operates on
1-D arrays extracted from an OHLCV DataFrame.  Functions return
``np.ndarray`` with the same length as the input; leading NaN values
are present until the lookback window is satisfied.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd

from kronos.utils.helpers import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Individual indicators
# ---------------------------------------------------------------------------


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index using Wilder's smoothing.

    Parameters
    ----------
    close : np.ndarray
        1-D closing prices.
    period : int
        Lookback window (default 14).

    Returns
    -------
    np.ndarray
        RSI values in ``[0, 100]``.  Leading ``period`` entries are NaN.
    """
    deltas = np.diff(close, prepend=close[0])
    gains = np.where(deltas > 0, deltas, 0.0).astype(np.float64)
    losses = np.where(deltas < 0, -deltas, 0.0).astype(np.float64)

    avg_gain = np.full_like(close, np.nan, dtype=np.float64)
    avg_loss = np.full_like(close, np.nan, dtype=np.float64)

    avg_gain[period] = float(np.mean(gains[1 : period + 1]))
    avg_loss[period] = float(np.mean(losses[1 : period + 1]))

    for i in range(period + 1, len(close)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period

    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def compute_bollinger_bands(
    close: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (middle, upper, lower).

    Returns
    -------
    tuple of np.ndarray
        ``(middle, upper, lower)`` — each with leading NaN values.
    """
    middle = np.full_like(close, np.nan, dtype=np.float64)
    upper = np.full_like(close, np.nan, dtype=np.float64)
    lower = np.full_like(close, np.nan, dtype=np.float64)

    for i in range(period - 1, len(close)):
        window = close[i - period + 1 : i + 1]
        m = float(np.mean(window))
        s = float(np.std(window, ddof=1))
        middle[i] = m
        upper[i] = m + std_dev * s
        lower[i] = m - std_dev * s

    return middle, upper, lower


def compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average True Range.

    Uses Wilder's smoothing.  The first ``period`` entries are NaN.
    """
    tr = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    atr = np.full_like(close, np.nan, dtype=np.float64)
    atr[period] = float(np.nanmean(tr[1 : period + 1]))
    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def compute_adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
) -> tuple[np.ndarray, np.ndarray]:
    """Average Directional Index and its direction.

    Returns
    -------
    adx : np.ndarray
        ADX values in ``[0, 100]``.
    direction : np.ndarray
        ``+1`` when ``+DI > -DI`` (bullish), ``-1`` when ``-DI > +DI``
        (bearish), ``0`` when equal.  Leading ``period + 1`` entries are NaN.
    """
    # True range
    tr = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(1, len(close)):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # +DM and -DM
    plus_dm = np.zeros_like(close, dtype=np.float64)
    minus_dm = np.zeros_like(close, dtype=np.float64)
    for i in range(1, len(close)):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down

    # Smoothed TR, +DM, -DM  (Wilder's)
    atr = np.full_like(close, np.nan, dtype=np.float64)
    smoothed_plus = np.full_like(close, np.nan, dtype=np.float64)
    smoothed_minus = np.full_like(close, np.nan, dtype=np.float64)

    atr[period] = float(np.nanmean(tr[1 : period + 1]))
    smoothed_plus[period] = float(np.sum(plus_dm[1 : period + 1]))
    smoothed_minus[period] = float(np.sum(minus_dm[1 : period + 1]))

    for i in range(period + 1, len(close)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
        smoothed_plus[i] = (smoothed_plus[i - 1] * (period - 1) + plus_dm[i]) / period
        smoothed_minus[i] = (smoothed_minus[i - 1] * (period - 1) + minus_dm[i]) / period

    # +DI, -DI
    plus_di = np.full_like(close, np.nan, dtype=np.float64)
    minus_di = np.full_like(close, np.nan, dtype=np.float64)
    dx = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period, len(close)):
        if atr[i] > 1e-10:
            plus_di[i] = 100.0 * smoothed_plus[i] / atr[i]
            minus_di[i] = 100.0 * smoothed_minus[i] / atr[i]
            di_diff = abs(plus_di[i] - minus_di[i])
            di_sum = plus_di[i] + minus_di[i]
            dx[i] = 100.0 * di_diff / di_sum if di_sum > 1e-10 else 0.0

    # ADX = smoothed DX
    adx = np.full_like(close, np.nan, dtype=np.float64)
    adx[period + period - 1] = float(np.nanmean(dx[period : period + period]))
    for i in range(period + period, len(close)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    # Direction
    direction = np.full_like(close, np.nan, dtype=np.float64)
    for i in range(period, len(close)):
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]):
            direction[i] = 0.0
        elif plus_di[i] > minus_di[i]:
            direction[i] = 1.0
        elif minus_di[i] > plus_di[i]:
            direction[i] = -1.0
        else:
            direction[i] = 0.0

    return adx, direction


# ---------------------------------------------------------------------------
# Aggregated snapshot
# ---------------------------------------------------------------------------


@dataclass
class TechnicalSnapshot:
    """Latest values of all computed technical indicators."""

    rsi: float = 50.0
    bb_width_pct: float = 0.0  # (upper - lower) / middle * 100
    bb_position: float = 0.0  # -1 below lower, +1 above upper, 0 inside
    atr_pct: float = 0.0  # ATR / close * 100 (volatility)
    adx: float = 20.0
    adx_direction: float = 0.0  # +1 bullish, -1 bearish
    close: float = 0.0


def compute_all_technicals(df: pd.DataFrame) -> TechnicalSnapshot:
    """Compute all indicators on *df* and return a snapshot of the latest
    values (``TechnicalSnapshot``).

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns ``open``, ``high``, ``low``, ``close`` (and
        optionally ``volume``).

    Returns
    -------
    TechnicalSnapshot
        Populated with the last bar's values.  If indicators have not
        stabilised yet (e.g. not enough bars), sensible defaults are
        returned.
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    last_close = float(close[-1])

    # RSI
    rsi_arr = compute_rsi(close)
    rsi_val = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else 50.0

    # Bollinger Bands
    mid, upper, lower = compute_bollinger_bands(close)
    if not np.isnan(mid[-1]) and mid[-1] > 1e-10:
        bw = ((upper[-1] - lower[-1]) / mid[-1]) * 100.0
        if last_close >= upper[-1]:
            bp = 1.0
        elif last_close <= lower[-1]:
            bp = -1.0
        else:
            bp = 0.0
    else:
        bw = 0.0
        bp = 0.0

    # ATR
    atr_arr = compute_atr(high, low, close)
    atr_pct = (float(atr_arr[-1]) / last_close * 100.0) if not np.isnan(atr_arr[-1]) and last_close > 1e-10 else 0.0

    # ADX
    adx_arr, direction_arr = compute_adx(high, low, close)
    adx_val = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else 20.0
    adx_dir = float(direction_arr[-1]) if not np.isnan(direction_arr[-1]) else 0.0

    return TechnicalSnapshot(
        rsi=cast(float, round(rsi_val, 1)),
        bb_width_pct=cast(float, round(bw, 2)),
        bb_position=bp,
        atr_pct=cast(float, round(atr_pct, 2)),
        adx=cast(float, round(adx_val, 1)),
        adx_direction=adx_dir,
        close=last_close,
    )
