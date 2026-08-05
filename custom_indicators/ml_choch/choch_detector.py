"""
CHoCH Detector -- Change of Character detection via market structure breaks.

Detects bullish and bearish CHoCH events by tracking swing highs/lows and
identifying when price breaks through these structural levels.

Category: trend / market-structure
"""

import numpy as np


def detect_choch(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    swing_len: int = 5,
) -> dict:
    """Detect CHoCH (Change of Character) events from OHLCV data.

    Uses pivot highs/lows to identify swing structure, then detects when
    close breaks through the most recent swing level, indicating a
    change in market character.

    Args:
        high: High prices (numpy array, pandas Series, or list)
        low: Low prices (numpy array, pandas Series, or list)
        close: Close prices (numpy array, pandas Series, or list)
        swing_len: Pivot lookback/lookforward length (default 5)

    Returns:
        dict with keys:
            is_bullish: np.ndarray[bool] -- True on bullish CHoCH bars
            is_bearish: np.ndarray[bool] -- True on bearish CHoCH bars
            market_trend: np.ndarray[int] -- +1 bullish, -1 bearish, 0 neutral
            last_swing_high: np.ndarray[float] -- Most recent swing high at each bar
            last_swing_low: np.ndarray[float] -- Most recent swing low at each bar
            swing_high_idx: np.ndarray[int] -- Bar index of last swing high
            swing_low_idx: np.ndarray[int] -- Bar index of last swing low
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    # --- Pivot detection (manual -- openalgo.ta lacks pivothigh/pivotlow) ---
    pivot_high = np.full(n, np.nan, dtype=np.float64)
    pivot_low = np.full(n, np.nan, dtype=np.float64)
    for i in range(swing_len, n - swing_len):
        if all(high[i] >= high[i - j] for j in range(1, swing_len + 1)) and \
           all(high[i] >= high[i + j] for j in range(1, swing_len + 1)):
            pivot_high[i] = high[i]
        if all(low[i] <= low[i - j] for j in range(1, swing_len + 1)) and \
           all(low[i] <= low[i + j] for j in range(1, swing_len + 1)):
            pivot_low[i] = low[i]

    # --- Stateful CHoCH detection (sequential -- each bar depends on previous state) ---
    last_swing_high = np.full(n, np.nan)
    last_swing_low = np.full(n, np.nan)
    swing_high_idx = np.full(n, -1, dtype=np.int64)
    swing_low_idx = np.full(n, -1, dtype=np.int64)
    market_trend = np.zeros(n, dtype=np.int64)
    is_bullish = np.zeros(n, dtype=bool)
    is_bearish = np.zeros(n, dtype=bool)

    _last_sh = np.nan
    _last_sl = np.nan
    _last_hi_idx = -1
    _last_lo_idx = -1
    _trend = 0  # 0=neutral, 1=bullish, -1=bearish

    for i in range(n):
        # Update swing highs
        if not np.isnan(pivot_high[i]):
            _last_sh = pivot_high[i]
            _last_hi_idx = i - swing_len  # pivot is confirmed swing_len bars back

        # Update swing lows
        if not np.isnan(pivot_low[i]):
            _last_sl = pivot_low[i]
            _last_lo_idx = i - swing_len

        # Record state
        last_swing_high[i] = _last_sh
        last_swing_low[i] = _last_sl
        swing_high_idx[i] = _last_hi_idx
        swing_low_idx[i] = _last_lo_idx

        # Detect CHoCH -- bullish: price breaks above last swing high when trend was not bullish
        if not np.isnan(_last_sh) and _trend <= 0 and close[i] > _last_sh:
            is_bullish[i] = True
            _trend = 1

        # Detect CHoCH -- bearish: price breaks below last swing low when trend was not bearish
        if not np.isnan(_last_sl) and _trend >= 0 and close[i] < _last_sl:
            is_bearish[i] = True
            _trend = -1

        market_trend[i] = _trend

    return {
        "is_bullish": is_bullish,
        "is_bearish": is_bearish,
        "market_trend": market_trend,
        "last_swing_high": last_swing_high,
        "last_swing_low": last_swing_low,
        "swing_high_idx": swing_high_idx,
        "swing_low_idx": swing_low_idx,
    }
