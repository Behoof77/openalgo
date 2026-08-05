# -*- coding: utf-8 -*-
"""
Technical Indicators Module

Wraps the openalgo.indicators library into a clean, categorized API.
All functions accept pandas Series and return pandas Series or DataFrames.

Usage:
    from feature_engine import technical

    ema_20 = technical.ema(close, period=20)
    rsi_14 = technical.rsi(close, period=14)
    macd_df = technical.macd(close)
    atr_14 = technical.atr(high, low, close, period=14)
"""

import pandas as pd
from typing import Tuple, Optional

try:
    from openalgo import ta as _ta
except ImportError:
    _ta = None


def _require_ta():
    if _ta is None:
        raise ImportError(
            "openalgo package is required for technical indicators. "
            "Install with: pip install openalgo"
        )


# ---------------------------------------------------------------------------
# Trend Indicators
# ---------------------------------------------------------------------------

def sma(close: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    _require_ta()
    return pd.Series(_ta.sma(close, period), index=close.index, name=f"SMA{period}")


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    _require_ta()
    return pd.Series(_ta.ema(close, period), index=close.index, name=f"EMA{period}")


def wma(close: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average."""
    _require_ta()
    return pd.Series(_ta.wma(close, period), index=close.index, name=f"WMA{period}")


def dema(close: pd.Series, period: int = 20) -> pd.Series:
    """Double Exponential Moving Average."""
    _require_ta()
    return pd.Series(_ta.dema(close, period), index=close.index, name=f"DEMA{period}")


def tema(close: pd.Series, period: int = 20) -> pd.Series:
    """Triple Exponential Moving Average."""
    _require_ta()
    return pd.Series(_ta.tema(close, period), index=close.index, name=f"TEMA{period}")


def hma(close: pd.Series, period: int = 20) -> pd.Series:
    """Hull Moving Average."""
    _require_ta()
    return pd.Series(_ta.hma(close, period), index=close.index, name=f"HMA{period}")


def vwma(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Weighted Moving Average."""
    _require_ta()
    return pd.Series(_ta.vwma(close, volume, period), index=close.index, name=f"VWMA{period}")


def alma(close: pd.Series, period: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average."""
    _require_ta()
    return pd.Series(_ta.alma(close, period, offset, sigma), index=close.index, name=f"ALMA{period}")


def kama(close: pd.Series, period: int = 10) -> pd.Series:
    """Kaufman Adaptive Moving Average."""
    _require_ta()
    return pd.Series(_ta.kama(close, period), index=close.index, name=f"KAMA{period}")


def zlema(close: pd.Series, period: int = 14) -> pd.Series:
    """Zero Lag Exponential Moving Average."""
    _require_ta()
    return pd.Series(_ta.zlema(close, period), index=close.index, name=f"ZLEMA{period}")


def supertrend(
    high: pd.Series, low: pd.Series, close: pd.Series,
    period: int = 10, multiplier: float = 3.0,
) -> Tuple[pd.Series, pd.Series]:
    """Supertrend indicator.

    Returns:
        Tuple of (supertrend_line, direction) where direction=+1 (long) or -1 (short).
    """
    _require_ta()
    st, direction = _ta.supertrend(high, low, close, period, multiplier)
    return (
        pd.Series(st, index=close.index, name="supertrend"),
        pd.Series(direction, index=close.index, name="supertrend_direction"),
    )


def ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series,
    tenkan: int = 9, kijun: int = 26, senkou_b: int = 52,
) -> pd.DataFrame:
    """Ichimoku Cloud.

    Returns DataFrame with columns: tenkan, kijun, senkou_a, senkou_b, chikou.
    """
    _require_ta()
    t, k, sa, sb, chikou = _ta.ichimoku(
        high, low, close,
        tenkan_period=tenkan, kijun_period=kijun, senkou_b_period=senkou_b,
    )
    return pd.DataFrame({
        "tenkan": t, "kijun": k, "senkou_a": sa, "senkou_b": sb, "chikou": chikou,
    }, index=close.index)


def donchian(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Donchian Channels.

    Returns:
        Tuple of (upper, middle, lower).
    """
    _require_ta()
    upper, middle, lower = _ta.donchian(high, low, period)
    idx = high.index
    return (
        pd.Series(upper, index=idx, name="donchian_upper"),
        pd.Series(middle, index=idx, name="donchian_middle"),
        pd.Series(lower, index=idx, name="donchian_lower"),
    )


# ---------------------------------------------------------------------------
# Momentum Indicators
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    _require_ta()
    return pd.Series(_ta.rsi(close, period), index=close.index, name=f"RSI{period}")


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9,
) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns: macd, signal, histogram.
    """
    _require_ta()
    macd_line, signal_line, histogram = _ta.macd(close, fast, slow, signal)
    return pd.DataFrame({
        "macd": macd_line, "signal": signal_line, "histogram": histogram,
    }, index=close.index)


def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k: int = 14, d: int = 3, smooth: int = 3,
) -> pd.DataFrame:
    """Stochastic Oscillator.

    Returns DataFrame with columns: k, d.
    """
    _require_ta()
    k_line, d_line = _ta.stochastic(high, low, close, k, d, smooth)
    return pd.DataFrame({
        "stoch_k": k_line, "stoch_d": d_line,
    }, index=close.index)


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    _require_ta()
    return pd.Series(_ta.cci(high, low, close, period), index=close.index, name=f"CCI{period}")


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Williams %R."""
    _require_ta()
    return pd.Series(_ta.williams_r(high, low, close, period), index=close.index, name=f"WR{period}")


# ---------------------------------------------------------------------------
# Volatility Indicators
# ---------------------------------------------------------------------------

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    _require_ta()
    return pd.Series(_ta.atr(high, low, close, period), index=close.index, name=f"ATR{period}")


def bbands(
    close: pd.Series, period: int = 20, std: float = 2.0,
) -> pd.DataFrame:
    """Bollinger Bands.

    Returns DataFrame with columns: upper, middle, lower.
    """
    _require_ta()
    upper, middle, lower = _ta.bbands(close, period, std)
    return pd.DataFrame({
        "bb_upper": upper, "bb_middle": middle, "bb_lower": lower,
    }, index=close.index)


def keltner(
    high: pd.Series, low: pd.Series, close: pd.Series,
    ema_period: int = 20, atr_period: int = 10, multiplier: float = 2.0,
) -> pd.DataFrame:
    """Keltner Channels.

    Returns DataFrame with columns: upper, middle, lower.
    """
    _require_ta()
    upper, middle, lower = _ta.keltner(high, low, close, ema_period, atr_period, multiplier)
    return pd.DataFrame({
        "kc_upper": upper, "kc_middle": middle, "kc_lower": lower,
    }, index=close.index)


# ---------------------------------------------------------------------------
# Volume Indicators
# ---------------------------------------------------------------------------

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    _require_ta()
    return pd.Series(_ta.obv(close, volume), index=close.index, name="OBV")


def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price."""
    _require_ta()
    return pd.Series(_ta.vwap(high, low, close, volume), index=close.index, name="VWAP")


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Money Flow Index."""
    _require_ta()
    return pd.Series(_ta.mfi(high, low, close, volume, period), index=close.index, name=f"MFI{period}")


def adl(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    """Accumulation/Distribution Line."""
    _require_ta()
    return pd.Series(_ta.adl(high, low, close, volume), index=close.index, name="ADL")


def cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Chaikin Money Flow."""
    _require_ta()
    return pd.Series(_ta.cmf(high, low, close, volume, period), index=close.index, name=f"CMF{period}")


def rvol(close: pd.Series, volume: pd.Series, period: int = 20) -> pd.Series:
    """Relative Volume."""
    _require_ta()
    return pd.Series(_ta.rvol(close, volume, period), index=close.index, name=f"RVOL{period}")


# ---------------------------------------------------------------------------
# Hybrid / Other Indicators
# ---------------------------------------------------------------------------

def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average Directional Index."""
    _require_ta()
    return pd.Series(_ta.adx(high, low, close, period), index=close.index, name=f"ADX{period}")


def aroon(
    high: pd.Series, low: pd.Series, period: int = 25,
) -> Tuple[pd.Series, pd.Series]:
    """Aroon Oscillator.

    Returns:
        Tuple of (aroon_up, aroon_down).
    """
    _require_ta()
    up, down = _ta.aroon(high, low, period)
    return (
        pd.Series(up, index=high.index, name="aroon_up"),
        pd.Series(down, index=high.index, name="aroon_down"),
    )


def sar(high: pd.Series, low: pd.Series, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Parabolic SAR."""
    _require_ta()
    return pd.Series(_ta.sar(high, low, acceleration, maximum), index=high.index, name="SAR")


# ---------------------------------------------------------------------------
# Signal Helpers
# ---------------------------------------------------------------------------

def crossover(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where series `a` crosses above `b`."""
    _require_ta()
    return _ta.crossover(a, b)


def crossunder(a: pd.Series, b: pd.Series) -> pd.Series:
    """True where series `a` crosses below `b`."""
    _require_ta()
    return _ta.crossunder(a, b)


def exrem(primary: pd.Series, secondary: pd.Series) -> pd.Series:
    """Remove consecutive same-direction signals."""
    _require_ta()
    return _ta.exrem(primary, secondary)


def flip(trigger_on: pd.Series, trigger_off: pd.Series) -> pd.Series:
    """Flip between two states."""
    _require_ta()
    return _ta.flip(trigger_on, trigger_off)


# ---------------------------------------------------------------------------
# Convenience: Get All Indicators for a OHLCV DataFrame
# ---------------------------------------------------------------------------

def compute_all(
    df: pd.DataFrame,
    ohlcv_cols: Optional[dict] = None,
    include: Optional[list] = None,
) -> pd.DataFrame:
    """Compute all technical indicators for an OHLCV DataFrame.

    Args:
        df: DataFrame with OHLCV columns.
        ohlcv_cols: Column name mapping. Default: open, high, low, close, volume.
        include: List of indicator names to compute. None = all.

    Returns:
        DataFrame with original columns plus all computed indicators.
    """
    cols = ohlcv_cols or {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
    }
    result = df.copy()
    o, h, l, c, v = (df[cols[k]] for k in ("open", "high", "low", "close", "volume"))

    indicator_fns = {
        "sma_20": lambda: sma(c, 20),
        "sma_50": lambda: sma(c, 50),
        "ema_20": lambda: ema(c, 20),
        "ema_50": lambda: ema(c, 50),
        "rsi_14": lambda: rsi(c, 14),
        "macd": lambda: macd(c),
        "bbands": lambda: bbands(c),
        "atr_14": lambda: atr(h, l, c, 14),
        "adx_14": lambda: adx(h, l, c, 14),
        "obv": lambda: obv(c, v),
        "vwap": lambda: vwap(h, l, c, v),
    }

    targets = include or list(indicator_fns.keys())
    for name in targets:
        if name not in indicator_fns:
            continue
        try:
            val = indicator_fns[name]()
            if isinstance(val, pd.DataFrame):
                for col in val.columns:
                    result[f"{name}_{col}"] = val[col]
            else:
                result[name] = val
        except Exception:
            pass

    return result
