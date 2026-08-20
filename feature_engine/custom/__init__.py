# -*- coding: utf-8 -*-
"""
Custom Feature Module

Domain-specific feature implementations for Indian equity markets.
These combine multiple raw indicators into composite scores.

Usage:
    from feature_engine import custom

    score = custom.breakout_score(high, low, close, volume)
    fii = custom.fii_strength(fii_buy, fii_sell)
    smart = custom.smart_money(close, volume, high, low)
"""

import numpy as np
import pandas as pd
from typing import Optional


def smart_money(
    close: pd.Series, volume: pd.Series,
    high: pd.Series, low: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Smart Money Index proxy.

    Combines late-day price action with volume to detect institutional activity.
    High values suggest institutional accumulation; low values suggest distribution.

    Formula: SMA(close * volume, period) / SMA(volume, period) normalized
    by price range.

    Returns:
        pd.Series: Smart money score (0-100 scale).
    """
    money_flow = close * volume
    sma_flow = money_flow.rolling(window=period).mean()
    sma_vol = volume.rolling(window=period).mean()
    avg_price = sma_flow / sma_vol.replace(0, np.nan)

    price_range = high.rolling(window=period).max() - low.rolling(window=period).min()
    price_range = price_range.replace(0, np.nan)

    score = ((avg_price - low.rolling(window=period).min()) / price_range) * 100
    return score.clip(0, 100)


def fii_strength(
    fii_buy: pd.Series, fii_sell: pd.Series, period: int = 5,
) -> pd.Series:
    """FII (Foreign Institutional Investor) strength indicator.

    Measures net FII activity as a percentage of total flow.

    Args:
        fii_buy: FII buy value (currency or volume).
        fii_sell: FII sell value.
        period: Rolling window for smoothing.

    Returns:
        pd.Series: FII strength (-100 to +100). Positive = net buying.
    """
    net = fii_buy - fii_sell
    total = fii_buy + fii_sell
    total = total.replace(0, np.nan)
    raw = (net / total) * 100
    return raw.rolling(window=period).mean().clip(-100, 100)


def breakout_score(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    lookback: int = 20, volume_mult: float = 1.5,
) -> pd.Series:
    """Composite breakout score.

    Combines price breakout above N-period high with volume surge.

    Scoring:
        0 = no breakout
        50 = price breakout, normal volume
        75 = price breakout + high volume
        100 = strong breakout (price > high + volume > mult * avg)

    Returns:
        pd.Series: Breakout score (0-100).
    """
    highest = high.rolling(window=lookback).max()
    avg_vol = volume.rolling(window=lookback).mean()

    price_breakout = close > highest.shift(1)
    volume_surge = volume > volume_mult * avg_vol

    score = pd.Series(0.0, index=close.index, dtype=float)
    score[price_breakout] = 50.0
    score[price_breakout & volume_surge] = 75.0

    # Strong breakout: price is > 1% above the breakout level
    strong = price_breakout & volume_surge & (close > highest.shift(1) * 1.01)
    score[strong] = 100.0

    return score


def institutional_score(
    fii_buy: pd.Series, fii_sell: pd.Series,
    dii_buy: pd.Series, dii_sell: pd.Series,
    period: int = 5,
) -> pd.Series:
    """Combined institutional activity score.

    Weights FII and DII net flows into a single score.
    FII weighted 0.6, DII weighted 0.4 (FII has more market impact).

    Returns:
        pd.Series: Institutional score (-100 to +100).
    """
    fii_net = fii_buy - fii_sell
    dii_net = dii_buy - dii_sell
    fii_total = (fii_buy + fii_sell).replace(0, np.nan)
    dii_total = (dii_buy + dii_sell).replace(0, np.nan)

    fii_pct = ((fii_net / fii_total) * 100).fillna(0)
    dii_pct = ((dii_net / dii_total) * 100).fillna(0)

    combined = 0.6 * fii_pct + 0.4 * dii_pct
    return combined.rolling(window=period).mean().clip(-100, 100)


def earnings_quality(
    revenue: pd.Series, net_income: pd.Series,
    operating_cf: Optional[pd.Series] = None,
) -> pd.Series:
    """Earnings quality proxy.

    Checks consistency between revenue growth, net income, and operating cash flow.
    Higher scores indicate more sustainable earnings.

    Returns:
        pd.Series: Quality score (0-100).
    """
    # Revenue consistency (positive growth)
    rev_growth = revenue.pct_change()
    rev_score = (rev_growth > 0).astype(float) * 25

    # Net income margin trend
    margin = net_income / revenue.replace(0, np.nan)
    margin_expanding = margin.expanding().mean()
    margin_score = ((margin > margin_expanding) & (margin > 0)).astype(float) * 25

    if operating_cf is not None:
        # Cash conversion: operating CF should be close to net income
        conversion = operating_cf / net_income.replace(0, np.nan)
        cf_score = ((conversion > 0.5) & (conversion < 2.0)).astype(float) * 25
    else:
        cf_score = pd.Series(12.5, index=revenue.index)

    # Positive net income
    profit_score = (net_income > 0).astype(float) * 25

    return (rev_score + margin_score + cf_score + profit_score).clip(0, 100)


def sector_rotation(
    sector_returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    """Sector rotation scoring.

    Identifies which sectors are leading/lagging based on relative strength.

    Args:
        sector_returns: DataFrame where columns are sector names, values are returns.
        window: Lookback window for ranking.

    Returns:
        DataFrame with same columns as input, values are rotation scores (-100 to +100).
        Positive = sector outperforming (leadership), negative = underperforming.
    """
    result = pd.DataFrame(index=sector_returns.index)
    for col in sector_returns.columns:
        cum_ret = sector_returns[col].rolling(window=window).sum()
        avg_cum = cum_ret.rolling(window=window).mean()
        std_cum = cum_ret.rolling(window=window).std()
        z = (cum_ret - avg_cum) / std_cum.replace(0, np.nan)
        result[col] = (z * 50).clip(-100, 100)
    return result


def orderflow_score(
    bid_volume: pd.Series, ask_volume: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Orderflow imbalance score.

    Measures buying vs selling pressure from bid/ask volume.

    Returns:
        pd.Series: Score (-100 to +100). Positive = buying pressure.
    """
    total = bid_volume + ask_volume
    total = total.replace(0, np.nan)
    imbalance = ((bid_volume - ask_volume) / total) * 100
    return imbalance.rolling(window=period).mean().clip(-100, 100)


def news_sentiment(
    sentiment_scores: pd.Series,
    volume_weight: Optional[pd.Series] = None,
    decay: float = 0.9,
    period: int = 5,
) -> pd.Series:
    """Aggregated news sentiment with exponential decay.

    Recent news has more impact than older news.

    Args:
        sentiment_scores: Raw sentiment (-1 to +1 per article/timestamp).
        volume_weight: Optional volume-weighted multiplier.
        decay: Exponential decay factor (0.9 = 10% decay per step).
        period: Rolling aggregation window.

    Returns:
        pd.Series: Aggregated sentiment score (-100 to +100).
    """
    if volume_weight is not None:
        weighted = sentiment_scores * volume_weight
    else:
        weighted = sentiment_scores

    # Apply exponential decay to recent scores
    weights = pd.Series(
        [decay ** i for i in range(period)][::-1],
        index=range(period),
    )
    result = weighted.rolling(window=period).apply(
        lambda x: np.dot(x, weights.values[: len(x)]) / weights.values[: len(x)].sum(),
        raw=True,
    )
    return (result * 100).clip(-100, 100)
