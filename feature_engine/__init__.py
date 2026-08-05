# -*- coding: utf-8 -*-
"""
OpenAlgo Feature Engine

A unified, reusable feature calculation engine for algorithmic trading.
Wraps the openalgo.indicators library and adds statistical, custom,
label, and preprocessing modules.

Usage:
    from feature_engine import technical, statistical, custom, labels, preprocessing

    # Technical indicators (delegates to openalgo.indicators)
    ema_20 = technical.ema(close, period=20)
    rsi_14 = technical.rsi(close, period=14)
    macd_line, signal, hist = technical.macd(close)

    # Statistical features
    returns = statistical.simple_returns(close)
    vol = statistical.realized_volatility(close, window=20)

    # Custom features
    score = custom.breakout_score(high, low, close, volume)

    # Labels
    target = labels.trend_target(close, horizon=5, threshold=0.02)

    # Preprocessing
    scaled = preprocessing.min_max_scale(features)
"""

from feature_engine import (
    technical,
    statistical,
    custom,
    labels,
    preprocessing,
)

__all__ = [
    "technical",
    "statistical",
    "custom",
    "labels",
    "preprocessing",
]
