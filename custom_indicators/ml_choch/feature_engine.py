"""
Feature Engine -- Extract ML features from CHoCH events.

Computes the three feature dimensions used by the KNN model:
- Volume Delta: average buying vs selling pressure imbalance
- Displacement (Z): normalized price move magnitude relative to ATR
- Price Velocity: speed of the price move (move per bar)

All features use openalgo.ta Rust-core primitives where possible.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd
from openalgo import ta


@dataclass
class ChochEvent:
    """A single CHoCH event with extracted features and optional outcome label.

    Attributes:
        bar_index: Index in the OHLCV array where CHoCH occurred
        volume_delta: Average buying-selling pressure ratio over recent bars
        displacement_z: Price move / ATR (normalized magnitude)
        price_velocity: Price move / duration (speed in price-units per bar)
        is_bullish: True if bullish CHoCH, False if bearish
        outcome_value: 1.0 if favorable run exceeded adverse, -1.0 otherwise (training label)
        favorable_run: Maximum favorable price move in lookahead window (training label)
    """

    bar_index: int
    volume_delta: float
    displacement_z: float
    price_velocity: float
    is_bullish: bool
    outcome_value: float = 0.0
    favorable_run: float = 0.0

    def feature_vector(self) -> np.ndarray:
        """Return the 3D feature vector for KNN distance computation."""
        return np.array([self.volume_delta, self.displacement_z, self.price_velocity])


def compute_volume_delta(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                         volume: np.ndarray) -> np.ndarray:
    """Compute per-bar volume delta (buying vs selling pressure imbalance).

    volume_delta = (buying_volume - selling_volume) / volume

    where buying_volume approximates volume * ((close - low) / range)
    and   selling_volume approximates volume * ((high - close) / range)

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume data

    Returns:
        np.ndarray of per-bar volume delta values (float64)
    """
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)

    candle_range = np.maximum(high - low, 1e-6)
    buying = volume * ((close - low) / candle_range)
    selling = volume * ((high - close) / candle_range)

    with np.errstate(invalid="ignore", divide="ignore"):
        vol_delta = np.where(volume > 0, (buying - selling) / volume, 0.0)

    return vol_delta


def extract_features(
    df: pd.DataFrame,
    choch_indices: np.ndarray,
    choch_directions: np.ndarray,
    atr_period: int = 14,
    vol_lookback: int = 50,
) -> List[ChochEvent]:
    """Extract ML features for each CHoCH event from OHLCV DataFrame.

    For each CHoCH bar, computes:
    1. Volume Delta -- average of recent bar volume deltas
    2. Displacement Z -- total move / ATR at the CHoCH bar
    3. Price Velocity -- total move / number of bars in the move

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        choch_indices: Boolean array marking CHoCH bars (True = CHoCH)
        choch_directions: Boolean array (True = bullish, False = bearish)
            Only relevant where choch_indices is True
        atr_period: ATR period for displacement normalization
        vol_lookback: Number of bars to average volume delta over

    Returns:
        List of ChochEvent objects with extracted features (no outcome labels)
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    opn = df["open"].values.astype(np.float64)
    n = len(close)

    # Pre-compute indicators via openalgo.ta (Rust core)
    atr = ta.atr(high, low, close, atr_period)
    atr = np.asarray(atr, dtype=np.float64)

    vol_delta = compute_volume_delta(high, low, close, volume)

    # Rolling average of volume delta
    vol_delta_avg = ta.sma(vol_delta, vol_lookback)
    vol_delta_avg = np.asarray(vol_delta_avg, dtype=np.float64)

    # Get indices where CHoCH fires
    choch_idx = np.where(choch_indices)[0]

    events: List[ChochEvent] = []
    for idx in choch_idx:
        if idx < 1 or idx >= n:
            continue

        is_bull = bool(choch_directions[idx])

        # Find the duration from the swing point to the CHoCH bar
        # For bullish: look back from the swing high index
        # For bearish: look back from the swing low index
        # Approximate: use the most recent swing as the reference point
        duration = max(1, min(idx, 50))

        # Volume delta feature -- average over recent bars
        start = max(0, idx - vol_lookback)
        vd_vals = vol_delta[start:idx + 1]
        vd_valid = vd_vals[~np.isnan(vd_vals)]
        vol_delta_feature = float(np.mean(vd_valid)) if len(vd_valid) > 0 else 0.0

        # Displacement feature -- total move / ATR
        total_move = abs(close[idx] - opn[max(0, idx - duration)])
        current_atr = atr[idx] if not np.isnan(atr[idx]) else 1.0
        displacement_z = total_move / max(current_atr, 1e-6)

        # Velocity feature -- total move / duration bars
        velocity = total_move / duration

        events.append(ChochEvent(
            bar_index=idx,
            volume_delta=vol_delta_feature,
            displacement_z=displacement_z,
            price_velocity=velocity,
            is_bullish=is_bull,
        ))

    return events


def label_events(
    df: pd.DataFrame,
    events: List[ChochEvent],
    lookahead: int = 20,
) -> List[ChochEvent]:
    """Label historical CHoCH events with outcome using future price data.

    Looks ahead `lookahead` bars after each CHoCH event to determine:
    - Whether the move was successful (favorable > adverse)
    - The maximum favorable run (used for TP target computation)

    Args:
        df: OHLCV DataFrame
        events: List of ChochEvent objects to label
        lookahead: Number of bars to look ahead for outcome assessment

    Returns:
        List of ChochEvent objects with outcome_value and favorable_run populated
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    n = len(close)

    labeled: List[ChochEvent] = []
    for event in events:
        idx = event.bar_index
        if idx + lookahead >= n:
            continue  # Cannot label -- not enough future data

        ref_price = close[idx]
        max_favorable = 0.0
        max_adverse = 0.0

        for look in range(1, lookahead + 1):
            future_idx = idx + look
            if future_idx >= n:
                break

            if event.is_bullish:
                max_favorable = max(max_favorable, high[future_idx] - ref_price)
                max_adverse = max(max_adverse, ref_price - low[future_idx])
            else:
                max_favorable = max(max_favorable, ref_price - low[future_idx])
                max_adverse = max(max_adverse, high[future_idx] - ref_price)

        event.outcome_value = 1.0 if max_favorable > max_adverse else -1.0
        event.favorable_run = max_favorable
        labeled.append(event)

    return labeled
