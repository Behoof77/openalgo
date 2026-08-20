"""
Signal Generator -- Orchestrate CHoCH detection, feature extraction, and ML prediction.

Combines the detector, feature engine, and Random Forest model into a single
pipeline that processes an OHLCV DataFrame and produces trading signals with
probability scores and take-profit targets.

Training approach: Rolling window retraining. At each CHoCH event, a fresh
Random Forest is trained on only the events within the rolling window that
precede it. This simulates real-time conditions -- the model at bar N only
knows about events from bars [N-window, N).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from openalgo import ta

from .choch_detector import detect_choch
from .choch_model import ChochMLModel
from .feature_engine import ChochEvent, extract_features, label_events


@dataclass
class SignalResult:
    """A trading signal produced by the ML CHoCH pipeline.

    Attributes:
        bar_index: Index in the DataFrame where the signal fires
        direction: True for bullish, False for bearish
        probability: Random Forest success probability (0-100)
        score: Significance score (probability if >= min_score, else 0)
        tp1: Conservative take-profit price (mean * scalar)
        tp2: Median take-profit price
        tp3: Aggressive take-profit price (75th percentile)
        is_valid: True if probability >= min_score
        db_size: Number of events in the rolling database at signal time
    """

    bar_index: int
    direction: bool
    probability: float
    score: float
    tp1: float
    tp2: float
    tp3: float
    is_valid: bool
    db_size: int


def generate_signals(
    df: pd.DataFrame,
    model: Optional[ChochMLModel] = None,
    lookahead: int = 20,
    swing_len: int = 5,
    atr_len: int = 14,
    vol_lookback: int = 50,
    min_score: float = 60.0,
    n_trees: int = 100,
    min_events: int = 10,
    window: int = 1500,
    scalar: float = 0.5,
    model_path: Optional[str] = None,
) -> tuple:
    """Run the full ML CHoCH signal pipeline on OHLCV data.

    Pipeline:
    1. Detect CHoCH events from swing structure breaks
    2. Extract ML features for each event
    3. Label historical events (using lookahead)
    4. For each event in chronological order:
       a. Train a fresh RF on events within the rolling window BEFORE this bar
       b. Predict probability and compute targets
       c. Add labeled event to rolling database for future predictions

    Args:
        df: DataFrame with columns [open, high, low, close, volume]
        model: Pre-loaded ChochMLModel (or None to create new one)
        lookahead: Bars to look ahead for training labels
        swing_len: Pivot detection period
        atr_len: ATR period for displacement normalization
        vol_lookback: Bars for volume delta averaging
        min_score: Minimum probability % for a valid signal
        n_trees: Number of trees in the Random Forest
        min_events: Minimum events needed before model can predict
        window: Rolling window size in bars (training memory)
        scalar: Conservative scalar for TP1
        model_path: If provided, load/save model from this path

    Returns:
        Tuple of (signals: List[SignalResult], model: ChochMLModel)
    """
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    n = len(close)

    # Initialize model
    if model is None:
        model = ChochMLModel(n_trees=n_trees, window=window, min_events=min_events)
        if model_path and Path(model_path).exists():
            model.load(model_path)

    # Step 1: Detect CHoCH events
    structure = detect_choch(high, low, close, swing_len)
    is_bullish = structure["is_bullish"]
    is_bearish = structure["is_bearish"]

    # Step 2: Extract features for ALL CHoCH events
    choch_mask = is_bullish | is_bearish
    choch_dirs = is_bullish  # True where bullish

    all_events = extract_features(df, choch_mask, choch_dirs, atr_len, vol_lookback)

    # Step 3: Label historical events (training data)
    labeled_events = label_events(df, all_events, lookahead)

    # Step 4: Rolling window prediction -- train fresh model at each event
    signals = _generate_rolling_signals(
        df, labeled_events, n_trees, min_events, window,
        lookahead, min_score, scalar,
    )

    # Save model if path provided
    if model_path:
        model.save(model_path)

    return signals, model


def _generate_rolling_signals(
    df: pd.DataFrame,
    labeled_events: List[ChochEvent],
    n_trees: int,
    min_events: int,
    window: int,
    lookahead: int,
    min_score: float,
    scalar: float,
) -> List[SignalResult]:
    """Generate signals using rolling window retraining.

    For each CHoCH event in chronological order:
    1. Build a fresh Random Forest from events within [event.bar_index - window, event.bar_index)
    2. Predict success probability for the current event
    3. Compute TP targets from similar successful neighbors
    4. Add this event (with label) to the rolling database for future predictions

    This ensures the model never sees future data during training -- faithful
    to real-time prediction conditions.

    Args:
        df: OHLCV DataFrame
        labeled_events: Events with outcome labels
        n_trees: Number of trees in the Random Forest
        min_events: Minimum events before model can predict
        window: Rolling window size in bars
        lookahead: Lookahead bars (for reference, events are already labeled)
        min_score: Minimum probability threshold
        scalar: Conservative scalar for TP1

    Returns:
        List of SignalResult sorted by bar_index
    """
    close = df["close"].values.astype(np.float64)
    n = len(close)

    signals: List[SignalResult] = []

    # Rolling database -- events accumulate as we move forward in time
    rolling_db: List[ChochEvent] = []

    # Process events in chronological order (they're sorted by bar_index)
    for event in labeled_events:
        idx = event.bar_index

        # Only predict for events that could have been labeled (enough future data)
        # Events too close to the end have outcome_value = 0 (unlabeled)
        # We still predict for them, but use only labeled events for training

        # Build a fresh model from events within the rolling window BEFORE this bar
        window_events = [
            e for e in rolling_db
            if e.bar_index + window > idx  # within window
        ]

        # Predict if we have enough training data
        if len(window_events) >= min_events:
            # Build fresh RF for this prediction
            temp_model = ChochMLModel(
                n_trees=n_trees, window=window, min_events=min_events
            )
            temp_model.fit(window_events)

            features = event.feature_vector()
            probability, similar_events = temp_model.predict(features, event.is_bullish)

            # Compute targets from similar successful neighbors
            current_price = close[idx]
            tp1, tp2, tp3 = temp_model.compute_targets(
                current_price, similar_events, event.is_bullish, scalar
            )

            is_valid = probability >= min_score
            score = probability if is_valid else 0.0

            signals.append(SignalResult(
                bar_index=idx,
                direction=event.is_bullish,
                probability=probability,
                score=score,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                is_valid=is_valid,
                db_size=len(window_events),
            ))
        else:
            # Not enough data yet -- record a low-confidence placeholder
            signals.append(SignalResult(
                bar_index=idx,
                direction=event.is_bullish,
                probability=50.0,
                score=0.0,
                tp1=0.0, tp2=0.0, tp3=0.0,
                is_valid=False,
                db_size=len(window_events),
            ))

        # Add this event to the rolling database for future predictions
        rolling_db.append(event)

    return signals
