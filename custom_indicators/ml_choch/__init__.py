"""
ML CHoCH -- Machine Learning Change of Character Indicator

A Python ML implementation of CHoCH detection and prediction using
Random Forest classification. Detects market structure breaks (swing
high/low), extracts 3D features (volume delta, displacement, velocity),
and uses scikit-learn RandomForestClassifier with rolling window training
to predict signal probability and compute take-profit targets.

Training approach: At each CHoCH event, a fresh Random Forest is trained
on only the events within the rolling window that preceded it. No future
data leaks into training -- faithful to real-time conditions.

Usage:
    from custom_indicators.ml_choch import generate_signals, ChochMLModel

    signals, model = generate_signals(df, min_score=60)
    for sig in signals:
        if sig.is_valid:
            print(f"{'BULL' if sig.direction else 'BEAR'} "
                  f"prob={sig.probability:.1f}% "
                  f"TP1={sig.tp1:.2f} TP2={sig.tp2:.2f} TP3={sig.tp3:.2f}")
"""

from .choch_detector import detect_choch
from .choch_model import ChochMLModel
from .feature_engine import ChochEvent, extract_features, label_events
from .signal_generator import SignalResult, generate_signals

__all__ = [
    "detect_choch",
    "ChochEvent",
    "ChochMLModel",
    "extract_features",
    "label_events",
    "SignalResult",
    "generate_signals",
]
