"""Metric computation for the unified benchmark.

Classification metrics treat direction 1 as the positive class (BUY) and
-1 as the negative class (SELL); flat (0) predictions/realizations are
excluded from per-class precision/recall but still count against accuracy.
Regression metrics are computed on fractional returns and reported as
percentages by the report layer.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0 else 0.0


def _round2(value: float) -> float:
    return round(value, 2)


def compute_classification(
    pred_dirs: list[int], actual_dirs: list[int]
) -> dict:
    """Direction accuracy, BUY/SELL precision/recall/F1, confusion counts."""
    n = len(pred_dirs)
    pairs = list(zip(pred_dirs, actual_dirs, strict=True))
    correct = sum(1 for p, a in pairs if p == a)
    tp = sum(1 for p, a in pairs if p == 1 and a == 1)
    fp = sum(1 for p, a in pairs if p == 1 and a != 1)
    fn = sum(1 for p, a in pairs if p != 1 and a == 1)
    tn = sum(1 for p, a in pairs if p == -1 and a == -1)

    buy_precision = _safe_div(tp, tp + fp) * 100.0
    buy_recall = _safe_div(tp, tp + fn) * 100.0
    buy_f1 = _safe_div(2 * buy_precision * buy_recall, buy_precision + buy_recall)
    sell_precision = _safe_div(tn, tn + fn) * 100.0
    sell_recall = _safe_div(tn, tn + fp) * 100.0
    sell_f1 = _safe_div(2 * sell_precision * sell_recall, sell_precision + sell_recall)

    return {
        "direction_accuracy_pct": _round2(correct / n * 100.0 if n else 0.0),
        "buy_precision_pct": _round2(buy_precision),
        "buy_recall_pct": _round2(buy_recall),
        "buy_f1": _round2(buy_f1),
        "sell_precision_pct": _round2(sell_precision),
        "sell_recall_pct": _round2(sell_recall),
        "sell_f1": _round2(sell_f1),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "n_predictions": n,
    }


def compute_regression(pred_returns: list[float], actual_returns: list[float]) -> dict:
    """MAE, RMSE, MAPE on fractional returns (kept as fractions, 4dp)."""
    n = len(pred_returns)
    if n == 0:
        return {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
    errs = [p - a for p, a in zip(pred_returns, actual_returns, strict=True)]
    mae = sum(abs(e) for e in errs) / n
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    mape_pts = [
        (p, a) for p, a in zip(pred_returns, actual_returns, strict=True) if a != 0
    ]
    if mape_pts:
        mape = sum(abs((p - a) / a) for p, a in mape_pts) / len(mape_pts)
    else:
        mape = 0.0
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(mape, 4),
    }


def compute_runtime(load_time_s: float, inference_ms: list[float]) -> dict:
    """Runtime statistics: load time, avg/min/max inference, throughput."""
    n = len(inference_ms)
    if n == 0:
        return {
            "load_time_s": round(load_time_s, 3),
            "avg_inference_ms": 0.0,
            "min_inference_ms": 0.0,
            "max_inference_ms": 0.0,
            "predictions_per_min": 0.0,
            "n_predictions": 0,
        }
    avg_ms = sum(inference_ms) / n
    return {
        "load_time_s": round(load_time_s, 3),
        "avg_inference_ms": round(avg_ms, 2),
        "min_inference_ms": round(min(inference_ms), 2),
        "max_inference_ms": round(max(inference_ms), 2),
        "predictions_per_min": round(60_000.0 / avg_ms, 1) if avg_ms > 0 else 0.0,
        "n_predictions": n,
    }


def compute_baselines(df: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Reference baselines aligned to df length (mirrors validate_15m).

    prev_candle: sign of close[i] - close[i-2] (first two rows zero).
    random: uniform +/-1 drawn with a fixed seed.
    """
    closes = df["close"].to_numpy(dtype=np.float64)
    n = len(closes)
    prev_candle = np.zeros(n, dtype=np.int8)
    diff = np.sign(closes[1:-1] - closes[:-2]).astype(np.int8)
    prev_candle[2:] = diff
    rng = random.Random(seed)
    rnd = np.asarray(rng.choices([-1, 1], k=n), dtype=np.int8)
    return prev_candle, rnd
