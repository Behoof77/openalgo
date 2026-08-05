"""Kronos adapter for the unified benchmark.

Wraps the Kronos server ModelManager read-only: no modification, no
retraining, no hyperparameter tuning. Mirrors validate_15m's environment
wiring so the same local checkpoints are used.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd

# --- environment wiring (must precede kronos_server imports) -----------------
sys.path.insert(0, str(Path.cwd()))
_KRONOS_SRC = Path.home() / "Projects" / "Kronos"
if _KRONOS_SRC.exists():
    sys.path.insert(0, str(_KRONOS_SRC))
os.environ.setdefault(
    "KRONOS_MODEL_PATH", str(Path("kronos-server") / "models" / "Kronos-small")
)
os.environ.setdefault(
    "KRONOS_TOKENIZER_PATH",
    str(Path("kronos-server") / "models" / "Kronos-Tokenizer-base"),
)

from benchmark.adapters.base import BaseAdapter, ModelPrediction, _dir_size_mb  # noqa: E402
from kronos_server.config import KronosServerConfig  # noqa: E402
from kronos_server.model_manager import ModelManager  # noqa: E402

_INTERVAL_TO_FREQ = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "D": "1D",
}


class KronosAdapter(BaseAdapter):
    """Benchmark adapter around the Kronos server ModelManager."""

    name = "kronos"

    def __init__(self, horizon: int = 5, profile: str = "fast", interval: str = "D") -> None:
        if interval not in _INTERVAL_TO_FREQ:
            raise ValueError(f"unsupported interval for kronos: {interval}")
        self.horizon = horizon
        self.profile = profile
        self.interval = interval
        self._mgr: ModelManager | None = None

    def load(self) -> None:
        cfg = KronosServerConfig.from_env()
        cfg.inference_profile = self.profile
        self._mgr = ModelManager(cfg)
        t0 = time.perf_counter()
        self._mgr.load()
        self.load_time_s = round(time.perf_counter() - t0, 3)
        self.param_count = int(self._mgr.param_count)
        self.model_dir = Path("kronos-server") / "models" / "Kronos-small"
        self.model_size_mb = _dir_size_mb(self.model_dir)

    def predict(self, df: pd.DataFrame) -> ModelPrediction:
        if self._mgr is None:
            raise RuntimeError("kronos adapter not loaded - call load() first")
        x_ts = "timestamp" if "timestamp" in df.columns else None
        t0 = time.perf_counter()
        result = self._mgr.predict(
            df=df,
            x_timestamp=x_ts,
            freq=_INTERVAL_TO_FREQ[self.interval],
            profile=self.profile,
            pred_len=self.horizon,
        )
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        raw = list(result.raw_predictions[: self.horizon])
        last_close = float(df["close"].iloc[-1])
        predicted_return = raw[-1] / last_close - 1.0
        direction = 1 if predicted_return > 0 else (-1 if predicted_return < 0 else 0)
        return ModelPrediction(
            predicted_return=predicted_return,
            predicted_direction=direction,
            confidence=float(result.confidence),
            inference_ms=inference_ms,
            horizon=self.horizon,
            raw=raw,
            metadata={"model_signal": int(result.prediction)},
        )
