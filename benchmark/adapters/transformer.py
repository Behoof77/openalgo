"""Transformer adapter for the unified benchmark.

Wraps the tiiip transformer-server ModelManager and analyze_stock
read-only. The model's native predict horizon is fixed at 5 days, so the
adapter refuses any other horizon.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

# --- environment wiring (must precede transformer_server imports) ------------
sys.path.insert(0, str(Path.cwd()))

from benchmark.adapters.base import BaseAdapter, ModelPrediction, _dir_size_mb  # noqa: E402
from transformer_server.inference import analyze_stock  # noqa: E402
from transformer_server.model_manager import ModelManager  # noqa: E402

_FIXED_HORIZON = 5


class TransformerAdapter(BaseAdapter):
    """Benchmark adapter around the transformer-server ModelManager."""

    name = "transformer"

    def __init__(self, horizon: int = _FIXED_HORIZON, profile: str = "fast") -> None:
        if horizon != _FIXED_HORIZON:
            raise ValueError(
                f"transformer model is trained with a fixed {_FIXED_HORIZON}-day "
                f"horizon; horizon={horizon} is not supported"
            )
        self.horizon = _FIXED_HORIZON
        self.profile = profile
        self._mgr: ModelManager | None = None

    def load(self) -> None:
        t0 = time.perf_counter()
        # Constructor loads the checkpoint and scaler immediately.
        mgr = ModelManager()
        self._mgr = mgr
        self.load_time_s = round(time.perf_counter() - t0, 3)
        self.param_count = int(sum(p.numel() for p in mgr.model.parameters()))
        self.model_dir = Path("transformer-server") / "models"
        self.model_size_mb = _dir_size_mb(self.model_dir)

    def predict(self, df: pd.DataFrame) -> ModelPrediction:
        if self._mgr is None:
            raise RuntimeError("transformer adapter not loaded - call load() first")
        t0 = time.perf_counter()
        result = analyze_stock(
            self._mgr.model,
            self._mgr.scaler,
            df,
            feature_cols=self._mgr.config.feature_cols,
            lookback=self._mgr.config.lookback,
            device=self._mgr.device,
        )
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        if result.empty:
            raise RuntimeError(
                "transformer produced no windows - the evaluation window must be "
                f"at least lookback ({self._mgr.config.lookback}) plus feature "
                "warmup rows"
            )
        last = result.iloc[-1]
        predicted_return = float(last["return_pred"])  # already a fraction
        direction = 1 if predicted_return > 0 else (-1 if predicted_return < 0 else 0)
        return ModelPrediction(
            predicted_return=predicted_return,
            predicted_direction=direction,
            confidence=float(last["confidence"]),
            inference_ms=inference_ms,
            horizon=self.horizon,
            raw=None,
            metadata={
                "signal": str(last["signal"]),
                "signal_index": int(last["signal_index"]),
            },
        )
