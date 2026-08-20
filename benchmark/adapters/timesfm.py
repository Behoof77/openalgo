"""TimesFM adapter for the unified benchmark.

Wraps the TimesFM server ModelManager read-only: no modification, no
retraining, no hyperparameter tuning.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pandas as pd

# --- environment wiring (must precede timesfm_server imports) ----------------
sys.path.insert(0, str(Path.cwd()))

from benchmark.adapters.base import BaseAdapter, ModelPrediction, _dir_size_mb  # noqa: E402
from timesfm_server.config import TimesFMConfig  # noqa: E402
from timesfm_server.model_manager import ModelManager  # noqa: E402


class TimesFMAdapter(BaseAdapter):
    """Benchmark adapter around the TimesFM server ModelManager."""

    name = "timesfm"

    def __init__(self, horizon: int = 5, profile: str = "fast") -> None:
        self.horizon = horizon
        self.profile = profile
        self._mgr: ModelManager | None = None

    def load(self) -> None:
        cfg = TimesFMConfig()
        cfg.inference_profile = self.profile  # must be set before ModelManager load
        self._mgr = ModelManager(cfg)
        t0 = time.perf_counter()
        self._mgr.load()
        self.load_time_s = round(time.perf_counter() - t0, 3)
        num_params = getattr(self._mgr.model, "num_parameters", None)
        if callable(num_params):
            self.param_count = int(cast(Callable[[], int], num_params)())
        else:
            # The compiled TimesFM_2p5_200M_torch wrapper exposes the real
            # torch module under .model (TimesFM_2p5_200M_torch_module).
            torch_module = getattr(self._mgr.model, "model", None)
            param_gen = getattr(torch_module, "parameters", None)
            if callable(param_gen):
                params = cast(Callable[[], list[Any]], param_gen)()
                self.param_count = sum(int(p.numel()) for p in params)
            else:
                self.param_count = 0
        if cfg.model_path.strip():
            self.model_dir = Path(cfg.model_path)
        elif cfg.model_cache_dir.strip():
            self.model_dir = Path(cfg.model_cache_dir)
        else:
            self.model_dir = None
        self.model_size_mb = _dir_size_mb(self.model_dir)

    def predict(self, df: pd.DataFrame) -> ModelPrediction:
        if self._mgr is None:
            raise RuntimeError("timesfm adapter not loaded - call load() first")
        t0 = time.perf_counter()
        result = self._mgr.forecast(df=df, horizon=self.horizon)
        inference_ms = round((time.perf_counter() - t0) * 1000.0, 1)
        raw = list(result.point_forecast[: self.horizon])
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
