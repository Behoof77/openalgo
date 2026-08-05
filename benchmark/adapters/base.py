"""Base adapter interface for the unified benchmark.

Every model under evaluation (Kronos, TimesFM, the tiiip Transformer, and
any future model including XGBoost variants) plugs in through a BaseAdapter
so the evaluator treats all models identically: same data window, same
horizon, same rolling methodology, same metrics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import psutil


@dataclass
class ModelPrediction:
    """Normalized output of a single model inference call."""

    predicted_return: float  # simple return over the horizon, as a FRACTION
    predicted_direction: int  # 1 = up, -1 = down, 0 = flat (dead-band)
    confidence: float  # 0.0 .. 1.0
    inference_ms: float
    horizon: int
    raw: list[float] | None = None  # model-native raw forecast, if available
    metadata: dict[str, Any] = field(default_factory=dict)


def _dir_size_mb(path: Path | None) -> float:
    """Total on-disk size of a model directory (or single file) in MB."""
    if path is None or not path.exists():
        return 0.0
    files = path.rglob("*") if path.is_dir() else [path]
    total = sum(f.stat().st_size for f in files if f.is_file())
    return round(total / (1024 * 1024), 2)


class BaseAdapter(ABC):
    """Interface every benchmarked model must implement."""

    name: str = "base"
    horizon: int = 5

    # Populated by load()
    load_time_s: float = 0.0
    param_count: int = 0
    model_size_mb: float = 0.0
    model_dir: Path | None = None

    @abstractmethod
    def load(self) -> None:
        """Load the model into memory and record load_time_s / model_size_mb.

        Must be safe to call once; the evaluator calls it exactly once
        before the prediction loop.
        """

    @abstractmethod
    def predict(self, df: pd.DataFrame) -> ModelPrediction:
        """Produce one prediction from a trailing window of OHLCV data.

        Args:
            df: a pandas DataFrame with at least open/high/low/close columns
                (plus volume where available), sorted by timestamp ascending.
                The last row is the most recent bar. The adapter applies its
                own native context slicing internally.

        Returns:
            ModelPrediction with predicted_return as a fraction over the
            adapter's horizon.
        """

    @staticmethod
    def memory_usage_mb() -> float:
        """Current process RSS in MB (called repeatedly by the evaluator)."""
        return round(psutil.Process().memory_info().rss / (1024 * 1024), 1)

    def describe(self) -> dict[str, Any]:
        """Resource summary recorded into report.json by the evaluator."""
        return {
            "name": self.name,
            "horizon": self.horizon,
            "load_time_s": round(self.load_time_s, 3),
            "param_count": self.param_count,
            "model_size_mb": self.model_size_mb,
            "model_dir": str(self.model_dir) if self.model_dir else None,
        }
