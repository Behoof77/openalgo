"""Benchmark run configuration.

Mirrors the env-driven dataclass convention of the sibling model servers
(kronos-server/config.py, timesfm-server/config.py) so every run is fully
reproducible from a single env/CLI snapshot.

Env prefix: BENCHMARK_  (OPENALGO_HOST / OPENALGO_API_KEY are read from the
project .env for data fetching, matching validate_15m.py).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

VALID_MODELS = ("kronos", "timesfm", "transformer", "all")
VALID_INTERVALS = ("1m", "3m", "5m", "10m", "15m", "30m", "1h", "D")


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_symbols(raw: str) -> list[str]:
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


@dataclass
class BenchmarkConfig:
    """Single configuration shared verbatim across all benchmarked models."""

    # Which model(s) to run: kronos | timesfm | transformer | all
    model: str = "all"

    # Data window (identical for every model in a run)
    symbols: list[str] = field(
        default_factory=lambda: _parse_symbols(_env_str("BENCHMARK_SYMBOLS", "RELIANCE"))
    )
    exchange: str = "NSE"
    interval: str = "D"
    start_date: str | None = None  # YYYY-MM-DD; None = earliest available
    end_date: str | None = None  # YYYY-MM-DD; None = today

    # Rolling-window evaluation methodology (same for every model)
    horizon: int = 5  # forward bars used to label each prediction
    window_size: int = 512  # trailing bars fed to the model per prediction
    step: int = 1  # bar offset between successive evaluation points
    max_pred: int | None = None  # cap on predictions (None = all available)

    # Inference profile passed to the model servers (fast | normal | accurate)
    profile: str = "fast"

    # Data source
    openalgo_host: str = _env_str("OPENALGO_HOST", "http://127.0.0.1:5000")
    openalgo_api_key: str = _env_str("OPENALGO_API_KEY", "")

    # Outputs
    out_dir: str = "benchmark/reports"

    # Baselines
    seed: int = 42

    def __post_init__(self) -> None:
        if self.model not in VALID_MODELS:
            raise ValueError(
                f"model must be one of {VALID_MODELS}, got {self.model!r}"
            )
        if self.interval not in VALID_INTERVALS:
            raise ValueError(
                f"interval must be one of {VALID_INTERVALS}, got {self.interval!r}"
            )
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")
        if self.window_size < 30:
            raise ValueError(
                f"window_size must be >= 30 (transformer needs 30 rows), "
                f"got {self.window_size}"
            )
        if self.step < 1:
            raise ValueError(f"step must be >= 1, got {self.step}")
        if self.profile not in ("fast", "normal", "accurate"):
            raise ValueError(
                "profile must be one of fast|normal|accurate, "
                f"got {self.profile!r}"
            )

    @classmethod
    def from_env(cls) -> BenchmarkConfig:
        """Build a config from the environment, overriding the defaults."""
        return cls(
            model=_env_str("BENCHMARK_MODEL", "all"),
            symbols=_parse_symbols(_env_str("BENCHMARK_SYMBOLS", "RELIANCE")),
            exchange=_env_str("BENCHMARK_EXCHANGE", "NSE"),
            interval=_env_str("BENCHMARK_INTERVAL", "D"),
            start_date=_env_str("BENCHMARK_START", "") or None,
            end_date=_env_str("BENCHMARK_END", "") or None,
            horizon=_env_int("BENCHMARK_HORIZON", 5),
            window_size=_env_int("BENCHMARK_WINDOW", 512),
            step=_env_int("BENCHMARK_STEP", 1),
            max_pred=_env_int("BENCHMARK_MAX_PRED", 0) or None,
            profile=_env_str("BENCHMARK_PROFILE", "fast"),
            out_dir=_env_str("BENCHMARK_OUT", "benchmark/reports"),
            seed=_env_int("BENCHMARK_SEED", 42),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializable snapshot used in report.json (config section)."""
        return {
            "model": self.model,
            "symbols": list(self.symbols),
            "exchange": self.exchange,
            "interval": self.interval,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "horizon": self.horizon,
            "window_size": self.window_size,
            "step": self.step,
            "max_pred": self.max_pred,
            "profile": self.profile,
            "openalgo_host": self.openalgo_host,
            "seed": self.seed,
        }

    def out_path(self) -> Path:
        return Path(self.out_dir)

    def __repr__(self) -> str:
        pairs = [f"{f.name}={getattr(self, f.name)!r}" for f in fields(self)]
        return "BenchmarkConfig(" + ", ".join(pairs) + ")"


def print_config() -> None:
    """CLI helper: `python -m benchmark.config`."""
    cfg = BenchmarkConfig.from_env()
    print(cfg)


if __name__ == "__main__":
    print_config()
