"""Rolling-window evaluation harness for the unified benchmark.

Fetches OHLCV through the OpenAlgo SDK, slices a uniform trailing window
ending at each evaluation index, asks the adapter for a prediction, and
labels it with the realized forward return over the same horizon. All
returns are simple fractional returns.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from benchmark.adapters.base import BaseAdapter
from benchmark.config import BenchmarkConfig
from benchmark.metrics import compute_baselines

log = logging.getLogger("benchmark.evaluator")

_INTERVAL_LOOKBACK_DAYS: dict[str, int] = {
    "1m": 15,
    "3m": 30,
    "5m": 60,
    "10m": 90,
    "15m": 120,
    "30m": 180,
    "1h": 365,
    "D": 1095,
}


def _resolve_dates(
    start_date: str | None, end_date: str | None, interval: str
) -> tuple[str | None, str | None]:
    """Resolve missing date bounds to a trailing window ending today.

    The OpenAlgo history API rejects null dates, so any None bound is
    replaced with an explicit date: end defaults to today, start to a
    trailing lookback sized per interval.
    """
    if start_date is not None and end_date is not None:
        return start_date, end_date
    end = end_date or date.today().isoformat()
    lookback = _INTERVAL_LOOKBACK_DAYS.get(interval, 1095)
    start = start_date or (date.today() - timedelta(days=lookback)).isoformat()
    return start, end


@dataclass
class EvalRecord:
    symbol: str
    timestamp: pd.Timestamp
    last_close: float
    predicted_return: float
    actual_return: float
    predicted_direction: int
    actual_direction: int
    confidence: float
    inference_ms: float
    baseline_prev_candle: int
    baseline_random: int
    abs_error: float


def _strict_sign(value: float) -> int:
    return 1 if value > 0 else (-1 if value < 0 else 0)


def fetch_ohlcv(
    client: Any,
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame | None:
    """Fetch and normalize OHLCV history from the OpenAlgo SDK.

    Returns a DataFrame sorted by timestamp with columns
    [timestamp, open, high, low, close, volume] (volume when present), or
    None on failure.
    """
    try:
        df = client.history(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception:
        log.exception("history fetch failed for %s", symbol)
        return None
    if not isinstance(df, pd.DataFrame):
        log.error("history returned non-DataFrame for %s: %s", symbol, df)
        return None
    if df.empty:
        log.error("history returned empty for %s", symbol)
        return None

    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns:
            ts_col: Any = "index" if "index" in df.columns else df.columns[0]
            df = df.rename(columns={ts_col: "timestamp"})
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        max_val = float(ts.max())
        if max_val > 1e12:
            ts = pd.to_datetime(ts, unit="ns")
        elif max_val > 1e6:
            ts = pd.to_datetime(ts, unit="s")
        else:
            ts = pd.to_datetime(ts)
    else:
        ts = pd.to_datetime(ts)
    df["timestamp"] = ts

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("history for %s missing required columns: %s", symbol, missing)
        return None
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


class Evaluator:
    """Runs the shared rolling-window evaluation for one adapter."""

    def __init__(
        self, cfg: BenchmarkConfig, adapter: BaseAdapter, logger: logging.Logger | None = None
    ) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.log = logger or log

    def _make_client(self) -> Any:
        from openalgo import api

        return api(api_key=self.cfg.openalgo_api_key, host=self.cfg.openalgo_host)

    def run(self) -> dict:
        cfg = self.cfg
        before_rss = self.adapter.memory_usage_mb()
        self.log.info(
            "loading adapter %s (rss before load %.1f MB)", self.adapter.name, before_rss
        )
        self.adapter.load()
        peak_rss = self.adapter.memory_usage_mb()
        self.log.info(
            "adapter %s loaded in %.3f s, %d params, %.1f MB on disk, rss %.1f MB",
            self.adapter.name,
            self.adapter.load_time_s,
            self.adapter.param_count,
            self.adapter.model_size_mb,
            peak_rss,
        )

        client = self._make_client()
        start_date, end_date = _resolve_dates(cfg.start_date, cfg.end_date, cfg.interval)
        self.log.info(
            "data window: %s to %s (interval %s)", start_date, end_date, cfg.interval
        )
        records: list[EvalRecord] = []
        inference_times: list[float] = []
        per_symbol: dict[str, dict] = {}

        for symbol in cfg.symbols:
            df = fetch_ohlcv(
                client, symbol, cfg.exchange, cfg.interval, start_date, end_date
            )
            if df is None:
                continue
            if len(df) < cfg.window_size + cfg.horizon:
                self.log.warning(
                    "symbol %s too short (%d bars) for window=%d horizon=%d",
                    symbol, len(df), cfg.window_size, cfg.horizon,
                )
                continue

            prev_baseline, rnd_baseline = compute_baselines(df, cfg.seed)
            closes = df["close"].to_numpy(dtype=np.float64)
            max_end = len(df) - cfg.horizon
            end_idxs = list(range(cfg.window_size, max_end + 1, cfg.step))
            if cfg.max_pred is not None:
                end_idxs = end_idxs[: cfg.max_pred]

            self.log.info(
                "symbol %s: %d bars, %d predictions planned (window %d, step %d)",
                symbol, len(df), len(end_idxs), cfg.window_size, cfg.step,
            )
            done = 0
            skipped = 0
            for end_idx in end_idxs:
                window = df.iloc[end_idx - cfg.window_size : end_idx].copy()
                try:
                    pred = self.adapter.predict(window)
                except Exception as exc:
                    self.log.warning(
                        "prediction failed at index %d for %s: %s", end_idx, symbol, exc
                    )
                    skipped += 1
                    continue
                actual_return = (
                    closes[end_idx + cfg.horizon - 1] / closes[end_idx - 1] - 1.0
                )
                actual_direction = _strict_sign(actual_return)
                records.append(
                    EvalRecord(
                        symbol=symbol,
                        timestamp=df["timestamp"].iloc[end_idx - 1],
                        last_close=float(closes[end_idx - 1]),
                        predicted_return=pred.predicted_return,
                        actual_return=actual_return,
                        predicted_direction=pred.predicted_direction,
                        actual_direction=actual_direction,
                        confidence=pred.confidence,
                        inference_ms=pred.inference_ms,
                        baseline_prev_candle=int(prev_baseline[end_idx]),
                        baseline_random=int(rnd_baseline[end_idx]),
                        abs_error=abs(pred.predicted_return - actual_return),
                    )
                )
                inference_times.append(pred.inference_ms)
                peak_rss = max(peak_rss, self.adapter.memory_usage_mb())
                done += 1
                if done % 25 == 0:
                    recent = records[-25:]
                    acc = (
                        sum(1 for r in recent if r.predicted_direction == r.actual_direction)
                        / len(recent)
                        * 100.0
                    )
                    self.log.info(
                        "progress %d/%d, recent direction accuracy %.1f%%, peak rss %.1f MB",
                        done, len(end_idxs), acc, peak_rss,
                    )
            per_symbol[symbol] = {"predictions": done, "skipped": skipped}
            self.log.info("symbol %s done: %d predictions, %d skipped", symbol, done, skipped)

        if not records:
            raise RuntimeError("no predictions produced for any symbol")

        return {
            "records": records,
            "inference_times": inference_times,
            "peak_rss_mb": round(peak_rss, 1),
            "before_load_rss_mb": round(before_rss, 1),
            "per_symbol": per_symbol,
        }
