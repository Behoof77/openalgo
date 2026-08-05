"""Unified benchmark CLI.

Runs one or more models over identical data windows and methodology:

    uv run python benchmark/benchmark.py --model all
    uv run python benchmark/benchmark.py --model kronos --symbols RELIANCE,INFY

Every model is evaluated through its adapter (same symbols, exchange,
interval, dates, window, horizon and rolling scheme). Results land in
benchmark/reports/<model>/ and a comparison of all executed models is built
in benchmark/reports/comparison/.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import logging
import sys
from pathlib import Path
from typing import Any

from benchmark.adapters.base import BaseAdapter
from benchmark.config import BenchmarkConfig

_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

_VALID_MODELS = ("kronos", "timesfm", "transformer", "all")
_VALID_INTERVALS = ("1m", "3m", "5m", "10m", "15m", "30m", "1h", "D")

# Noisy third-party loggers silenced during a run
_NOISY_LOGGERS = ("httpx", "model_manager", "timesfm")

# Adapter class name per model (timesfm -> TimesFMAdapter, not TimesfmAdapter)
_ADAPTER_CLASSES = {
    "kronos": "KronosAdapter",
    "timesfm": "TimesFMAdapter",
    "transformer": "TransformerAdapter",
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description=(
            "OpenAlgo unified model benchmark (Kronos, TimesFM, Transformer). "
            "Every model gets the same data window, horizon and rolling scheme."
        ),
    )
    parser.add_argument(
        "--model",
        choices=_VALID_MODELS,
        default="all",
        help="model to benchmark, or all (default: all)",
    )
    parser.add_argument(
        "--symbols",
        default="RELIANCE",
        help="comma-separated symbols (default: RELIANCE)",
    )
    parser.add_argument("--exchange", default="NSE", help="exchange (default: NSE)")
    parser.add_argument(
        "--interval",
        choices=_VALID_INTERVALS,
        default="D",
        help="candle interval (default: D)",
    )
    parser.add_argument(
        "--start",
        dest="start_date",
        default=None,
        help="start date YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--end",
        dest="end_date",
        default=None,
        help="end date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="prediction horizon in bars (default: 5; transformer is fixed at 5)",
    )
    parser.add_argument(
        "--window",
        dest="window_size",
        type=int,
        default=512,
        help="trailing window rows fed per prediction (default: 512)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="bar offset between evaluation points (default: 1)",
    )
    parser.add_argument(
        "--max-pred",
        dest="max_pred",
        type=int,
        default=None,
        help="cap on predictions per symbol (default: all available)",
    )
    parser.add_argument(
        "--profile",
        choices=("fast", "normal", "accurate"),
        default="fast",
        help="inference profile (default: fast)",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        default="benchmark/reports",
        help="output directory (default: benchmark/reports)",
    )
    return parser.parse_args(argv)


def _cfg_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    """Build a BenchmarkConfig from CLI args (CLI wins over env)."""
    return BenchmarkConfig(
        model=args.model,
        symbols=[s.strip().upper() for s in args.symbols.split(",") if s.strip()],
        exchange=args.exchange.upper(),
        interval=args.interval,
        start_date=args.start_date,
        end_date=args.end_date,
        horizon=args.horizon,
        window_size=args.window_size,
        step=args.step,
        max_pred=args.max_pred,
        profile=args.profile,
        out_dir=args.out_dir,
    )


def _get_adapter(model: str, cfg: BenchmarkConfig) -> BaseAdapter:
    """Lazily import and construct the adapter for a model name.

    The adapter module is imported only when its model is about to run, so a
    single-model run never pays torch import cost for the other models.
    """
    module = importlib.import_module(f"benchmark.adapters.{model}")
    factory = getattr(module, _ADAPTER_CLASSES[model])
    kwargs: dict[str, Any] = {"horizon": cfg.horizon, "profile": cfg.profile}
    if model == "kronos":
        kwargs["interval"] = cfg.interval
    return factory(**kwargs)


def run(argv: list[str] | None = None) -> int:
    """Run the benchmark for the requested model(s) and build a comparison."""
    args = _parse_args(argv)
    cfg = _cfg_from_args(args)

    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT, datefmt=_LOG_DATEFMT)
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    log = logging.getLogger("benchmark")
    log.info("benchmark start: model=%s symbols=%s %s/%s", cfg.model, cfg.symbols, cfg.exchange, cfg.interval)

    from benchmark.evaluator import Evaluator
    from benchmark.report import build_comparison, setup_model_logging, write_report

    models = [m for m in _VALID_MODELS if m != "all"] if cfg.model == "all" else [cfg.model]

    results: dict[str, dict[str, Any]] = {}
    for model in models:
        model_cfg = dataclasses.replace(cfg, model=model)
        model_dir = setup_model_logging(Path(model_cfg.out_dir), model)
        log.info("=== benchmarking model: %s ===", model)
        try:
            adapter = _get_adapter(model, model_cfg)
            evaluator = Evaluator(model_cfg, adapter)
            eval_result = evaluator.run()
        except Exception:
            log.exception("benchmark failed for model %s", model)
            continue
        results[model] = write_report(model_cfg, adapter, eval_result, model_dir)

    if not results:
        log.error("no model produced results; nothing written")
        return 1

    comparison = build_comparison(list(results), Path(cfg.out_dir))
    if comparison is not None:
        comparison_txt = Path(cfg.out_dir) / "comparison" / "comparison.txt"
        if comparison_txt.exists():
            print()
            print(comparison_txt.read_text(encoding="utf-8"))
    log.info("benchmark complete: %s", ", ".join(sorted(results)))
    return 0


if __name__ == "__main__":
    sys.exit(run())
