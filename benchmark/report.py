"""Report generation for the unified benchmark.

Per-model outputs (benchmark/reports/<model>/):
    report.json      - full machine-readable results
    report.txt       - human-readable summary
    predictions.csv  - one row per prediction
    benchmark.log    - run log (DEBUG level)

Aggregate outputs (benchmark/reports/comparison/):
    comparison.json  - matrix of headline metrics per model
    comparison.csv   - same matrix, CSV
    comparison.txt   - same matrix, aligned text table

The comparison is built purely from the per-model report.json files, so any
future model that plugs in through an adapter gets compared automatically.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.adapters.base import BaseAdapter
from benchmark.config import BenchmarkConfig
from benchmark.evaluator import EvalRecord
from benchmark.metrics import (
    compute_classification,
    compute_regression,
    compute_runtime,
)

log = logging.getLogger("benchmark.report")

BENCHMARK_VERSION = "0.1.0"

_LOG_FORMAT = "%(asctime)s [%(levelname)-5s] %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

# (metric key inside report.json, human label, cell format) for comparison rows
_COMPARISON_ROWS = [
    ("direction_accuracy_pct", "Direction Accuracy", "{:.2f}%"),
    ("buy_precision_pct", "BUY Precision", "{:.2f}%"),
    ("buy_recall_pct", "BUY Recall", "{:.2f}%"),
    ("buy_f1", "BUY F1", "{:.2f}"),
    ("sell_precision_pct", "SELL Precision", "{:.2f}%"),
    ("sell_recall_pct", "SELL Recall", "{:.2f}%"),
    ("sell_f1", "SELL F1", "{:.2f}"),
    ("mae", "MAE", "{:.2f}%"),
    ("rmse", "RMSE", "{:.2f}%"),
    ("mape", "MAPE", "{:.2f}%"),
    ("avg_inference_ms", "Avg Inference Time", "{:.2f} ms"),
    ("load_time_s", "Model Load Time", "{:.2f} s"),
    ("peak_rss_mb", "Peak RAM", "{:.2f} MB"),
    ("model_size_mb", "Model Size", "{:.2f} MB"),
]


def _records_to_df(records: list[EvalRecord]) -> pd.DataFrame:
    """Convert evaluation records to the predictions.csv dataframe."""
    return pd.DataFrame(
        [
            {
                "symbol": r.symbol,
                "timestamp": r.timestamp.isoformat(),
                "last_close": r.last_close,
                "predicted_return": r.predicted_return,
                "actual_return": r.actual_return,
                "predicted_direction": r.predicted_direction,
                "actual_direction": r.actual_direction,
                "confidence": r.confidence,
                "inference_ms": r.inference_ms,
                "baseline_prev_candle": r.baseline_prev_candle,
                "baseline_random": r.baseline_random,
                "abs_error": r.abs_error,
            }
            for r in records
        ]
    )


def _baseline_accuracy(records: list[EvalRecord], attr: str) -> float:
    """Accuracy of a naive baseline vs realized direction (zeros skipped)."""
    hits = 0
    total = 0
    for rec in records:
        baseline = int(getattr(rec, attr))
        if baseline == 0:
            continue
        total += 1
        if baseline == rec.actual_direction:
            hits += 1
    return round(hits / total * 100, 2) if total else 0.0


def setup_model_logging(out_dir: Path, model_name: str) -> Path:
    """Create the model report dir and attach its benchmark.log to root.

    Any benchmark file handler from a previous model in the same process is
    removed first so successive models in an "all" run do not interleave logs.
    Returns the created model report directory.
    """
    model_dir = Path(out_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        if getattr(handler, "_benchmark_file_handler", False):
            root.removeHandler(handler)
            handler.close()
    file_handler = logging.FileHandler(
        model_dir / "benchmark.log", mode="w", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))
    file_handler._benchmark_file_handler = True
    root.addHandler(file_handler)
    return model_dir


def write_report(
    cfg: BenchmarkConfig,
    adapter: BaseAdapter,
    eval_result: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Write report.json / report.txt / predictions.csv for one model.

    Args:
        cfg: the shared benchmark configuration.
        adapter: the evaluated model adapter (already loaded).
        eval_result: the dict returned by Evaluator.run().
        model_dir: report directory (benchmark/reports/<model>).

    Returns:
        The report dict that was written to report.json.
    """
    records: list[EvalRecord] = eval_result["records"]
    inference_times: list[float] = eval_result["inference_times"]

    classification = compute_classification(
        [r.predicted_direction for r in records],
        [r.actual_direction for r in records],
    )
    regression = compute_regression(
        [r.predicted_return for r in records],
        [r.actual_return for r in records],
    )
    runtime = compute_runtime(adapter.load_time_s, inference_times)
    baselines = {
        "prev_candle_accuracy_pct": _baseline_accuracy(
            records, "baseline_prev_candle"
        ),
        "random_accuracy_pct": _baseline_accuracy(records, "baseline_random"),
    }

    report: dict[str, Any] = {
        "benchmark_version": BENCHMARK_VERSION,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": cfg.to_dict(),
        "adapter": adapter.describe(),
        "classification": classification,
        "regression": regression,
        "runtime": runtime,
        "baselines": baselines,
        "per_symbol": eval_result["per_symbol"],
        "peak_rss_mb": eval_result["peak_rss_mb"],
        "before_load_rss_mb": eval_result["before_load_rss_mb"],
    }

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (model_dir / "report.txt").write_text(_render_text(report), encoding="utf-8")
    _records_to_df(records).to_csv(model_dir / "predictions.csv", index=False)
    log.info("wrote %s report to %s", adapter.name, model_dir)
    return report


def _render_text(report: dict[str, Any]) -> str:
    """Render a human-readable report.txt from the report dict."""
    cfg = report["config"]
    adapter = report["adapter"]
    cls = report["classification"]
    reg = report["regression"]
    rt = report["runtime"]
    base = report["baselines"]
    lines = [
        "=" * 62,
        f"OpenAlgo Unified Benchmark - {adapter['name']}",
        "=" * 62,
        "",
        "CONFIG",
        f"  symbols           : {', '.join(cfg['symbols'])}",
        f"  exchange          : {cfg['exchange']}",
        f"  interval          : {cfg['interval']}",
        f"  start_date        : {cfg['start_date'] or '-'}",
        f"  end_date          : {cfg['end_date'] or '-'}",
        f"  horizon           : {cfg['horizon']}",
        f"  window_size       : {cfg['window_size']}",
        f"  step              : {cfg['step']}",
        f"  profile           : {cfg['profile']}",
        "",
        "MODEL",
        f"  name              : {adapter['name']}",
        f"  horizon           : {adapter['horizon']}",
        f"  load_time_s       : {adapter['load_time_s']}",
        f"  param_count       : {adapter['param_count']:,}",
        f"  model_size_mb     : {adapter['model_size_mb']}",
        "",
        "RESOURCES",
        f"  before_load_rss_mb: {report['before_load_rss_mb']}",
        f"  peak_rss_mb       : {report['peak_rss_mb']}",
        "",
        "CLASSIFICATION",
        f"  direction_accuracy: {cls['direction_accuracy_pct']:.2f}%",
        f"  buy  precision    : {cls['buy_precision_pct']:.2f}%  "
        f"recall: {cls['buy_recall_pct']:.2f}%  f1: {cls['buy_f1']:.2f}",
        f"  sell precision    : {cls['sell_precision_pct']:.2f}%  "
        f"recall: {cls['sell_recall_pct']:.2f}%  f1: {cls['sell_f1']:.2f}",
        f"  confusion tp/fp/fn/tn: "
        f"{cls['confusion']['tp']}/{cls['confusion']['fp']}/"
        f"{cls['confusion']['fn']}/{cls['confusion']['tn']}",
        f"  n_predictions     : {cls['n_predictions']}",
        "",
        "REGRESSION (fractions shown as %)",
        f"  MAE               : {reg['mae'] * 100:.2f}%",
        f"  RMSE              : {reg['rmse'] * 100:.2f}%",
        f"  MAPE              : {reg['mape'] * 100:.2f}%",
        "",
        "RUNTIME",
        f"  avg inference ms  : {rt['avg_inference_ms']:.2f}",
        f"  min inference ms  : {rt['min_inference_ms']:.2f}",
        f"  max inference ms  : {rt['max_inference_ms']:.2f}",
        f"  predictions/min   : {rt['predictions_per_min']:.1f}",
        f"  n_predictions     : {rt['n_predictions']}",
        "",
        "BASELINES (direction accuracy, zero baselines skipped)",
        f"  prev candle       : {base['prev_candle_accuracy_pct']:.2f}%",
        f"  random (seed 42)  : {base['random_accuracy_pct']:.2f}%",
        "",
        "PER SYMBOL",
    ]
    for symbol, info in sorted(report["per_symbol"].items()):
        lines.append(
            f"  {symbol:12s}: {info['predictions']} predictions, "
            f"{info['skipped']} skipped"
        )
    lines.append("")
    return "\n".join(lines)


def _metric_from_report(report: dict[str, Any], key: str) -> float:
    """Resolve a comparison metric key inside a model report.json."""
    if key in {
        "direction_accuracy_pct",
        "buy_precision_pct",
        "buy_recall_pct",
        "buy_f1",
        "sell_precision_pct",
        "sell_recall_pct",
        "sell_f1",
    }:
        return float(report["classification"][key])
    if key in {"mae", "rmse", "mape"}:
        # stored as fractions; comparison shows percentages
        return float(report["regression"][key]) * 100
    if key == "avg_inference_ms":
        return float(report["runtime"]["avg_inference_ms"])
    if key == "load_time_s":
        return float(report["runtime"]["load_time_s"])
    if key == "peak_rss_mb":
        return float(report["peak_rss_mb"])
    if key == "model_size_mb":
        return float(report["adapter"]["model_size_mb"])
    raise KeyError(f"unknown comparison metric key: {key}")


def _render_table(rows: list[list[str]]) -> str:
    """Render an aligned text table (first row = header)."""
    label_width = max(len(row[0]) for row in rows)
    value_widths = [
        max(len(row[col]) for row in rows) for col in range(1, len(rows[0]))
    ]
    width = label_width + sum(value_widths) + 2 * len(value_widths)
    lines = [
        "=" * width,
        "OpenAlgo Unified Benchmark - Model Comparison",
        "=" * width,
    ]
    for index, row in enumerate(rows):
        parts = [row[0].ljust(label_width)]
        parts.extend(
            row[col].rjust(value_widths[col - 1]) for col in range(1, len(row))
        )
        lines.append("  ".join(parts))
        if index == 0:
            lines.append("-" * width)
    lines.append("")
    return "\n".join(lines)


def build_comparison(model_names: list[str], out_dir: Path) -> dict[str, Any] | None:
    """Build comparison.json / comparison.csv / comparison.txt.

    Reads the per-model report.json files under out_dir/<model>/; models
    without a report are skipped. Returns the comparison matrix dict, or
    None when no model report is available.
    """
    reports: dict[str, dict[str, Any]] = {}
    for name in model_names:
        path = Path(out_dir) / name / "report.json"
        if path.exists():
            reports[name] = json.loads(path.read_text(encoding="utf-8"))
        else:
            log.warning(
                "no report.json for %s at %s; skipping in comparison",
                name,
                path,
            )
    if not reports:
        log.warning("build_comparison: no model reports under %s", out_dir)
        return None

    names = sorted(reports)
    matrix: dict[str, Any] = {"models": names, "metrics": {}}
    rows: list[list[str]] = [["Metric"] + names]
    for key, label, fmt in _COMPARISON_ROWS:
        values = [_metric_from_report(reports[name], key) for name in names]
        matrix["metrics"][label] = dict(zip(names, values, strict=True))
        rows.append([label] + [fmt.format(value) for value in values])

    comparison_dir = Path(out_dir) / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    (comparison_dir / "comparison.json").write_text(
        json.dumps(matrix, indent=2) + "\n", encoding="utf-8"
    )
    (comparison_dir / "comparison.txt").write_text(
        _render_table(rows), encoding="utf-8"
    )
    with open(
        comparison_dir / "comparison.csv", "w", newline="", encoding="utf-8"
    ) as fh:
        csv.writer(fh).writerows(rows)
    log.info("wrote comparison for %s to %s", ", ".join(names), comparison_dir)
    return matrix
