# ML Project Migration — 2026-08-06

## Summary

The machine-learning projects that lived inside the OpenAlgo repository were
extracted into dedicated repositories. OpenAlgo is a trading platform (broker
API, Python strategy host, Flow builder, options suite). ML research, training,
and serving now iterate independently in their own repos and are consumed via
APIs / MCP instead of living inside the trading codebase.

## What moved where

| Project | Removed from OpenAlgo | Now lives in | Extraction commit |
|---|---|---|---|
| Kronos time-series forecasting + benchmark harness | `kronos/`, `benchmark/`, `benchmark_api.py` | Kronos repo (`mine` remote, Behoof77/Kronos) | `e7e8dc7`, `c9aae1d` |
| Options transformer | `strategies/options-ml-transformer/` (untracked) | options-ml-transformer | `867c93b` |
| Options ML training | `strategies/options-ml-training/` (untracked) | options-ml-training | `d8ba67a` |
| Skopaq AI (feature engine, choch model, xgb reversals) | `feature_engine/`, `custom_indicators/ml_choch/`, `strategies/xgb_reversals_NIFTY/` | skopaq-ai | `1294beb` |

The five upstream commits above preserve the extracted code (including server
runners and datasets) in their new homes.

## What changed in OpenAlgo

Removed (tracked, staged in this commit):

- `kronos/` (26 files), `benchmark/` (20 files), `benchmark_api.py`
- `feature_engine/`, `custom_indicators/ml_choch/` (12 files),
  `strategies/xgb_reversals_NIFTY/` (5 files)
- `docs/feature-engine.md` (spec of the moved package)
- `test/test_feature_engine.py` (tested only the moved package)
- `run_backtest.sh` (ran the moved xgb_reversals strategy)

Removed from disk (untracked, never in git history):

- `strategies/options-ml-transformer/`, `strategies/options-ml-training/`
- Runtime artifacts: `cookie*.txt`, `*.pid`, `tmp_*.py`, `backtest_output/`,
  `outputs/`, `explore_historify_vm.py`, `.cors_check.py`

Updated:

- `pyproject.toml`, `pyrightconfig.json` — ruff exclude list cleaned of the
  removed packages
- `docs/INDEX.md` — dropped the feature-engine entry
- `docs/dataset-research.md` — feature_engine references now point to skopaq-ai
- `.gitignore` — added `cookie*.txt`, `*.pid`, `tmp_*.py`, `backtest_output/`,
  `outputs/`, `.env.bak-*`

## Why

- OpenAlgo stays focused on its four product surfaces; no ML imports,
  no ML documentation (except intentional references), no duplicate code.
- The ML projects release on their own cadence and are consumed through their
  own APIs / MCP.
- This commit is the single migration checkpoint: `git log` shows exactly which
  ML surface left and where it went.

## Migration commit

- This cleanup commit: `refactor: extract ML projects to dedicated repos` (this commit)
