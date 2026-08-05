#!/bin/bash
export PATH="$HOME/.local/bin:$PATH"
cd /home/ubuntu/openalgo
uv run python strategies/xgb_reversals_NIFTY/strategy.py --mode backtest >> strategies/xgb_reversals_NIFTY/backtest_run2.log 2>&1
