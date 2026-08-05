"""
ML CHoCH NIFTY Futures -- VectorBT + OpenStatz Tearsheet
========================================================
Runs the ML CHoCH signal pipeline with optimized parameters, executes the
faithful event-driven backtest (scale-out exits), then feeds the equity
curve to openstatz for a modern interactive tearsheet.

Also produces a VectorBT comparison table vs Buy & Hold benchmark.

Usage:
  uv run python custom_indicators/ml_choch/backtest_vectorbt.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openalgo import api

from custom_indicators.ml_choch.backtest_nifty import (
    fetch_nifty_data,
    run_backtest,
    compute_stats,
    print_stats,
    export_trades,
    ATR_LEN,
    NIFTY_LOT_SIZE,
    TOTAL_UNITS,
)


# ---------------------------------------------------------------------------
# Config -- optimized from grid search
# ---------------------------------------------------------------------------
SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "D"
LOOKBACK_YEARS = 5
INIT_CASH = 30_00_000

# Best params from optimization (top combo: Sharpe-adjusted)
BEST_SWING_LEN = 4
BEST_SCALAR = 0.7
BEST_LOOKAHEAD = 20
BEST_MIN_EVENTS = 10
BEST_MIN_SCORE = 35
BEST_N_TREES = 100
BEST_WINDOW = 1500

# F&O Futures fees (Indian market)
FEES = 0.00018
FIXED_FEES = 20


def build_equity_returns(equity_curve, init_cash=INIT_CASH):
    """Convert custom equity curve to daily returns Series.

    The backtest equity curve is in absolute PnL (cumulated realized + unrealized).
    We add init_cash to get a proper equity level, then compute pct_change.
    """
    equities = np.array([e.equity for e in equity_curve]) + init_cash
    dates = pd.to_datetime([e.date for e in equity_curve])

    equity_series = pd.Series(equities, index=dates)
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")]
    equity_series = equity_series.sort_index()

    equity_series.index = pd.DatetimeIndex(equity_series.index).tz_localize(None)
    returns = equity_series.pct_change().fillna(0.0)
    return returns


def build_benchmark_returns(df):
    """Build NIFTY buy-and-hold benchmark returns from price data."""
    bench_close = df["close"].copy()
    bench_close.index = pd.DatetimeIndex(bench_close.index).tz_localize(None)
    returns = bench_close.pct_change().fillna(0.0)
    returns.name = "NIFTY 50 Buy & Hold"
    return returns


def print_comparison_table(stats, returns_strat, returns_bench):
    """Print a Strategy vs Benchmark comparison table."""
    import openstatz as ostz

    print("\n" + "=" * 60)
    print("  STRATEGY vs BENCHMARK COMPARISON")
    print("=" * 60)

    strat_cagr = ostz.stats.cagr(returns_strat)
    strat_sharpe = ostz.stats.sharpe(returns_strat)
    strat_sortino = ostz.stats.sortino(returns_strat)
    strat_dd = ostz.stats.max_drawdown(returns_strat)
    strat_vol = ostz.stats.volatility(returns_strat)

    bench_cagr = ostz.stats.cagr(returns_bench)
    bench_sharpe = ostz.stats.sharpe(returns_bench)
    bench_sortino = ostz.stats.sortino(returns_bench)
    bench_dd = ostz.stats.max_drawdown(returns_bench)
    bench_vol = ostz.stats.volatility(returns_bench)

    table = pd.DataFrame({
        "ML CHoCH Strategy": [
            f"{strat_cagr * 100:.2f}%",
            f"{strat_sharpe:.2f}",
            f"{strat_sortino:.2f}",
            f"{strat_dd * 100:.2f}%",
            f"{strat_vol * 100:.2f}%",
            f"{stats['total_trades']}",
            f"{stats['win_rate_pct']}%",
            f"Rs. {stats['total_pnl_rupees']:,.0f}",
            f"Rs. {stats['max_drawdown_rupees']:,.0f}",
            f"{stats['profit_factor']}",
        ],
        "NIFTY Buy & Hold": [
            f"{bench_cagr * 100:.2f}%",
            f"{bench_sharpe:.2f}",
            f"{bench_sortino:.2f}",
            f"{bench_dd * 100:.2f}%",
            f"{bench_vol * 100:.2f}%",
            "-",
            "-",
            f"Rs. {(df['close'].iloc[-1] - df['close'].iloc[0]) * NIFTY_LOT_SIZE * TOTAL_UNITS:,.0f}",
            "-",
            "-",
        ],
    }, index=[
        "CAGR", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
        "Volatility", "Total Trades", "Win Rate", "Total PnL (6 lots)",
        "Max DD (Rs.)", "Profit Factor",
    ])

    print(table.to_string())
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=False)

    client = api(
        api_key=os.getenv("OPENALGO_API_KEY"),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    print("ML CHoCH NIFTY Futures -- VectorBT + OpenStatz Tearsheet")
    print("=" * 60)
    print(f"  Symbol: {SYMBOL} | Exchange: {EXCHANGE} | Interval: {INTERVAL}")
    print(f"  Lookback: {LOOKBACK_YEARS} years | Lot Size: {NIFTY_LOT_SIZE}")
    print(f"  Contracts per trade: {TOTAL_UNITS} ({TOTAL_UNITS * NIFTY_LOT_SIZE} qty)")
    print(f"  Best params: SWING={BEST_SWING_LEN} SCALAR={BEST_SCALAR} "
          f"LK={BEST_LOOKAHEAD} ME={BEST_MIN_EVENTS} MS={BEST_MIN_SCORE}")
    print(f"  Fees: {FEES*100:.4f}% + Rs.{FIXED_FEES}/order (F&O Futures)")
    print("=" * 60)

    # --- Fetch data ---
    df = fetch_nifty_data(client)

    # --- Override config to use best params ---
    import custom_indicators.ml_choch.backtest_nifty as bt_module
    bt_module.MIN_SCORE = BEST_MIN_SCORE
    bt_module.SWING_LEN = BEST_SWING_LEN
    bt_module.SCALAR = BEST_SCALAR

    # --- Run the faithful event-driven backtest (runs signal pipeline internally) ---
    print("\nRunning event-driven backtest with best params...")
    trades, equity_curve = run_backtest(df)

    # --- Compute stats ---
    stats = compute_stats(trades, equity_curve, df)
    print_stats(stats)

    # --- Export trades ---
    if trades:
        export_trades(trades, df)

    # --- Build returns series ---
    returns_strat = build_equity_returns(equity_curve)
    returns_strat.name = "ML CHoCH NIFTY (6 lots)"

    returns_bench = build_benchmark_returns(df)

    # --- OpenStatz Tearsheet ---
    print("\n--- OpenStatz Tearsheet ---")
    try:
        import openstatz as ostz

        mc = ostz.stats.montecarlo(returns_strat, sims=1000, bust=-0.10, goal=0.30)
        print(f"Monte Carlo (1000 sims): Bust prob={mc.bust_probability:.1%}, Goal prob={mc.goal_probability:.1%}")

        tearsheet_path = Path(__file__).parent / "tearsheet.html"
        ostz.dashboard(
            returns_strat,
            benchmark=returns_bench,
            output=str(tearsheet_path),
            title="ML CHoCH NIFTY Tearsheet",
            open_browser=False,
        )
        print(f"Tearsheet saved to: {tearsheet_path}")

        print_comparison_table(stats, returns_strat, returns_bench)

        print("\n--- Explanation ---")
        print(f"  Strategy returned {ostz.stats.cagr(returns_strat) * 100:.2f}% CAGR "
              f"vs NIFTY {ostz.stats.cagr(returns_bench) * 100:.2f}% Buy & Hold")
        print(f"  Max drawdown: {ostz.stats.max_drawdown(returns_strat) * 100:.2f}% "
              f"vs NIFTY {ostz.stats.max_drawdown(returns_bench) * 100:.2f}%")
        print(f"  On Rs {INIT_CASH:,} capital, worst temporary loss = "
              f"Rs {abs(ostz.stats.max_drawdown(returns_strat)) * INIT_CASH:,.0f}")
        print(f"  Profit Factor: {stats['profit_factor']} | "
              f"Win Rate: {stats['win_rate_pct']}% ({stats['total_trades']} trades)")

    except ImportError:
        print("\nOpenStatz not installed. Install: pip install openstatz")
        print("Skipping tearsheet generation.")
        print("\n--- Console Metrics ---")
        daily_returns = returns_strat.values
        print(f"  Mean daily return: {np.mean(daily_returns) * 100:.4f}%")
        print(f"  Std daily return:  {np.std(daily_returns) * 100:.4f}%")
        ann_return = np.mean(daily_returns) * 252
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        print(f"  Annualized return: {ann_return * 100:.2f}%")
        print(f"  Annualized vol:    {ann_vol * 100:.2f}%")
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        print(f"  Sharpe ratio:      {sharpe:.2f}")
