"""
ML CHoCH NIFTY 3m Intraday -- OpenStatz Tearsheet
===================================================
Runs the ML CHoCH signal pipeline on 3-minute intraday data with:
  - EOD session flatten (no overnight carry)
  - 6-lot benchmark (same capital as strategy)
  - 2-year lookback
  - OpenStatz interactive tearsheet output

Usage:
  uv run python custom_indicators/ml_choch/backtest_intraday.py
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
    run_backtest,
    compute_stats,
    print_stats,
    export_trades,
    NIFTY_LOT_SIZE,
    TOTAL_UNITS,
)


# ---------------------------------------------------------------------------
# Config -- tuned for 3m intraday
# ---------------------------------------------------------------------------
SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "3m"
LOOKBACK_YEARS = 2
INIT_CASH = 30_00_000

# Intraday params (swing detection needs wider windows on 3m bars)
BEST_SWING_LEN = 10
BEST_SCALAR = 0.7
BEST_LOOKAHEAD = 40
BEST_MIN_EVENTS = 15
BEST_MIN_SCORE = 35
BEST_N_TREES = 100
BEST_WINDOW = 5000

# Indian F&O intraday fees
FEES = 0.00018
FIXED_FEES = 20

SCRIPT_DIR = Path(__file__).resolve().parent


def fetch_3m_data(client) -> pd.DataFrame:
    """Fetch NIFTY 3m data in 3-month chunks to avoid gateway timeouts."""
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    print(f"Fetching {SYMBOL} {INTERVAL} data from {start_date} to {end_date} (chunked)...")

    chunks = []
    chunk_start = start_date
    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=90), end_date)
        print(f"  {chunk_start} -> {chunk_end}...", end=" ", flush=True)

        result = client.history(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            interval=INTERVAL,
            start_date=chunk_start.strftime("%Y-%m-%d"),
            end_date=chunk_end.strftime("%Y-%m-%d"),
        )
        if isinstance(result, dict):
            print(f"API error: {result.get('message', result)}")
            chunk_start = chunk_end
            continue
        if result is not None and not result.empty:
            chunks.append(result)
            print(f"{len(result)} bars")
        else:
            print("empty")

        chunk_start = chunk_end + timedelta(days=1)

    if not chunks:
        raise ValueError(f"No {INTERVAL} data returned from API for any chunk")

    df = pd.concat(chunks)

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df[~df.index.duplicated(keep="last")]

    print(f"  Total: {len(df)} bars ({df.index[0]} to {df.index[-1]})")
    sessions = df.index.normalize().nunique()
    print(f"  {sessions} trading sessions, ~{len(df) // max(sessions,1)} bars/session")
    return df


def build_equity_returns(equity_curve, init_cash=INIT_CASH):
    """Convert absolute PnL equity curve to returns Series for openstatz."""
    equities = np.array([e.equity for e in equity_curve]) + init_cash
    dates = pd.to_datetime([e.date for e in equity_curve])

    equity_series = pd.Series(equities, index=dates)
    equity_series = equity_series[~equity_series.index.duplicated(keep="last")]
    equity_series = equity_series.sort_index()
    equity_series.index = pd.DatetimeIndex(equity_series.index).tz_localize(None)

    returns = equity_series.pct_change().fillna(0.0)
    return returns


def build_benchmark_returns_6lots(df):
    """Buy & hold 6 lots NIFTY futures -- same capital base as strategy.

    Returns a bar-level returns series where each bar's return reflects
    the PnL of holding 6 x 25 = 150 qty from bar 0.
    """
    close = df["close"].astype(np.float64)
    qty = TOTAL_UNITS * NIFTY_LOT_SIZE  # 150
    initial_value = close.iloc[0] * qty

    position_value = close * qty
    returns = position_value.pct_change().fillna(0.0)
    returns.name = "NIFTY 6-lot Buy & Hold"
    return returns


def print_comparison_table(stats, returns_strat, returns_bench, df):
    """Print a Strategy vs 6-lot Benchmark comparison table."""
    import openstatz as ostz

    print("\n" + "=" * 65)
    print("  STRATEGY vs 6-LOT NIFTY BUY & HOLD")
    print("=" * 65)

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

    qty = TOTAL_UNITS * NIFTY_LOT_SIZE
    bh_pnl = (df["close"].iloc[-1] - df["close"].iloc[0]) * qty

    table = pd.DataFrame({
        "ML CHoCH 3m": [
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
        "NIFTY 6-lot B&H": [
            f"{bench_cagr * 100:.2f}%",
            f"{bench_sharpe:.2f}",
            f"{bench_sortino:.2f}",
            f"{bench_dd * 100:.2f}%",
            f"{bench_vol * 100:.2f}%",
            "-",
            "-",
            f"Rs. {bh_pnl:,.0f}",
            "-",
            "-",
        ],
    }, index=[
        "CAGR", "Sharpe", "Sortino", "Max DD%",
        "Volatility", "Trades", "Win Rate", "PnL (6 lots)",
        "Max DD (Rs.)", "Profit Factor",
    ])

    print(table.to_string())
    print("=" * 65)


if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=False)

    client = api(
        api_key=os.getenv("OPENALGO_API_KEY"),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    print("ML CHoCH NIFTY 3m Intraday -- OpenStatz Tearsheet")
    print("=" * 65)
    print(f"  Symbol: {SYMBOL} | Exchange: {EXCHANGE} | Interval: {INTERVAL}")
    print(f"  Lookback: {LOOKBACK_YEARS} year(s) | Lot Size: {NIFTY_LOT_SIZE}")
    print(f"  Contracts: {TOTAL_UNITS} ({TOTAL_UNITS * NIFTY_LOT_SIZE} qty)")
    print(f"  Params: SWING={BEST_SWING_LEN} SCALAR={BEST_SCALAR} "
          f"LK={BEST_LOOKAHEAD} ME={BEST_MIN_EVENTS} MS={BEST_MIN_SCORE}")
    print(f"  Mode: INTRADAY (EOD flatten every session)")
    print("=" * 65)

    # --- Fetch 3m data ---
    df = fetch_3m_data(client)

    # --- Override config ---
    import custom_indicators.ml_choch.backtest_nifty as bt_module
    bt_module.MIN_SCORE = BEST_MIN_SCORE
    bt_module.SWING_LEN = BEST_SWING_LEN
    bt_module.SCALAR = BEST_SCALAR
    bt_module.LOOKAHEAD = BEST_LOOKAHEAD
    bt_module.MIN_EVENTS = BEST_MIN_EVENTS
    bt_module.N_TREES = BEST_N_TREES
    bt_module.WINDOW = BEST_WINDOW

    # --- Run event-driven backtest with intraday flatten ---
    print("\nRunning intraday backtest (EOD flatten)...")
    trades, equity_curve = run_backtest(df, intraday=True)

    # --- Compute stats ---
    stats = compute_stats(trades, equity_curve, df)
    print_stats(stats)

    # --- Export trades ---
    if trades:
        csv_path = SCRIPT_DIR / "backtest_trades_intraday_3m.csv"
        rows = []
        for t in trades:
            rows.append({
                "entry_date": str(df.index[t.entry_bar]) if t.entry_bar < len(df) else "",
                "exit_date": str(df.index[t.exit_bar]) if 0 <= t.exit_bar < len(df) else "",
                "direction": "LONG" if t.direction else "SHORT",
                "entry_price": round(t.entry_price, 2),
                "exit_price": round(t.exit_price, 2),
                "sl": round(t.sl_price, 2),
                "tp1": round(t.tp1, 2),
                "tp2": round(t.tp2, 2),
                "tp3": round(t.tp3, 2),
                "probability": round(t.probability, 1),
                "exit_reason": t.exit_reason,
                "pnl_points": round(t.pnl_points, 2),
                "pnl_rupees": round(t.pnl_points * NIFTY_LOT_SIZE, 2),
                "holding_bars": t.exit_bar - t.entry_bar if t.exit_bar >= 0 else 0,
            })
        pd.DataFrame(rows).to_csv(str(csv_path), index=False)
        print(f"Trade log saved: {csv_path}")

    # --- Exit breakdown ---
    if trades:
        exit_reasons = {}
        for t in trades:
            r = t.exit_reason
            if r not in exit_reasons:
                exit_reasons[r] = {"count": 0, "pnl": 0.0}
            exit_reasons[r]["count"] += 1
            exit_reasons[r]["pnl"] += t.pnl_points * NIFTY_LOT_SIZE
        print("\nExit breakdown:")
        for reason, data in sorted(exit_reasons.items()):
            print(f"  {reason:12s}: {data['count']:3d} trades, Rs.{data['pnl']:>12,.0f}")

    # --- Build returns series ---
    if not equity_curve:
        print("\nNo equity curve -- cannot generate tearsheet.")
        sys.exit(1)

    returns_strat = build_equity_returns(equity_curve)
    returns_strat.name = "ML CHoCH NIFTY 3m (6 lots)"

    returns_bench = build_benchmark_returns_6lots(df)

    # --- OpenStatz Tearsheet ---
    print("\n--- OpenStatz Tearsheet ---")
    try:
        import openstatz as ostz

        mc = ostz.stats.montecarlo(returns_strat, sims=1000, bust=-0.10, goal=0.30)
        print(f"Monte Carlo (1000 sims): Bust={mc.bust_probability:.1%}, Goal={mc.goal_probability:.1%}")

        tearsheet_path = SCRIPT_DIR / "tearsheet_intraday_3m.html"
        ostz.dashboard(
            returns_strat,
            benchmark=returns_bench,
            output=str(tearsheet_path),
            title="ML CHoCH NIFTY 3m Intraday",
            open_browser=False,
        )
        print(f"Tearsheet saved: {tearsheet_path}")

        print_comparison_table(stats, returns_strat, returns_bench, df)

        print("\n--- Summary ---")
        print(f"  Strategy CAGR: {ostz.stats.cagr(returns_strat) * 100:.2f}% "
              f"| NIFTY B&H: {ostz.stats.cagr(returns_bench) * 100:.2f}%")
        print(f"  Strategy Sharpe: {ostz.stats.sharpe(returns_strat):.2f} "
              f"| NIFTY: {ostz.stats.sharpe(returns_bench):.2f}")
        print(f"  Max DD: {ostz.stats.max_drawdown(returns_strat) * 100:.2f}% "
              f"| NIFTY: {ostz.stats.max_drawdown(returns_bench) * 100:.2f}%")
        print(f"  On Rs {INIT_CASH:,}, worst loss = "
              f"Rs {abs(ostz.stats.max_drawdown(returns_strat)) * INIT_CASH:,.0f}")

    except ImportError:
        print("\nOpenStatz not installed. Install: pip install openstatz")
        if len(returns_strat) > 1:
            ann_ret = returns_strat.mean() * 252
            ann_vol = returns_strat.std() * np.sqrt(252)
            print(f"  Annualized return: {ann_ret * 100:.2f}%")
            print(f"  Annualized vol:    {ann_vol * 100:.2f}%")
            print(f"  Sharpe:            {ann_ret / ann_vol:.2f}" if ann_vol > 0 else "  Sharpe: N/A")
