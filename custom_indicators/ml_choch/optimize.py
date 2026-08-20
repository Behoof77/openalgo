"""
ML CHoCH Parameter Optimization
=================================
Grid search over key parameters with two phases:

Phase 1 (expensive): SWING_LEN x SCALAR grid, each generating signals at min_score=0,
  then sweeping MIN_SCORE as a cheap post-hoc filter.
Phase 2 (expensive): Fix best SWING_LEN/SCALAR, sweep LOOKAHEAD x MIN_EVENTS,
  again filtering by MIN_SCORE cheaply.

Output:
  - optimization_report.html  (heatmaps + ranked table)
  - optimization_results.csv  (all combos ranked by Sharpe)
"""

import os
import sys
import time
import itertools
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dotenv import find_dotenv, load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openalgo import api
from custom_indicators.ml_choch.signal_generator import generate_signals, SignalResult
from custom_indicators.ml_choch.choch_detector import detect_choch
from custom_indicators.ml_choch.backtest_nifty import (
    Trade, DailyPnl, process_bar_long, process_bar_short,
    compute_stats, NIFTY_LOT_SIZE, TOTAL_UNITS, SYMBOL, EXCHANGE, INTERVAL, LOOKBACK_YEARS,
)

OUTPUT_DIR = Path(__file__).parent

# Fixed constants
N_TREES = 100
ATR_LEN = 14
WINDOW = 1500

# Phase 1 grid
SWING_LEN_RANGE = [3, 4, 5, 6, 7]
SCALAR_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7]

# MIN_SCORE sweep (applied as filter — no re-generation)
MIN_SCORE_RANGE = [35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]

# Phase 2 grid
LOOKAHEAD_RANGE = [10, 15, 20, 25, 30]
MIN_EVENTS_RANGE = [5, 8, 10, 15]

MIN_TRADES_FOR_VALID = 5


@dataclass
class OptResult:
    phase: int
    swing_len: int
    scalar: float
    lookahead: int
    min_events: int
    min_score: float
    total_trades: int
    win_rate: float
    total_pnl: float
    sharpe: float
    max_dd: float
    profit_factor: float
    long_trades: int
    short_trades: int
    avg_holding: float
    buy_hold: float


def fetch_data(client) -> pd.DataFrame:
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    try:
        result = client.history(
            symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
        )
        if isinstance(result, dict):
            raise ConnectionError(result.get("message", "API error"))
        df = result
        if df is not None and not df.empty:
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)
            print(f"  [OpenAlgo API] {len(df)} bars")
            return df
    except Exception as e:
        print(f"  OpenAlgo API unavailable: {e}")

    print("  Falling back to yfinance...")
    import yfinance as yf
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        raise ValueError("No data")
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.sort_index()
    print(f"  [yfinance] {len(df)} bars")
    return df


def run_backtest_for_signals(
    df: pd.DataFrame,
    signals: List[SignalResult],
    swing_len: int,
    min_score: float,
) -> tuple:
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)

    structure = detect_choch(high, low, close, swing_len)

    valid_signals = [s for s in signals if s.probability >= min_score]
    signal_map = {s.bar_index: s for s in valid_signals}

    trades: List[Trade] = []
    current_trade: Optional[Trade] = None

    for i in range(len(df)):
        if current_trade is not None:
            if current_trade.direction:
                closed = process_bar_long(current_trade, i, high[i], low[i], close[i])
            else:
                closed = process_bar_short(current_trade, i, high[i], low[i], close[i])
            if closed:
                trades.append(current_trade)
                current_trade = None

        if current_trade is None and i in signal_map:
            sig = signal_map[i]
            sl = structure["last_swing_low"][i] if sig.direction else structure["last_swing_high"][i]
            if np.isnan(sl) or sl <= 0:
                continue
            if sig.tp1 <= 0 or sig.tp2 <= 0 or sig.tp3 <= 0:
                continue
            entry_p = close[i]
            if sig.direction:
                if not (entry_p < sig.tp1 and sig.tp1 <= sig.tp3):
                    continue
            else:
                if not (entry_p > sig.tp1 and sig.tp1 >= sig.tp3):
                    continue

            current_trade = Trade(
                entry_bar=i, entry_price=close[i], direction=sig.direction,
                sl_price=sl, tp1=sig.tp1, tp2=sig.tp2, tp3=sig.tp3,
                probability=sig.probability, price_extreme_after_tp1=0.0,
                price_extreme_after_tp2=0.0,
            )

    if current_trade is not None:
        if current_trade.direction:
            process_bar_long(current_trade, len(df) - 1, close[-1], close[-1], close[-1])
        else:
            process_bar_short(current_trade, len(df) - 1, close[-1], close[-1], close[-1])
        trades.append(current_trade)

    return trades


def trades_to_stats(trades: List[Trade], df: pd.DataFrame) -> dict:
    if not trades:
        return {"error": "No trades"}
    return compute_stats(trades, [], df)


def score_result(stats: dict, min_trades: int = MIN_TRADES_FOR_VALID) -> OptResult:
    if "error" in stats:
        return None
    if stats["total_trades"] < min_trades:
        return None
    return OptResult(
        phase=0, swing_len=0, scalar=0.0, lookahead=0, min_events=0, min_score=0.0,
        total_trades=stats["total_trades"],
        win_rate=stats["win_rate_pct"],
        total_pnl=stats["total_pnl_rupees"],
        sharpe=stats["sharpe_ratio"],
        max_dd=stats["max_drawdown_rupees"],
        profit_factor=stats["profit_factor"],
        long_trades=stats["long_trades"],
        short_trades=stats["short_trades"],
        avg_holding=stats["avg_holding_bars"],
        buy_hold=stats["buy_hold_pnl_rupees"],
    )


def generate_phase1_signals(df, swing_len, scalar):
    print(f"  Phase 1: SWING_LEN={swing_len}, SCALAR={scalar} ...", end=" ", flush=True)
    t0 = time.time()
    signals, _ = generate_signals(
        df, lookahead=20, swing_len=swing_len, atr_len=ATR_LEN,
        min_score=0, n_trees=N_TREES, min_events=10, window=WINDOW, scalar=scalar,
    )
    elapsed = time.time() - t0
    n_valid = sum(1 for s in signals if s.probability > 0)
    print(f"{elapsed:.1f}s, {n_valid} signals")
    return signals


def generate_phase2_signals(df, swing_len, scalar, lookahead, min_events):
    print(f"  Phase 2: LK={lookahead}, ME={min_events} ...", end=" ", flush=True)
    t0 = time.time()
    signals, _ = generate_signals(
        df, lookahead=lookahead, swing_len=swing_len, atr_len=ATR_LEN,
        min_score=0, n_trees=N_TREES, min_events=min_events, window=WINDOW, scalar=scalar,
    )
    elapsed = time.time() - t0
    n_valid = sum(1 for s in signals if s.probability > 0)
    print(f"{elapsed:.1f}s, {n_valid} signals")
    return signals


def run_optimization():
    load_dotenv(find_dotenv(), override=False)
    client = api(
        api_key=os.getenv("OPENALGO_API_KEY"),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    print("=" * 60)
    print("  ML CHoCH PARAMETER OPTIMIZATION")
    print("=" * 60)

    df = fetch_data(client)
    print(f"  Data: {len(df)} bars, {df.index[0].date()} to {df.index[-1].date()}")

    # Pre-compute benchmark
    bh_pnl = (df["close"].iloc[-1] - df["close"].iloc[0]) * NIFTY_LOT_SIZE

    all_results: List[OptResult] = []
    signal_cache = {}

    # -----------------------------------------------------------------------
    # Phase 1: SWING_LEN x SCALAR grid, then sweep MIN_SCORE
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  PHASE 1: SWING_LEN x SCALAR ({len(SWING_LEN_RANGE)}x{len(SCALAR_RANGE)} = "
          f"{len(SWING_LEN_RANGE)*len(SCALAR_RANGE)} signal runs)")
    print(f"  Then filter by {len(MIN_SCORE_RANGE)} MIN_SCORE values each")
    print(f"{'='*60}\n")

    phase1_combos = list(itertools.product(SWING_LEN_RANGE, SCALAR_RANGE))
    total_phase1 = len(phase1_combos) * len(MIN_SCORE_RANGE)
    run_count = 0

    for swing_len, scalar in phase1_combos:
        signals = generate_phase1_signals(df, swing_len, scalar)
        signal_cache[(swing_len, scalar)] = signals

        for min_score in MIN_SCORE_RANGE:
            run_count += 1
            trades = run_backtest_for_signals(df, signals, swing_len, min_score)
            stats = trades_to_stats(trades, df)
            result = score_result(stats)
            if result:
                result.phase = 1
                result.swing_len = swing_len
                result.scalar = scalar
                result.lookahead = 20
                result.min_events = 10
                result.min_score = min_score
                result.buy_hold = bh_pnl
                all_results.append(result)

        print(f"    -> {run_count}/{total_phase1} backtests done", flush=True)

    # -----------------------------------------------------------------------
    # Phase 2: Fix best SWING_LEN/SCALAR, sweep LOOKAHEAD x MIN_EVENTS
    # -----------------------------------------------------------------------
    phase1_valid = [r for r in all_results if r.phase == 1]
    if not phase1_valid:
        print("\nNo valid Phase 1 results. Aborting.")
        return

    phase1_valid.sort(key=lambda r: r.sharpe, reverse=True)
    best = phase1_valid[0]
    best_swing = best.swing_len
    best_scalar = best.scalar

    print(f"\n{'='*60}")
    print(f"  PHASE 1 BEST: SWING_LEN={best_swing}, SCALAR={best_scalar}, "
          f"Sharpe={best.sharpe:.2f}, PnL=Rs.{best.total_pnl:,.0f}, "
          f"Trades={best.total_trades}")
    print(f"{'='*60}")

    print(f"\n  PHASE 2: LOOKAHEAD x MIN_EVENTS ({len(LOOKAHEAD_RANGE)}x{len(MIN_EVENTS_RANGE)} = "
          f"{len(LOOKAHEAD_RANGE)*len(MIN_EVENTS_RANGE)} signal runs)")
    print(f"  Then filter by {len(MIN_SCORE_RANGE)} MIN_SCORE values each\n")

    phase2_combos = list(itertools.product(LOOKAHEAD_RANGE, MIN_EVENTS_RANGE))
    total_phase2 = len(phase2_combos) * len(MIN_SCORE_RANGE)
    run_count2 = 0

    for lookahead, min_events in phase2_combos:
        signals = generate_phase2_signals(df, best_swing, best_scalar, lookahead, min_events)

        for min_score in MIN_SCORE_RANGE:
            run_count2 += 1
            trades = run_backtest_for_signals(df, signals, best_swing, min_score)
            stats = trades_to_stats(trades, df)
            result = score_result(stats)
            if result:
                result.phase = 2
                result.swing_len = best_swing
                result.scalar = best_scalar
                result.lookahead = lookahead
                result.min_events = min_events
                result.min_score = min_score
                result.buy_hold = bh_pnl
                all_results.append(result)

        print(f"    -> {run_count2}/{total_phase2} backtests done", flush=True)

    # -----------------------------------------------------------------------
    # Aggregate and rank results
    # -----------------------------------------------------------------------
    if not all_results:
        print("\nNo valid results found. Try lowering MIN_SCORE or MIN_TRADES.")
        return

    results_df = pd.DataFrame([vars(r) for r in all_results])
    results_df["excess_return"] = results_df["total_pnl"] - results_df["buy_hold"]
    results_df["sharpe_rank"] = results_df["sharpe"].rank(ascending=False)
    results_df["composite_score"] = (
        results_df["sharpe"] * 0.5
        + (results_df["total_pnl"] / max(results_df["total_pnl"].max(), 1)) * 0.3
        - (results_df["max_dd"] / max(results_df["max_dd"].max(), 1)) * 0.2
    )
    results_df = results_df.sort_values("composite_score", ascending=False)

    print(f"\n{'='*60}")
    print(f"  OPTIMIZATION COMPLETE: {len(results_df)} valid parameter combos")
    print(f"{'='*60}")

    print(f"\n  TOP 10 BY COMPOSITE SCORE:")
    print("-" * 110)
    top = results_df.head(10)
    for _, r in top.iterrows():
        print(f"  SL={int(r['swing_len']):2d}  SC={r['scalar']:.1f}  LK={int(r['lookahead']):2d}  "
              f"ME={int(r['min_events']):2d}  MS={r['min_score']:5.1f}  |  "
              f"Trades={int(r['total_trades']):3d}  WR={r['win_rate']:5.1f}%  "
              f"Sharpe={r['sharpe']:6.2f}  PnL=Rs.{r['total_pnl']:>12,.0f}  "
              f"DD=Rs.{r['max_dd']:>10,.0f}  PF={r['profit_factor']:5.2f}")

    # -----------------------------------------------------------------------
    # Save CSV
    # -----------------------------------------------------------------------
    csv_path = OUTPUT_DIR / "optimization_results.csv"
    results_df.to_csv(str(csv_path), index=False)
    print(f"\n  Full results saved to: {csv_path}")

    # -----------------------------------------------------------------------
    # Generate heatmaps and report
    # -----------------------------------------------------------------------
    generate_report(results_df, best_swing, best_scalar, bh_pnl)


def generate_report(results_df: pd.DataFrame, best_swing: int, best_scalar: float, bh_pnl: float):
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=[
            "Sharpe: SWING_LEN x SCALAR (Phase 1, MS=45)",
            "Trades: SWING_LEN x SCALAR (Phase 1, MS=45)",
            "Sharpe: LOOKAHEAD x MIN_EVENTS (Phase 2, best SL/SC, MS=45)",
            "Trades: LOOKAHEAD x MIN_EVENTS (Phase 2, best SL/SC, MS=45)",
            "Sharpe: SWING_LEN x MIN_SCORE",
            "PnL: SWING_LEN x MIN_SCORE",
            "Sharpe: SCALAR x MIN_SCORE",
            "PnL: SCALAR x MIN_SCORE",
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1,
    )

    # Phase 1 heatmaps: SWING_LEN x SCALAR at fixed MIN_SCORE=45
    p1 = results_df[results_df["phase"] == 1]
    ms45 = p1[p1["min_score"] == 45.0] if len(p1[p1["min_score"] == 45.0]) > 0 else p1

    if not ms45.empty:
        # Sharpe heatmap
        pivot = ms45.pivot_table(values="sharpe", index="swing_len", columns="scalar", aggfunc="max")
        if not pivot.empty:
            fig.add_trace(go.Heatmap(
                z=pivot.values, x=[f"{c:.1f}" for c in pivot.columns],
                y=pivot.index.astype(str), colorscale="RdYlGn",
                text=np.round(pivot.values, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=1, col=1)

        # Trades heatmap
        pivot_t = ms45.pivot_table(values="total_trades", index="swing_len", columns="scalar", aggfunc="max")
        if not pivot_t.empty:
            fig.add_trace(go.Heatmap(
                z=pivot_t.values, x=[f"{c:.1f}" for c in pivot_t.columns],
                y=pivot_t.index.astype(str), colorscale="Blues",
                text=pivot_t.values.astype(int), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=1, col=2)

    # Phase 2 heatmaps: LOOKAHEAD x MIN_EVENTS at fixed MIN_SCORE=45
    p2 = results_df[results_df["phase"] == 2]
    ms45_p2 = p2[p2["min_score"] == 45.0] if len(p2[p2["min_score"] == 45.0]) > 0 else p2

    if not ms45_p2.empty:
        pivot2 = ms45_p2.pivot_table(values="sharpe", index="lookahead", columns="min_events", aggfunc="max")
        if not pivot2.empty:
            fig.add_trace(go.Heatmap(
                z=pivot2.values, x=pivot2.columns.astype(str),
                y=pivot2.index.astype(str), colorscale="RdYlGn",
                text=np.round(pivot2.values, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=2, col=1)

        pivot2_t = ms45_p2.pivot_table(values="total_trades", index="lookahead", columns="min_events", aggfunc="max")
        if not pivot2_t.empty:
            fig.add_trace(go.Heatmap(
                z=pivot2_t.values, x=pivot2_t.columns.astype(str),
                y=pivot2_t.index.astype(str), colorscale="Blues",
                text=pivot2_t.values.astype(int), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=2, col=2)

    # Combined: SWING_LEN x MIN_SCORE (best scalar for each)
    all_valid = results_df.copy()
    if not all_valid.empty:
        pivot3 = all_valid.pivot_table(values="sharpe", index="swing_len", columns="min_score", aggfunc="max")
        if not pivot3.empty:
            fig.add_trace(go.Heatmap(
                z=pivot3.values, x=pivot3.columns.astype(str),
                y=pivot3.index.astype(str), colorscale="RdYlGn",
                text=np.round(pivot3.values, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=3, col=1)

        pivot3_pnl = all_valid.pivot_table(values="total_pnl", index="swing_len", columns="min_score", aggfunc="max")
        if not pivot3_pnl.empty:
            fig.add_trace(go.Heatmap(
                z=pivot3_pnl.values / 1e5, x=pivot3_pnl.columns.astype(str),
                y=pivot3_pnl.index.astype(str), colorscale="RdYlGn",
                text=np.round(pivot3_pnl.values / 1e5, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=3, col=2)

        pivot4 = all_valid.pivot_table(values="sharpe", index="scalar", columns="min_score", aggfunc="max")
        if not pivot4.empty:
            fig.add_trace(go.Heatmap(
                z=pivot4.values, x=pivot4.columns.astype(str),
                y=[f"{v:.1f}" for v in pivot4.index], colorscale="RdYlGn",
                text=np.round(pivot4.values, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=4, col=1)

        pivot4_pnl = all_valid.pivot_table(values="total_pnl", index="scalar", columns="min_score", aggfunc="max")
        if not pivot4_pnl.empty:
            fig.add_trace(go.Heatmap(
                z=pivot4_pnl.values / 1e5, x=pivot4_pnl.columns.astype(str),
                y=[f"{v:.1f}" for v in pivot4_pnl.index], colorscale="RdYlGn",
                text=np.round(pivot4_pnl.values / 1e5, 2), texttemplate="%{text}",
                textfont={"size": 10},
            ), row=4, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=1400,
        width=1200,
        title_text=(
            f"ML CHoCH Optimization Report | Best: SL={best_swing} SC={best_scalar} | "
            f"Buy&Hold=Rs.{bh_pnl:,.0f}"
        ),
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=40),
    )

    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(title_text="Scalar" if j == 1 else "Min Score" if i >= 3 else "Min Events", row=i, col=j)
            fig.update_yaxes(title_text="Swing Len" if i <= 2 else "Lookahead" if i == 2 and j == 1 else "Scalar" if i == 4 else "Swing Len", row=i, col=j)

    out_path = OUTPUT_DIR / "optimization_report.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\n  Heatmap report saved to: {out_path}")


if __name__ == "__main__":
    run_optimization()
