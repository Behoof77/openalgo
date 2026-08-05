"""
ML CHoCH NIFTY Futures Backtest
================================
Event-driven backtest of the ML CHoCH indicator on NIFTY daily data (5 years).

Strategy:
  - LONG: Buy 6 NIFTY futures on bullish CHoCH (probability >= min_score)
  - SHORT: Sell 6 NIFTY futures on bearish CHoCH (probability >= min_score)
  - Scale-out: 4 units at TP1, 1 unit at TP2, 1 unit at TP3
  - Trailing exit: if price fades back to a TP level after exceeding it,
    exit all remaining at that level
  - SL: swing structure level (swing low for longs, swing high for shorts)
  - One position at a time (flat before new entry)

Usage:
  uv run python custom_indicators/ml_choch/backtest_nifty.py
"""

import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import find_dotenv, load_dotenv

# Ensure project root is on path for imports
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from openalgo import api, ta

from custom_indicators.ml_choch.signal_generator import generate_signals
from custom_indicators.ml_choch.choch_detector import detect_choch


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "D"
LOOKBACK_YEARS = 5
NIFTY_LOT_SIZE = 25  # Current lot size (changed from 50 in late 2024)
TOTAL_UNITS = 6       # Number of futures contracts per trade

# Signal generation params
MIN_SCORE = 60.0
N_TREES = 100
MIN_EVENTS = 10
WINDOW = 1500
SWING_LEN = 5
ATR_LEN = 14
LOOKAHEAD = 20
SCALAR = 0.5

# Position sizing
POSITION_SIZE = TOTAL_UNITS * NIFTY_LOT_SIZE  # total quantity in shares


# ---------------------------------------------------------------------------
# Trade tracking
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    """Single trade record."""
    entry_bar: int
    entry_price: float
    direction: bool          # True = long, False = short
    sl_price: float
    tp1: float
    tp2: float
    tp3: float
    probability: float
    # State
    units_remaining: int = TOTAL_UNITS
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    tp1_hit_bar: int = -1
    tp2_hit_bar: int = -1
    # Trailing state
    price_extreme_after_tp1: float = 0.0  # max(high) after TP1 for longs
    price_extreme_after_tp2: float = 0.0  # max(high) after TP2 for longs
    # Exit tracking
    exit_bar: int = -1
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_points: float = 0.0
    partial_exits: list = field(default_factory=list)  # [(price, qty, label)]


@dataclass
class DailyPnl:
    """Daily equity snapshot."""
    bar_index: int
    date: str
    equity: float
    drawdown: float
    trade_pnl: float


# ---------------------------------------------------------------------------
# Position management engine
# ---------------------------------------------------------------------------
def process_bar_long(
    trade: Trade,
    bar_idx: int,
    bar_high: float,
    bar_low: float,
    bar_close: float,
) -> bool:
    """Process one daily bar for a LONG position. Returns True if position is closed."""
    if trade.units_remaining <= 0:
        return True

    # --- Priority 1: Stop Loss ---
    if bar_low <= trade.sl_price:
        trade.partial_exits.append((trade.sl_price, trade.units_remaining, "SL"))
        trade.pnl_points += trade.units_remaining * (trade.sl_price - trade.entry_price)
        trade.units_remaining = 0
        trade.exit_bar = bar_idx
        trade.exit_price = trade.sl_price
        trade.exit_reason = "SL"
        return True

    # --- Priority 2: TP3 (exit all remaining) ---
    if bar_high >= trade.tp3 and not trade.tp3_hit and trade.tp3 > 0:
        trade.partial_exits.append((trade.tp3, trade.units_remaining, "TP3"))
        trade.pnl_points += trade.units_remaining * (trade.tp3 - trade.entry_price)
        trade.units_remaining = 0
        trade.tp3_hit = True
        trade.exit_bar = bar_idx
        trade.exit_price = trade.tp3
        trade.exit_reason = "TP3"
        return True

    # --- Priority 3: TP2 (sell 1 unit) ---
    if bar_high >= trade.tp2 and not trade.tp2_hit and trade.tp2 > 0 and trade.units_remaining > 0:
        sell_qty = min(1, trade.units_remaining)
        trade.partial_exits.append((trade.tp2, sell_qty, "TP2"))
        trade.pnl_points += sell_qty * (trade.tp2 - trade.entry_price)
        trade.units_remaining -= sell_qty
        trade.tp2_hit = True
        trade.tp2_hit_bar = bar_idx

    # --- Priority 4: TP1 (sell 4 units) ---
    if bar_high >= trade.tp1 and not trade.tp1_hit and trade.tp1 > 0 and trade.units_remaining > 0:
        sell_qty = min(4, trade.units_remaining)
        trade.partial_exits.append((trade.tp1, sell_qty, "TP1"))
        trade.pnl_points += sell_qty * (trade.tp1 - trade.entry_price)
        trade.units_remaining -= sell_qty
        trade.tp1_hit = True
        trade.tp1_hit_bar = bar_idx

    if trade.units_remaining <= 0:
        trade.exit_bar = bar_idx
        trade.exit_price = trade.tp1  # last exit was at TP1
        trade.exit_reason = "TP1+TP2"
        return True

    # --- Update trailing extremes ---
    if trade.tp1_hit and not trade.tp2_hit:
        trade.price_extreme_after_tp1 = max(trade.price_extreme_after_tp1, bar_high)
    if trade.tp2_hit and not trade.tp3_hit:
        trade.price_extreme_after_tp2 = max(trade.price_extreme_after_tp2, bar_high)

    # --- Trailing exit: price faded back to TP2 after exceeding it ---
    if trade.tp2_hit and not trade.tp3_hit and bar_idx > trade.tp2_hit_bar:
        if trade.price_extreme_after_tp2 > trade.tp2 and bar_low <= trade.tp2:
            trade.partial_exits.append((trade.tp2, trade.units_remaining, "TRAIL_TP2"))
            trade.pnl_points += trade.units_remaining * (trade.tp2 - trade.entry_price)
            trade.units_remaining = 0
            trade.exit_bar = bar_idx
            trade.exit_price = trade.tp2
            trade.exit_reason = "TRAIL_TP2"
            return True

    # --- Trailing exit: price faded back to TP1 after exceeding it ---
    if trade.tp1_hit and not trade.tp2_hit and bar_idx > trade.tp1_hit_bar:
        if trade.price_extreme_after_tp1 > trade.tp1 and bar_low <= trade.tp1:
            trade.partial_exits.append((trade.tp1, trade.units_remaining, "TRAIL_TP1"))
            trade.pnl_points += trade.units_remaining * (trade.tp1 - trade.entry_price)
            trade.units_remaining = 0
            trade.exit_bar = bar_idx
            trade.exit_price = trade.tp1
            trade.exit_reason = "TRAIL_TP1"
            return True

    return False


def process_bar_short(
    trade: Trade,
    bar_idx: int,
    bar_high: float,
    bar_low: float,
    bar_close: float,
) -> bool:
    """Process one daily bar for a SHORT position. Returns True if position is closed."""
    if trade.units_remaining <= 0:
        return True

    # --- Priority 1: Stop Loss ---
    if bar_high >= trade.sl_price:
        trade.partial_exits.append((trade.sl_price, trade.units_remaining, "SL"))
        trade.pnl_points += trade.units_remaining * (trade.entry_price - trade.sl_price)
        trade.units_remaining = 0
        trade.exit_bar = bar_idx
        trade.exit_price = trade.sl_price
        trade.exit_reason = "SL"
        return True

    # --- Priority 2: TP3 (cover all remaining) ---
    if bar_low <= trade.tp3 and not trade.tp3_hit and trade.tp3 > 0:
        trade.partial_exits.append((trade.tp3, trade.units_remaining, "TP3"))
        trade.pnl_points += trade.units_remaining * (trade.entry_price - trade.tp3)
        trade.units_remaining = 0
        trade.tp3_hit = True
        trade.exit_bar = bar_idx
        trade.exit_price = trade.tp3
        trade.exit_reason = "TP3"
        return True

    # --- Priority 3: TP2 (cover 1 unit) ---
    if bar_low <= trade.tp2 and not trade.tp2_hit and trade.tp2 > 0 and trade.units_remaining > 0:
        cover_qty = min(1, trade.units_remaining)
        trade.partial_exits.append((trade.tp2, cover_qty, "TP2"))
        trade.pnl_points += cover_qty * (trade.entry_price - trade.tp2)
        trade.units_remaining -= cover_qty
        trade.tp2_hit = True
        trade.tp2_hit_bar = bar_idx

    # --- Priority 4: TP1 (cover 4 units) ---
    if bar_low <= trade.tp1 and not trade.tp1_hit and trade.tp1 > 0 and trade.units_remaining > 0:
        cover_qty = min(4, trade.units_remaining)
        trade.partial_exits.append((trade.tp1, cover_qty, "TP1"))
        trade.pnl_points += cover_qty * (trade.entry_price - trade.tp1)
        trade.units_remaining -= cover_qty
        trade.tp1_hit = True
        trade.tp1_hit_bar = bar_idx

    if trade.units_remaining <= 0:
        trade.exit_bar = bar_idx
        trade.exit_price = trade.tp1
        trade.exit_reason = "TP1+TP2"
        return True

    # --- Update trailing extremes ---
    if trade.tp1_hit and not trade.tp2_hit:
        trade.price_extreme_after_tp1 = min(
            trade.price_extreme_after_tp1 if trade.price_extreme_after_tp1 > 0 else bar_low,
            bar_low,
        )
    if trade.tp2_hit and not trade.tp3_hit:
        trade.price_extreme_after_tp2 = min(
            trade.price_extreme_after_tp2 if trade.price_extreme_after_tp2 > 0 else bar_low,
            bar_low,
        )

    # --- Trailing exit: price faded back to TP2 after exceeding it ---
    if trade.tp2_hit and not trade.tp3_hit and bar_idx > trade.tp2_hit_bar:
        if trade.price_extreme_after_tp2 > 0 and trade.price_extreme_after_tp2 < trade.tp2 and bar_high >= trade.tp2:
            trade.partial_exits.append((trade.tp2, trade.units_remaining, "TRAIL_TP2"))
            trade.pnl_points += trade.units_remaining * (trade.entry_price - trade.tp2)
            trade.units_remaining = 0
            trade.exit_bar = bar_idx
            trade.exit_price = trade.tp2
            trade.exit_reason = "TRAIL_TP2"
            return True

    # --- Trailing exit: price faded back to TP1 after exceeding it ---
    if trade.tp1_hit and not trade.tp2_hit and bar_idx > trade.tp1_hit_bar:
        if trade.price_extreme_after_tp1 > 0 and trade.price_extreme_after_tp1 < trade.tp1 and bar_high >= trade.tp1:
            trade.partial_exits.append((trade.tp1, trade.units_remaining, "TRAIL_TP1"))
            trade.pnl_points += trade.units_remaining * (trade.entry_price - trade.tp1)
            trade.units_remaining = 0
            trade.exit_bar = bar_idx
            trade.exit_price = trade.tp1
            trade.exit_reason = "TRAIL_TP1"
            return True

    return False


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------
def fetch_nifty_data(client) -> pd.DataFrame:
    """Fetch NIFTY daily data for the last N years.

    Tries OpenAlgo API first, falls back to yfinance if server is not running.
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_YEARS * 365)

    print(f"Fetching {SYMBOL} {INTERVAL} data from {start_date} to {end_date}...")

    # Try OpenAlgo API first
    try:
        result = client.history(
            symbol=SYMBOL,
            exchange=EXCHANGE,
            interval=INTERVAL,
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
            print(f"  [OpenAlgo API] {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
            return df
    except Exception as e:
        print(f"  OpenAlgo API unavailable: {e}")

    # Fallback: yfinance
    print("  Falling back to yfinance (^NSEI)...")
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required as fallback. Install: uv add yfinance"
        )

    ticker = yf.Ticker("^NSEI")
    df = ticker.history(start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
    if df.empty:
        raise ValueError("No data from yfinance for ^NSEI")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.sort_index()

    print(f"  [yfinance] {len(df)} bars ({df.index[0].date()} to {df.index[-1].date()})")
    return df


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------
def run_backtest(df: pd.DataFrame, intraday: bool = False) -> tuple:
    """Run the event-driven backtest.

    Args:
        df: OHLCV DataFrame with DatetimeIndex.
        intraday: When True, force-close open positions at each session boundary
                  (date change). Positions never carry across trading sessions.

    Returns:
        (trades: List[Trade], equity_curve: List[DailyPnl])
    """
    print("\nRunning ML CHoCH signal pipeline...")
    signals, model = generate_signals(
        df,
        lookahead=LOOKAHEAD,
        swing_len=SWING_LEN,
        atr_len=ATR_LEN,
        min_score=MIN_SCORE,
        n_trees=N_TREES,
        min_events=MIN_EVENTS,
        window=WINDOW,
        scalar=SCALAR,
    )

    # Get swing structure for SL levels
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    structure = detect_choch(high, low, close, SWING_LEN)

    # Filter to valid signals only
    valid_signals = [s for s in signals if s.is_valid]
    print(f"  Total signals: {len(signals)}, Valid (score >= {MIN_SCORE}%): {len(valid_signals)}")
    print(f"  Bullish: {sum(1 for s in valid_signals if s.direction)}, "
          f"Bearish: {sum(1 for s in valid_signals if not s.direction)}")

    # Build a lookup of signal bar_index for fast access
    signal_map = {s.bar_index: s for s in valid_signals}

    # Run event-driven simulation
    trades: List[Trade] = []
    current_trade: Optional[Trade] = None
    equity = 0.0
    peak_equity = 0.0
    equity_curve: List[DailyPnl] = []

    for i in range(len(df)):
        bar_high = high[i]
        bar_low = low[i]
        bar_close = close[i]

        # Intraday session flatten: force-close at date boundary
        if intraday and current_trade is not None and i > 0:
            prev_date = df.index[i - 1].date()
            curr_date = df.index[i].date()
            if curr_date != prev_date:
                if current_trade.direction:
                    process_bar_long(current_trade, i - 1, high[i-1], low[i-1], close[i-1])
                else:
                    process_bar_short(current_trade, i - 1, high[i-1], low[i-1], close[i-1])
                equity += current_trade.pnl_points * NIFTY_LOT_SIZE
                current_trade.exit_reason = "EOD"
                trades.append(current_trade)
                current_trade = None

        # If in a position, process the bar
        if current_trade is not None:
            if current_trade.direction:
                closed = process_bar_long(current_trade, i, bar_high, bar_low, bar_close)
            else:
                closed = process_bar_short(current_trade, i, bar_high, bar_low, bar_close)

            if closed:
                equity += current_trade.pnl_points * NIFTY_LOT_SIZE
                trades.append(current_trade)
                current_trade = None

        # If flat, check for new entry signal
        if current_trade is None and i in signal_map:
            sig = signal_map[i]

            # Get SL from swing structure
            if sig.direction:
                sl = structure["last_swing_low"][i]
            else:
                sl = structure["last_swing_high"][i]

            # Skip if SL is NaN or invalid
            if np.isnan(sl) or sl <= 0:
                continue

            # Skip if any TP is invalid
            if sig.tp1 <= 0 or sig.tp2 <= 0 or sig.tp3 <= 0:
                continue

            # Validate TP ordering
            entry_p = close[i]
            if sig.direction:
                # Long: TPs must be above entry, tp1 <= tp2 <= tp3
                if not (entry_p < sig.tp1 and sig.tp1 <= sig.tp3):
                    continue
            else:
                # Short: TPs must be below entry, tp1 >= tp2 >= tp3
                if not (entry_p > sig.tp1 and sig.tp1 >= sig.tp3):
                    continue

            current_trade = Trade(
                entry_bar=i,
                entry_price=close[i],
                direction=sig.direction,
                sl_price=sl,
                tp1=sig.tp1,
                tp2=sig.tp2,
                tp3=sig.tp3,
                probability=sig.probability,
                price_extreme_after_tp1=0.0,
                price_extreme_after_tp2=0.0,
            )

        # Record daily equity (mark-to-market)
        unrealized = 0.0
        if current_trade is not None:
            if current_trade.direction:
                unrealized = current_trade.units_remaining * (bar_close - current_trade.entry_price)
            else:
                unrealized = current_trade.units_remaining * (current_trade.entry_price - bar_close)
            unrealized *= NIFTY_LOT_SIZE

        total_equity = equity + unrealized
        peak_equity = max(peak_equity, total_equity)
        dd = (peak_equity - total_equity) if peak_equity > 0 else 0.0

        equity_curve.append(DailyPnl(
            bar_index=i,
            date=str(df.index[i].date()),
            equity=total_equity,
            drawdown=dd,
            trade_pnl=0.0,
        ))

    # Close any remaining position at last close
    if current_trade is not None:
        if current_trade.direction:
            process_bar_long(current_trade, len(df) - 1, close[-1], close[-1], close[-1])
        else:
            process_bar_short(current_trade, len(df) - 1, close[-1], close[-1], close[-1])
        equity += current_trade.pnl_points * NIFTY_LOT_SIZE
        trades.append(current_trade)

    return trades, equity_curve


# ---------------------------------------------------------------------------
# Performance stats
# ---------------------------------------------------------------------------
def compute_stats(trades: List[Trade], equity_curve: List[DailyPnl], df: pd.DataFrame) -> dict:
    """Compute comprehensive backtest statistics."""
    if not trades:
        return {"error": "No trades executed"}

    n_trades = len(trades)
    winners = [t for t in trades if t.pnl_points > 0]
    losers = [t for t in trades if t.pnl_points < 0]
    breakeven = [t for t in trades if t.pnl_points == 0]

    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    total_pnl_points = sum(t.pnl_points for t in trades)
    total_pnl_rupees = total_pnl_points * NIFTY_LOT_SIZE

    avg_win_points = np.mean([t.pnl_points for t in winners]) if winners else 0
    avg_loss_points = np.mean([t.pnl_points for t in losers]) if losers else 0
    avg_trade_points = np.mean([t.pnl_points for t in trades]) if trades else 0

    profit_factor = (
        abs(sum(t.pnl_points for t in winners) / sum(t.pnl_points for t in losers))
        if losers and sum(t.pnl_points for t in losers) != 0
        else float("inf")
    )

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        r = t.exit_reason
        if r not in exit_reasons:
            exit_reasons[r] = {"count": 0, "pnl": 0.0}
        exit_reasons[r]["count"] += 1
        exit_reasons[r]["pnl"] += t.pnl_points * NIFTY_LOT_SIZE

    # Max drawdown from equity curve
    max_dd = max((e.drawdown for e in equity_curve), default=0)

    # Sharpe ratio (annualized, assuming 252 trading days)
    if len(equity_curve) > 1:
        daily_returns = []
        for i in range(1, len(equity_curve)):
            prev_eq = equity_curve[i - 1].equity
            curr_eq = equity_curve[i].equity
            if prev_eq != 0:
                daily_returns.append((curr_eq - prev_eq) / max(abs(prev_eq), 1))
        daily_returns = np.array(daily_returns)
        if daily_returns.std() > 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # Long/short breakdown
    longs = [t for t in trades if t.direction]
    shorts = [t for t in trades if not t.direction]
    long_pnl = sum(t.pnl_points for t in longs) * NIFTY_LOT_SIZE
    short_pnl = sum(t.pnl_points for t in shorts) * NIFTY_LOT_SIZE

    # Average holding period
    holding_bars = [t.exit_bar - t.entry_bar for t in trades if t.exit_bar >= 0]
    avg_hold = np.mean(holding_bars) if holding_bars else 0

    # Buy & hold comparison
    bh_start = close_first = df["close"].iloc[0]
    bh_end = df["close"].iloc[-1]
    bh_return = (bh_end - bh_start) * NIFTY_LOT_SIZE

    # TP hit rate
    tp1_hits = sum(1 for t in trades if t.tp1_hit)
    tp2_hits = sum(1 for t in trades if t.tp2_hit)
    tp3_hits = sum(1 for t in trades if t.tp3_hit)

    return {
        "total_trades": n_trades,
        "winners": len(winners),
        "losers": len(losers),
        "breakeven": len(breakeven),
        "win_rate_pct": round(win_rate, 1),
        "total_pnl_points": round(total_pnl_points, 2),
        "total_pnl_rupees": round(total_pnl_rupees, 2),
        "avg_trade_points": round(avg_trade_points, 2),
        "avg_win_points": round(avg_win_points, 2),
        "avg_loss_points": round(avg_loss_points, 2),
        "profit_factor": round(profit_factor, 2),
        "max_drawdown_rupees": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "long_trades": len(longs),
        "short_trades": len(shorts),
        "long_pnl_rupees": round(long_pnl, 2),
        "short_pnl_rupees": round(short_pnl, 2),
        "avg_holding_bars": round(avg_hold, 1),
        "tp1_hit_rate": round(tp1_hits / n_trades * 100, 1),
        "tp2_hit_rate": round(tp2_hits / n_trades * 100, 1),
        "tp3_hit_rate": round(tp3_hits / n_trades * 100, 1),
        "exit_reasons": exit_reasons,
        "buy_hold_pnl_rupees": round(bh_return, 2),
    }


def print_stats(stats: dict):
    """Print formatted backtest statistics."""
    print("\n" + "=" * 60)
    print("  ML CHoCH NIFTY FUTURES BACKTEST RESULTS")
    print("=" * 60)

    print(f"\n  TOTAL TRADES:       {stats['total_trades']}")
    print(f"  Winners / Losers:   {stats['winners']} / {stats['losers']}")
    print(f"  Win Rate:           {stats['win_rate_pct']}%")

    print(f"\n  --- PNL ---")
    print(f"  Total PNL (pts):    {stats['total_pnl_points']:,.2f}")
    print(f"  Total PNL (INR):    Rs. {stats['total_pnl_rupees']:,.2f}")
    print(f"  Avg Trade (pts):    {stats['avg_trade_points']:,.2f}")
    print(f"  Avg Win (pts):      {stats['avg_win_points']:,.2f}")
    print(f"  Avg Loss (pts):     {stats['avg_loss_points']:,.2f}")
    print(f"  Profit Factor:      {stats['profit_factor']}")
    print(f"  Sharpe Ratio:       {stats['sharpe_ratio']}")

    print(f"\n  --- RISK ---")
    print(f"  Max Drawdown:       Rs. {stats['max_drawdown_rupees']:,.2f}")

    print(f"\n  --- DIRECTION ---")
    print(f"  Long Trades:        {stats['long_trades']}  (PNL: Rs. {stats['long_pnl_rupees']:,.2f})")
    print(f"  Short Trades:       {stats['short_trades']}  (PNL: Rs. {stats['short_pnl_rupees']:,.2f})")

    print(f"\n  --- EXIT ANALYSIS ---")
    print(f"  Avg Holding:        {stats['avg_holding_bars']} bars")
    print(f"  TP1 Hit Rate:       {stats['tp1_hit_rate']}%")
    print(f"  TP2 Hit Rate:       {stats['tp2_hit_rate']}%")
    print(f"  TP3 Hit Rate:       {stats['tp3_hit_rate']}%")
    for reason, data in stats["exit_reasons"].items():
        print(f"    {reason:15s}  {data['count']:3d} trades  Rs. {data['pnl']:>12,.2f}")

    print(f"\n  --- BENCHMARK ---")
    print(f"  Buy & Hold PNL:     Rs. {stats['buy_hold_pnl_rupees']:,.2f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Charting
# ---------------------------------------------------------------------------
def plot_results(
    df: pd.DataFrame,
    trades: List[Trade],
    equity_curve: List[DailyPnl],
    stats: dict,
):
    """Create an interactive Plotly chart with equity curve, drawdown, and trade markers."""
    dates = [e.date for e in equity_curve]
    equities = [e.equity for e in equity_curve]
    drawdowns = [e.drawdown for e in equity_curve]

    fig = make_subplots(
        rows=4, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.45, 0.2, 0.2, 0.15],
        subplot_titles=[
            f"NIFTY {INTERVAL} - ML CHoCH Backtest ({stats['total_trades']} trades, "
            f"PNL Rs. {stats['total_pnl_rupees']:,.0f})",
            "Equity Curve",
            "Drawdown (Rs.)",
            "Volume",
        ],
    )

    # --- Row 1: Candlestick with trade markers ---
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="NIFTY",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ), row=1, col=1)

    # Plot SL levels as dashed lines
    for t in trades:
        color = "#26a69a" if t.direction else "#ef5350"
        # Entry marker
        marker = "triangle-up" if t.direction else "triangle-down"
        entry_date = df.index[t.entry_bar]
        fig.add_trace(go.Scatter(
            x=[entry_date], y=[t.entry_price],
            mode="markers",
            marker=dict(symbol=marker, size=10, color=color),
            name=f"{'Long' if t.direction else 'Short'} Entry",
            showlegend=False,
            hovertext=f"Entry: {t.entry_price:.2f}<br>Prob: {t.probability:.1f}%<br>SL: {t.sl_price:.2f}",
        ), row=1, col=1)

        # Exit marker
        if t.exit_bar >= 0 and t.exit_bar < len(df):
            exit_color = "#26a69a" if t.pnl_points > 0 else "#ef5350"
            exit_date = df.index[t.exit_bar]
            fig.add_trace(go.Scatter(
                x=[exit_date], y=[t.exit_price],
                mode="markers",
                marker=dict(symbol="x", size=8, color=exit_color),
                name=f"Exit ({t.exit_reason})",
                showlegend=False,
                hovertext=f"Exit: {t.exit_price:.2f}<br>PNL: {t.pnl_points * NIFTY_LOT_SIZE:,.0f} INR<br>{t.exit_reason}",
            ), row=1, col=1)

            # Draw TP levels as horizontal segments
            for tp_price, tp_label, tp_color in [
                (t.tp1, "TP1", "rgba(120,200,160,0.5)"),
                (t.tp2, "TP2", "rgba(180,160,255,0.5)"),
                (t.tp3, "TP3", "rgba(220,130,180,0.5)"),
            ]:
                x_start = df.index[t.entry_bar]
                x_end = df.index[min(t.exit_bar, len(df) - 1)] if t.exit_bar >= 0 else df.index[-1]
                fig.add_trace(go.Scatter(
                    x=[x_start, x_end], y=[tp_price, tp_price],
                    mode="lines",
                    line=dict(color=tp_color, width=1, dash="dot"),
                    name=tp_label,
                    showlegend=False,
                    hoverinfo="name+text",
                    hovertext=f"{tp_label}: {tp_price:.2f}",
                ), row=1, col=1)

    # --- Row 2: Equity curve ---
    fig.add_trace(go.Scatter(
        x=dates, y=equities,
        mode="lines",
        name="Equity",
        line=dict(color="#7c4dff", width=2),
        fill="tozeroy",
        fillcolor="rgba(124,77,255,0.1)",
    ), row=2, col=1)

    # --- Row 3: Drawdown ---
    fig.add_trace(go.Scatter(
        x=dates, y=[-d for d in drawdowns],
        mode="lines",
        name="Drawdown",
        line=dict(color="#ef5350", width=1),
        fill="tozeroy",
        fillcolor="rgba(239,83,80,0.15)",
    ), row=3, col=1)

    # --- Row 4: Volume ---
    fig.add_trace(go.Bar(
        x=df.index, y=df["volume"],
        name="Volume",
        marker_color="rgba(100,100,200,0.3)",
    ), row=4, col=1)

    # Layout
    fig.update_layout(
        template="plotly_dark",
        height=900,
        xaxis_rangeslider_visible=False,
        xaxis_type="category",
        showlegend=False,
        margin=dict(l=60, r=30, t=60, b=30),
    )

    # Update axes
    for row in range(1, 5):
        fig.update_xaxes(type="category", row=row, col=1)

    # Save
    out_path = Path(__file__).parent / "backtest_report.html"
    fig.write_html(str(out_path), include_plotlyjs="cdn")
    print(f"\nChart saved to: {out_path}")
    return fig


# ---------------------------------------------------------------------------
# Trade log export
# ---------------------------------------------------------------------------
def export_trades(trades: List[Trade], df: pd.DataFrame):
    """Export trade log to CSV."""
    rows = []
    for t in trades:
        rows.append({
            "entry_date": str(df.index[t.entry_bar].date()) if t.entry_bar < len(df) else "",
            "exit_date": str(df.index[t.exit_bar].date()) if 0 <= t.exit_bar < len(df) else "",
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
            "partial_exits": str(t.partial_exits),
        })

    out_path = Path(__file__).parent / "backtest_trades.csv"
    pd.DataFrame(rows).to_csv(str(out_path), index=False)
    print(f"Trade log saved to: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    load_dotenv(find_dotenv(), override=False)

    client = api(
        api_key=os.getenv("OPENALGO_API_KEY"),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    # Fetch data
    df = fetch_nifty_data(client)

    # Run backtest
    trades, equity_curve = run_backtest(df)

    # Compute and print stats
    stats = compute_stats(trades, equity_curve, df)
    print_stats(stats)

    # Export trades
    if trades:
        export_trades(trades, df)

    # Plot chart
    plot_results(df, trades, equity_curve, stats)


if __name__ == "__main__":
    main()
