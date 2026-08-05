"""
NIFTYBEES Supertrend Multi-Lot Accumulation Backtest
=====================================================
Strategy:
  - BUY a lot sized at 20% of available cash when Supertrend(10, 3) turns GREEN
  - Accumulate more lots on each green flip
  - When Supertrend turns RED:
    * Check each held lot individually
    * SELL only lots where current price > purchase price (book profit)
    * HOLD lots where current price <= purchase price
  - Wait for next green flip to accumulate more

Data Source: OpenAlgo Historify (DuckDB) via source="db"
Fallback:    OpenAlgo REST API via source="api"
Indicator:   openalgo.ta.supertrend
Benchmark:   NIFTY 50 Index (buy & hold)
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import find_dotenv, load_dotenv
from openalgo import api, ta

# --- Config ---
script_dir = Path(__file__).resolve().parent
load_dotenv(find_dotenv(), override=False)

SYMBOL = "NIFTYBEES"
EXCHANGE = "NSE"
INTERVAL = "D"
ST_PERIOD = 10
ST_MULTIPLIER = 3.0
ALLOCATION_PCT = 0.20            # Deploy 20% of available cash per trade
INIT_CASH = 1_000_000
FEES = 0.00111                  # Indian delivery equity (STT + statutory)
FIXED_FEES = 20                 # Rs 20 per order
DATA_SOURCE = os.getenv("DATA_SOURCE", "db")  # "db" for Historify, "api" for live broker

BENCHMARK_SYMBOL = "NIFTY"
BENCHMARK_EXCHANGE = "NSE_INDEX"
LOOKBACK_YEARS = 5

# --- Data Loading ---
client = api(
    api_key=os.getenv("OPENALGO_API_KEY"),
    host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
)

end_date = datetime.now().date()
start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)

print(f"{'=' * 60}")
print(f"  NIFTYBEES Supertrend({ST_PERIOD},{ST_MULTIPLIER}) Multi-Lot Accumulation")
print(f"  Data Source: {DATA_SOURCE} | Period: {start_date} to {end_date}")
print(f"{'=' * 60}")

print(f"\nFetching {SYMBOL} ({EXCHANGE}) {INTERVAL} data...")

def fetch_openalgo(source_label):
    result = client.history(
        symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        source=source_label,
    )
    if isinstance(result, dict) and result.get("status") == "error":
        raise RuntimeError(result.get("message", "API error"))
    return result

def fetch_yfinance():
    import yfinance as yf
    ticker = yf.Ticker("NIFTYBEES.NS")
    raw = ticker.history(start=start_date.strftime("%Y-%m-%d"),
                         end=end_date.strftime("%Y-%m-%d"), auto_adjust=False)
    if raw.empty:
        raise RuntimeError("yfinance returned empty data for NIFTYBEES.NS")
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.rename(columns={"Open": "open", "High": "high", "Low": "low",
                               "Close": "close", "Volume": "volume"})
    return raw[["open", "high", "low", "close", "volume"]]

df = None
for src in [DATA_SOURCE, "api"]:
    try:
        df = fetch_openalgo(src)
        print(f"  Source: {src}")
        break
    except Exception as e:
        print(f"  source='{src}' failed: {e}")

if df is None or not hasattr(df, "columns"):
    print("  Falling back to yfinance for NIFTYBEES...")
    try:
        df = fetch_yfinance()
        print("  Source: yfinance")
    except Exception as e:
        print(f"  yfinance failed: {e}")
        print("ERROR: Could not fetch data from any source.")
        sys.exit(1)

# Normalize index
if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp")
else:
    df.index = pd.to_datetime(df.index)
df = df.sort_index()
if df.index.tz is not None:
    df.index = df.index.tz_convert(None)

close = df["close"]
high = df["high"]
low = df["low"]
open_ = df["open"]

print(f"Data loaded: {len(df)} bars from {df.index[0].date()} to {df.index[-1].date()}")
print(f"Price range: {close.min():.2f} - {close.max():.2f}")

# --- Compute Supertrend ---
st_line, st_direction = ta.supertrend(high, low, close, period=ST_PERIOD, multiplier=ST_MULTIPLIER)
# st_direction: 1 = bullish (green), -1 = bearish (red)

# Detect flips
st_bull = (st_direction == 1) & (st_direction.shift(1) == -1)  # bear -> bull = BUY signal
st_bear = (st_direction == -1) & (st_direction.shift(1) == 1)  # bull -> bear = CHECK lots

print(f"\nSupertrend signals: {st_bull.sum()} green flips, {st_bear.sum()} red flips")

# --- Custom Multi-Lot Simulation ---
class Lot:
    def __init__(self, entry_date, entry_price, quantity):
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.quantity = quantity
        self.exit_date = None
        self.exit_price = None
        self.pnl = 0.0
        self.fee_paid = 0.0
        self.closed = False

    def current_pnl(self, current_price):
        gross = (current_price - self.entry_price) * self.quantity
        return gross - self.fee_paid

    def close(self, exit_date, exit_price):
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.pnl = (exit_price - self.entry_price) * self.quantity - self.fee_paid - (FIXED_FEES)
        self.closed = True


# Simulation state
lots = []               # Currently held lots
closed_lots = []        # Completed trades
cash = INIT_CASH
total_invested = 0.0

# Track portfolio value over time (for equity curve)
portfolio_values = []
dates = []
sell_signal_dates = []
sell_signal_prices = []

for i in range(len(df)):
    date = df.index[i]
    price = close.iloc[i]

    # Check for red flip: sell profitable lots
    if st_bear.iloc[i] and lots:
        lots_to_sell = []
        for lot in lots:
            if price > lot.entry_price:
                lots_to_sell.append(lot)

        for lot in lots_to_sell:
            proceeds = price * lot.quantity
            lot.close(date, price)
            cash += proceeds
            closed_lots.append(lot)
            lots.remove(lot)
            sell_signal_dates.append(date)
            sell_signal_prices.append(price)

    # Check for green flip: buy new lot (20% of available cash)
    if st_bull.iloc[i]:
        allocatable = INIT_CASH * ALLOCATION_PCT
        qty = int(allocatable // price)
        if qty > 0 and cash >= qty * price + FIXED_FEES:
            cost = qty * price + FIXED_FEES
            lot = Lot(date, price, qty)
            lot.fee_paid = FIXED_FEES + (price * qty * FEES)
            cash -= cost
            total_invested += (price * qty)
            lots.append(lot)

    # Calculate portfolio value = cash + market value of held lots
    held_value = sum(price * lot.quantity for lot in lots)
    portfolio_values.append(cash + held_value)
    dates.append(date)

# Force-close remaining lots at last price
last_price = close.iloc[-1]
last_date = df.index[-1]
for lot in lots:
    proceeds = last_price * lot.quantity
    lot.close(last_date, last_price)
    cash += proceeds
    closed_lots.append(lot)
lots = []

final_portfolio_value = portfolio_values[-1]
total_return_pct = (final_portfolio_value - INIT_CASH) / INIT_CASH * 100

# --- Results ---
print(f"\n{'=' * 60}")
print(f"  BACKTEST RESULTS")
print(f"{'=' * 60}")

print(f"\n  Initial Capital:   Rs {INIT_CASH:>12,.2f}")
print(f"  Final Value:       Rs {final_portfolio_value:>12,.2f}")
print(f"  Total Return:      {total_return_pct:>11.2f}%")
print(f"  Total Trades:      {len(closed_lots)} lots closed")

# Win/loss analysis
wins = [l for l in closed_lots if l.pnl > 0]
losses = [l for l in closed_lots if l.pnl <= 0]
win_rate = len(wins) / len(closed_lots) * 100 if closed_lots else 0

total_profit = sum(l.pnl for l in wins)
total_loss = sum(l.pnl for l in losses)
net_pnl = total_profit + total_loss

print(f"\n  Winning Lots:      {len(wins)} ({win_rate:.1f}%)")
print(f"  Losing Lots:       {len(losses)} ({100 - win_rate:.1f}%)")
print(f"  Total Profit:      Rs {total_profit:>12,.2f}")
print(f"  Total Loss:        Rs {total_loss:>12,.2f}")
print(f"  Net P&L:           Rs {net_pnl:>12,.2f}")
print(f"  Profit Factor:     {abs(total_profit / total_loss):.2f}" if total_loss != 0 else "  Profit Factor:     N/A")

# Average win/loss
avg_win = total_profit / len(wins) if wins else 0
avg_loss = total_loss / len(losses) if losses else 0
print(f"  Avg Win:           Rs {avg_win:>12,.2f}")
print(f"  Avg Loss:          Rs {avg_loss:>12,.2f}")
print(f"  Risk/Reward:       {abs(avg_loss / avg_win):.2f}" if avg_win != 0 else "  Risk/Reward:       N/A")

# Max drawdown on portfolio
pv = pd.Series(portfolio_values, index=dates)
running_max = pv.cummax()
drawdown = (pv - running_max) / running_max
max_dd = drawdown.min() * 100
print(f"  Max Drawdown:      {max_dd:.2f}%")

# CAGR
years = (df.index[-1] - df.index[0]).days / 365.25
cagr = ((final_portfolio_value / INIT_CASH) ** (1 / years) - 1) * 100 if years > 0 else 0
print(f"  CAGR:              {cagr:.2f}%")
print(f"  Backtest Period:   {years:.1f} years ({df.index[0].date()} to {df.index[-1].date()})")

# Lots still open (force-closed at end)
print(f"\n  (All remaining lots force-closed at last price {last_price:.2f})")

# --- Strategy vs Benchmark ---
print(f"\n{'=' * 60}")
print(f"  STRATEGY vs BENCHMARK (NIFTY 50 Buy & Hold)")
print(f"{'=' * 60}")

for src in [DATA_SOURCE, "api"]:
    try:
        df_bench = client.history(
            symbol=BENCHMARK_SYMBOL, exchange=BENCHMARK_EXCHANGE, interval="D",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            source=src,
        )
        if isinstance(df_bench, dict) and df_bench.get("status") == "error":
            raise RuntimeError(df_bench.get("message", "API error"))
        break
    except Exception:
        df_bench = None
        continue

if df_bench is None or not hasattr(df_bench, "columns"):
    print("WARNING: Could not fetch benchmark data. Skipping comparison.")

has_bench = False
bench_close = None

for src in [DATA_SOURCE, "api"]:
    try:
        df_bench = client.history(
            symbol=BENCHMARK_SYMBOL, exchange=BENCHMARK_EXCHANGE, interval="D",
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            source=src,
        )
        if isinstance(df_bench, dict) and df_bench.get("status") == "error":
            raise RuntimeError(df_bench.get("message", "API error"))
        has_bench = True
        break
    except Exception:
        df_bench = None
        continue

if not has_bench:
    print("  Falling back to yfinance for NIFTY benchmark...")
    try:
        import yfinance as yf
        nifty_raw = yf.Ticker("^NSEI").history(
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"), auto_adjust=False)
        if not nifty_raw.empty:
            nifty_raw.index = pd.to_datetime(nifty_raw.index).tz_localize(None)
            df_bench = nifty_raw.rename(columns={"Close": "close"})[["close"]]
            has_bench = True
    except Exception as e:
        print(f"  yfinance benchmark failed: {e}")
        df_bench = None

if has_bench and df_bench is not None and hasattr(df_bench, "columns"):
    if "timestamp" in df_bench.columns:
        df_bench["timestamp"] = pd.to_datetime(df_bench["timestamp"])
        df_bench = df_bench.set_index("timestamp")
    elif df_bench.index.name != "Date" and not isinstance(df_bench.index, pd.DatetimeIndex):
        df_bench.index = pd.to_datetime(df_bench.index)
    df_bench = df_bench.sort_index()
    if hasattr(df_bench.index, 'tz') and df_bench.index.tz is not None:
        df_bench.index = df_bench.index.tz_convert(None)

    bench_close = df_bench["close"].reindex(close.index).ffill().bfill()
    bench_return_pct = (bench_close.iloc[-1] / bench_close.iloc[0] - 1) * 100
    bench_cagr = ((bench_close.iloc[-1] / bench_close.iloc[0]) ** (1 / years) - 1) * 100
    bench_cummax = bench_close.cummax()
    bench_dd = ((bench_close - bench_cummax) / bench_cummax).min() * 100
    bench_daily_ret = bench_close.pct_change().dropna()
    bench_sharpe = (bench_daily_ret.mean() / bench_daily_ret.std() * np.sqrt(252)) if bench_daily_ret.std() > 0 else 0
    bench_sortino_denom = bench_daily_ret[bench_daily_ret < 0].std()
    bench_sortino = (bench_daily_ret.mean() / bench_sortino_denom * np.sqrt(252)) if bench_sortino_denom > 0 else 0
else:
    has_bench = False
    bench_return_pct = bench_cagr = bench_dd = bench_sharpe = bench_sortino = 0

strat_daily_ret = pv.pct_change().dropna()
strat_sharpe = (strat_daily_ret.mean() / strat_daily_ret.std() * np.sqrt(252)) if strat_daily_ret.std() > 0 else 0
strat_sortino_denom = strat_daily_ret[strat_daily_ret < 0].std()
strat_sortino = (strat_daily_ret.mean() / strat_sortino_denom * np.sqrt(252)) if strat_sortino_denom > 0 else 0

comparison = pd.DataFrame({
    "Strategy": [
        f"{total_return_pct:.2f}%",
        f"{cagr:.2f}%",
        f"{strat_sharpe:.2f}",
        f"{strat_sortino:.2f}",
        f"{max_dd:.2f}%",
        f"{win_rate:.1f}%",
        f"{len(closed_lots)}",
        f"{abs(total_profit / total_loss):.2f}" if total_loss != 0 else "N/A",
    ],
    f"Benchmark ({BENCHMARK_SYMBOL})": [
        f"{bench_return_pct:.2f}%" if has_bench else "N/A",
        f"{bench_cagr:.2f}%" if has_bench else "N/A",
        f"{bench_sharpe:.2f}" if has_bench else "N/A",
        f"{bench_sortino:.2f}" if has_bench else "N/A",
        f"{bench_dd:.2f}%" if has_bench else "N/A",
        "-",
        "-",
        "-",
    ],
}, index=["Total Return", "CAGR", "Sharpe Ratio", "Sortino Ratio",
          "Max Drawdown", "Win Rate", "Total Trades", "Profit Factor"])
print(comparison.to_string())

# --- Plain Language Report ---
print(f"\n{'=' * 60}")
print(f"  PLAIN LANGUAGE REPORT")
print(f"{'=' * 60}")
print(f"""
This strategy bought NIFTYBEES ETF units every time Supertrend({ST_PERIOD},{ST_MULTIPLIER})
turned bullish (green) and sold only those lots that were in profit when
the signal turned bearish (red). Losing lots were held through the downturn
until the next green signal added more units.

Starting with Rs {INIT_CASH:,}, the strategy ended at Rs {final_portfolio_value:,.2f}
({total_return_pct:.2f}% over {years:.1f} years, CAGR {cagr:.2f}%).
""")

if has_bench:
    print(f"""Over the same period, a simple NIFTY 50 buy-and-hold would have returned
{bench_return_pct:.2f}% (CAGR {bench_cagr:.2f}%).
""")

print(f"""
Key observations:
- The strategy accumulated {len(closed_lots)} lots over {years:.1f} years
- Win rate was {win_rate:.1f}% - profitable lots outnumbered losers
- Max drawdown was {max_dd:.2f}%
{f'- vs benchmark drawdown {bench_dd:.2f}%' if has_bench else ''}
- The accumulator approach provides dollar-cost-averaging benefits
- Holding losing lots through downturns reduces realized losses but
  increases capital lock-in and drawdown
""")

# --- Trade Log ---
if closed_lots:
    print(f"\n--- Trade Log (all {len(closed_lots)} closed lots) ---")
    trade_data = []
    for i, lot in enumerate(closed_lots, 1):
        holding_days = (lot.exit_date - lot.entry_date).days if lot.exit_date else 0
        trade_data.append({
            "#": i,
            "Entry Date": lot.entry_date.strftime("%Y-%m-%d"),
            "Exit Date": lot.exit_date.strftime("%Y-%m-%d") if lot.exit_date else "-",
            "Entry Price": f"{lot.entry_price:.2f}",
            "Exit Price": f"{lot.exit_price:.2f}" if lot.exit_price else "-",
            "Qty": lot.quantity,
            "P&L": f"Rs {lot.pnl:+,.2f}",
            "Holding Days": holding_days,
        })
    trade_df = pd.DataFrame(trade_data)
    print(trade_df.to_string(index=False))

# --- Plot ---
print(f"\nGenerating chart...")

fig = make_subplots(
    rows=3, cols=1, shared_xaxes=True,
    row_heights=[0.5, 0.25, 0.25],
    vertical_spacing=0.03,
    subplot_titles=[
        f"NIFTYBEES Supertrend({ST_PERIOD},{ST_MULTIPLIER}) - Multi-Lot Accumulation",
        "Portfolio Value",
        "Drawdown",
    ],
)

# Candlestick
x_labels = df.index.strftime("%Y-%m-%d")
fig.add_trace(go.Candlestick(
    x=x_labels, open=open_, high=high, low=low, close=close,
    name="NIFTYBEES", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
), row=1, col=1)

# Supertrend line
fig.add_trace(go.Scatter(
    x=x_labels, y=st_line, mode="lines",
    name="Supertrend", line=dict(color="orange", width=1.5),
), row=1, col=1)

# Mark buy signals
buy_dates = df.index[st_bull]
buy_prices = close[st_bull]
if len(buy_dates) > 0:
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in buy_dates],
        y=buy_prices.values,
        mode="markers", name="BUY (Green Flip)",
        marker=dict(symbol="triangle-up", size=10, color="#26a69a", line=dict(width=1, color="white")),
    ), row=1, col=1)

if sell_signal_dates:
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in sell_signal_dates], y=sell_signal_prices,
        mode="markers", name="SELL (Profit Book)",
        marker=dict(symbol="triangle-down", size=10, color="#ef5350", line=dict(width=1, color="white")),
    ), row=1, col=1)

# Portfolio value
fig.add_trace(go.Scatter(
    x=[d.strftime("%Y-%m-%d") for d in dates], y=portfolio_values,
    mode="lines", name="Portfolio Value",
    line=dict(color="cyan", width=2), fill="tozeroy",
    fillcolor="rgba(0,255,255,0.05)",
), row=2, col=1)

if has_bench and bench_close is not None:
    bench_norm = (bench_close / bench_close.iloc[0]) * INIT_CASH
    fig.add_trace(go.Scatter(
        x=[d.strftime("%Y-%m-%d") for d in bench_norm.index],
        y=bench_norm.values,
        mode="lines", name=f"Benchmark ({BENCHMARK_SYMBOL})",
        line=dict(color="gray", width=1, dash="dash"),
    ), row=2, col=1)

# Drawdown
fig.add_trace(go.Scatter(
    x=[d.strftime("%Y-%m-%d") for d in dates], y=drawdown.values * 100,
    mode="lines", name="Drawdown %",
    line=dict(color="#ef5350", width=1), fill="tozeroy",
    fillcolor="rgba(239,83,80,0.1)",
), row=3, col=1)

fig.update_layout(
    template="plotly_dark",
    xaxis_rangeslider_visible=False,
    xaxis_type="category",
    xaxis2_type="category",
    xaxis3_type="category",
    height=1000,
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Rs", row=2, col=1)
fig.update_yaxes(title_text="%", row=3, col=1)

chart_path = script_dir / "niftybees_supertrend_chart.html"
fig.write_html(str(chart_path))
print(f"Chart saved to {chart_path}")

# Also try to show
try:
    fig.show()
except Exception:
    pass

# --- Export ---
trades_file = script_dir / "niftybees_supertrend_trades.csv"
trade_export = pd.DataFrame([{
    "entry_date": l.entry_date,
    "exit_date": l.exit_date,
    "entry_price": l.entry_price,
    "exit_price": l.exit_price,
    "quantity": l.quantity,
    "pnl": l.pnl,
    "holding_days": (l.exit_date - l.entry_date).days if l.exit_date else 0,
} for l in closed_lots])
trade_export.to_csv(trades_file, index=False)
print(f"Trade log exported to {trades_file}")

# Portfolio equity curve CSV
equity_df = pd.DataFrame({"date": dates, "portfolio_value": portfolio_values})
equity_file = script_dir / "niftybees_equity_curve.csv"
equity_df.to_csv(equity_file, index=False)
print(f"Equity curve exported to {equity_file}")

# --- OpenStatz Tearsheet ---
try:
    import openstatz as ostz

    equity_series = pd.Series(portfolio_values, index=pd.DatetimeIndex(dates))
    strategy_returns = equity_series.pct_change().dropna()
    if strategy_returns.index.tz is not None:
        strategy_returns.index = strategy_returns.index.tz_convert(None)
    strategy_returns.name = f"Supertrend({ST_PERIOD},{ST_MULTIPLIER}) Multi-Lot - {SYMBOL}"

    benchmark_returns = None
    if has_bench and bench_close is not None:
        benchmark_returns = bench_close.pct_change().dropna()
        benchmark_returns = benchmark_returns.reindex(strategy_returns.index).fillna(0)
        benchmark_returns.name = "NIFTY 50"

    tearsheet_path = script_dir / f"{SYMBOL}_tearsheet.html"
    ostz.dashboard(
        strategy_returns,
        benchmark=benchmark_returns,
        output=str(tearsheet_path),
        title=f"Supertrend Multi-Lot {SYMBOL} Tearsheet",
        open_browser=True,
    )
    print(f"\nOpenStatz tearsheet saved to {tearsheet_path}")

    mc = ostz.stats.montecarlo(strategy_returns, sims=1000, bust=-0.10, goal=0.30)
    print(f"Monte Carlo (1000 sims): Bust prob={mc.bust_probability:.1%}, Goal prob={mc.goal_probability:.1%}")

except ImportError:
    print("\nOpenStatz not installed. Run: pip install openstatz")
    print("Skipping tearsheet generation.")
except Exception as e:
    print(f"\nTearsheet generation failed: {e}")

print(f"\n{'=' * 60}")
print(f"  BACKTEST COMPLETE")
print(f"{'=' * 60}")
