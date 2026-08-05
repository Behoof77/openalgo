#!/usr/bin/env python3
"""1-year 1m backtest of XGBoost reversal strategy using Historify DuckDB."""

import os, json, warnings, sys, calendar
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import duckdb
import numpy as np
import pandas as pd
import vectorbt as vbt
from openalgo import ta

warnings.filterwarnings("ignore")

load_dotenv(find_dotenv(), override=False)

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs" / "xgb_backtest_1yr_1m"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
LOOKBACK = 500      # bars to train each model
RETRAIN_EVERY = 100 # retrain interval
PROB_THRESHOLD = 0.60
ADX_THRESHOLD = 22
ATR_PERIOD = 14
ATR_MULT_SL = 2.0
ATR_MULT_TP = 3.0
INIT_CASH = 30_000_000  # ~1Cr, enough for 1 NIFTY futures lot at ~24K
FEES = 0.00018      # F&O futures fees
FIXED_FEES = 20     # per order
MIN_SIZE = 65       # NIFTY lot size
DIRECTION = "shortonly"

HISTORIFY_DB = os.getenv("HISTORIFY_DUCKDB_PATH")

if not HISTORIFY_DB:
    print("ERROR: HISTORIFY_DUCKDB_PATH not set!")
    print("Hint: add to .env like: HISTORIFY_DUCKDB_PATH=/home/ubuntu/openalgo/db/historify.duckdb")
    sys.exit(1)

# ============================================================
# DATA LOADING
# ============================================================
print("=== Loading 1m NIFTY data from Historify DuckDB ===")
con = duckdb.connect(HISTORIFY_DB, read_only=True)
df = con.execute("""
    SELECT 
        to_timestamp(timestamp) AT TIME ZONE 'Asia/Kolkata' AS ts,
        open, high, low, close, volume
    FROM market_data
    WHERE symbol = 'NIFTY'
      AND exchange = 'NSE_INDEX'
      AND interval = '1m'
      AND timestamp >= EXTRACT(EPOCH FROM TIMESTAMP '2025-06-01')
      AND timestamp < EXTRACT(EPOCH FROM TIMESTAMP '2026-07-01')
    ORDER BY timestamp
""").fetchdf()
con.close()

print(f"Raw rows: {len(df)}")

if len(df) == 0:
    print("No data found!")
    sys.exit(1)

# Clean timestamps
df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
df = df.drop_duplicates(subset="ts").sort_values("ts").set_index("ts")
df = df[["open", "high", "low", "close", "volume"]].dropna()
print(f"Clean rows: {len(df)}, date range: {df.index[0]} to {df.index[-1]}")

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n=== Computing features ===")
close = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]

features = pd.DataFrame(index=df.index)

# RSI(14)
features["rsi14"] = ta.rsi(close, 14)

# Relative Volume (20-period) — safe for index data (volume may be 0)
avg_vol = volume.rolling(20).mean()
rel_vol = volume / avg_vol.replace(0, np.nan)
features["rel_vol"] = rel_vol.fillna(1.0)

# ZScore of close (20-period)
features["zscore20"] = (close - close.rolling(20).mean()) / close.rolling(20).std().replace(0, np.nan)

# ADX(14) — ta.adx returns (ADX, +DI, -DI) as a tuple of arrays
adx_val = ta.adx(high, low, close, 14)
features["adx14"] = adx_val[0] if isinstance(adx_val, (tuple, list)) else adx_val

# ATR(14)
features["atr14"] = ta.atr(high, low, close, 14)

# Pivot-based features (swing highs/lows over 10 bars)
features["pivot_high"] = (high == high.rolling(5, center=True).max()).astype(float)
features["pivot_low"] = (low == low.rolling(5, center=True).min()).astype(float)

# Rolling std
features["std20"] = close.rolling(20).std()

features = features.dropna()
print(f"Features shape: {features.shape}")

# ============================================================
# TARGET ENGINEERING
# ============================================================
print("\n=== Building targets ===")

# Forward returns
fwd_5 = close.pct_change(5).shift(-5)  # 5-bar forward return
fwd_10 = close.pct_change(10).shift(-10)  # 10-bar forward return

# Classification targets
target_cls_long = (fwd_10 > 0.005).astype(int).astype(float)  # 0.5% up in 10 bars
target_cls_short = (fwd_10 < -0.005).astype(int).astype(float)  # 0.5% down in 10 bars

# Regression targets
target_reg_long = fwd_5.clip(-0.02, 0.02)  # clipped 5-bar forward return
target_reg_short = -fwd_5.clip(-0.02, 0.02)

# Align all data
data = features.join(target_cls_long.rename("tgt_cls_long"))
data = data.join(target_cls_short.rename("tgt_cls_short"))
data = data.join(target_reg_long.rename("tgt_reg_long"))
data = data.join(target_reg_short.rename("tgt_reg_short"))
data = data.dropna()
print(f"Data shape: {data.shape}")

feature_cols = ["rsi14", "rel_vol", "zscore20", "adx14", "atr14", "pivot_high", "pivot_low", "std20"]
print(f"Features: {feature_cols}")
print(f"Date range: {data.index[0]} to {data.index[-1]}")
print(f"  Long targets (>=0.5% up): {data['tgt_cls_long'].sum():.0f}/{len(data)}")
print(f"  Short targets (>=0.5% down): {data['tgt_cls_short'].sum():.0f}/{len(data)}")

# ============================================================
# STRATEGY BACKTEST (Walk-forward XGBoost)
# ============================================================
print(f"\n=== Running walk-forward backtest ===")
print(f"Lookback: {LOOKBACK}, Retrain every: {RETRAIN_EVERY}, Prob threshold: {PROB_THRESHOLD}, ADX threshold: {ADX_THRESHOLD}")

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    print("XGBoost not installed. Install with: uv add xgboost")
    sys.exit(1)

all_signals = []
models_trained = 0
total_windows = 0

for i in range(LOOKBACK, len(data), RETRAIN_EVERY):
    train_end = i
    pred_start = i
    pred_end = min(i + RETRAIN_EVERY, len(data))
    
    train_data = data.iloc[train_end - LOOKBACK:train_end]
    pred_data = data.iloc[pred_start:pred_end]
    
    if len(pred_data) == 0:
        continue
    
    # Filter by ADX threshold for training (trending regime)
    train_adx_high = train_data[train_data["adx14"] >= ADX_THRESHOLD]
    
    total_windows += 1
    
    # Skip if not enough training data
    if len(train_adx_high) < 50:
        # Use prediction data with neutral signal
        for idx in pred_data.index:
            all_signals.append({
                "timestamp": idx, "signal": 0,
                "prob_long": 0.5, "prob_short": 0.5,
                "pred_return": 0.0
            })
        continue
    
    X_train = train_adx_high[feature_cols]
    y_train_cls_long = train_adx_high["tgt_cls_long"]
    y_train_cls_short = train_adx_high["tgt_cls_short"]
    y_train_reg_long = train_adx_high["tgt_reg_long"]
    y_train_reg_short = train_adx_high["tgt_reg_short"]
    
    # Train classifiers
    cls_long = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=42, verbosity=0, use_label_encoder=False,
                             eval_metric="logloss")
    cls_long.fit(X_train, y_train_cls_long)
    
    cls_short = XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1,
                              random_state=42, verbosity=0, use_label_encoder=False,
                              eval_metric="logloss")
    cls_short.fit(X_train, y_train_cls_short)
    
    # Train regressors
    reg_long = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                            random_state=42, verbosity=0)
    reg_long.fit(X_train, y_train_reg_long)
    
    reg_short = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1,
                             random_state=42, verbosity=0)
    reg_short.fit(X_train, y_train_reg_short)
    
    models_trained += 1
    
    # Predict
    X_pred = pred_data[feature_cols]
    
    probL = cls_long.predict_proba(X_pred)[:, 1]
    probS = cls_short.predict_proba(X_pred)[:, 1]
    predR_L = reg_long.predict(X_pred)
    predR_S = reg_short.predict(X_pred)
    
    for j, idx in enumerate(pred_data.index):
        adx = pred_data.loc[idx, "adx14"]
        signal = 0
        if probS[j] >= PROB_THRESHOLD and adx >= ADX_THRESHOLD:
            signal = -1  # short
        elif probL[j] >= PROB_THRESHOLD and adx >= ADX_THRESHOLD:
            signal = 1   # long
        
        all_signals.append({
            "timestamp": idx, "signal": signal,
            "prob_long": float(probL[j]),
            "prob_short": float(probS[j]),
            "pred_return": float(predR_S[j] if signal == -1 else predR_L[j]),
            "adx": float(adx)
        })
    
    if total_windows % 5 == 0:
        print(f"  Window {total_windows}: pred {pred_data.index[0].strftime('%Y-%m-%d')} to {pred_data.index[-1].strftime('%Y-%m-%d')}, "
              f"signals={len([s for s in all_signals[-len(pred_data):] if s['signal'] != 0])}")

print(f"\nModels trained: {models_trained}, Total windows: {total_windows}")
print(f"Total signal rows: {len(all_signals)}")

# Build entries/exits
sig_df = pd.DataFrame(all_signals).set_index("timestamp")
sig_df = sig_df.reindex(data.index).fillna({"signal": 0, "prob_long": 0.5, "prob_short": 0.5})

if DIRECTION == "shortonly":
    entries = (sig_df["signal"] == -1).values
    exits = pd.Series(False, index=sig_df.index).values
    exit_on_signal = pd.Series(False, index=sig_df.index).values
elif DIRECTION == "longonly":
    entries = (sig_df["signal"] == 1).values
    exits = pd.Series(False, index=sig_df.index).values
    exit_on_signal = pd.Series(False, index=sig_df.index).values
else:
    entries = (sig_df["signal"] != 0).values
    exits = pd.Series(False, index=sig_df.index).values
    exit_on_signal = pd.Series(False, index=sig_df.index).values

n_entries = entries.sum()
print(f"Entry signals: {n_entries} / {len(entries)} ({100*n_entries/len(entries):.1f}%)")

if n_entries == 0:
    print("No signals generated. Try lowering PROB_THRESHOLD or ADX_THRESHOLD.")
    # Save partial output
    sig_df.to_csv(output_dir / "signals.csv")
    signal_counts = sig_df["signal"].value_counts().to_dict()
    with open(output_dir / "summary.json", "w") as f:
        json.dump({"total_rows": len(data), "signals": int(n_entries),
                   "models_trained": models_trained, "signal_counts": {
                       str(k): int(v) for k, v in signal_counts.items()}}, f, indent=2)
    print(f"Partial output saved to {output_dir}")
    sys.exit(0)

# ============================================================
# VECTORBT BACKTEST
# ============================================================
print("\n=== Running VectorBT backtest (1m) ===")

close_ts = close.reindex(data.index)
high_ts = high.reindex(data.index)
low_ts = low.reindex(data.index)

# SL/TP based on ATR
atr_vals = features["atr14"].reindex(data.index)

# For NIFTY index (value ~24000), ATR of 1m bar is small
# But we want %-based SL/TP to be meaningful
use_pct_sl = True

if use_pct_sl:
    # Use percentage-based SL/TP
    sl_pct = 0.002  # 0.2%
    tp_pct = 0.004  # 0.4%
    
    if DIRECTION == "shortonly":
        sl = close_ts * (1 + sl_pct)
        tp = close_ts * (1 - tp_pct)
    elif DIRECTION == "longonly":
        sl = close_ts * (1 - sl_pct)
        tp = close_ts * (1 + tp_pct)
    else:
        sl = close_ts * (1 - sl_pct)
        tp = close_ts * (1 + tp_pct)
else:
    sl = atr_vals * ATR_MULT_SL
    tp = atr_vals * ATR_MULT_TP

# For short-only: direction = -1 (short), entries = short signals
# For long-only: direction = 1 (long), entries = long signals
pf_kwargs = dict(
    init_cash=INIT_CASH,
    size=1,
    size_type="value",
    fees=FEES,
    fixed_fees=FIXED_FEES,
    min_size=MIN_SIZE,
    size_granularity=MIN_SIZE,
    freq="1min",
)

if DIRECTION == "shortonly":
    pf = vbt.Portfolio.from_signals(
        close_ts, entries, exits,
        direction="both",
        sl_stop=sl.values,
        tp_stop=tp.values,
        **pf_kwargs,
    )
elif DIRECTION == "longonly":
    pf = vbt.Portfolio.from_signals(
        close_ts, entries, exits,
        sl_stop=sl.values,
        tp_stop=tp.values,
        **pf_kwargs,
    )
else:
    pf = vbt.Portfolio.from_signals(
        close_ts, entries, exits,
        sl_stop=sl.values,
        tp_stop=tp.values,
        **pf_kwargs,
    )

# Stats
ret = pf.total_return()
sharpe = pf.sharpe_ratio()
sortino = pf.sortino_ratio()
max_dd = pf.max_drawdown()
win_rate = pf.trades.win_rate()
n_trades = pf.trades.count()
profit_factor = pf.trades.profit_factor()

print(f"\n=== Backtest Results ===")
print(f"Total Return: {ret*100:.2f}%")
print(f"Sharpe Ratio: {sharpe:.2f}")
print(f"Sortino Ratio: {sortino:.2f}")
print(f"Max Drawdown: {max_dd*100:.2f}%")
print(f"Win Rate: {win_rate*100:.1f}%")
print(f"Total Trades: {n_trades}")
print(f"Profit Factor: {profit_factor:.2f}")

# Benchmark (NIFTY buy & hold)
pf_bench = vbt.Portfolio.from_holding(close_ts, init_cash=INIT_CASH, fees=FEES, freq="1min")
bench_ret = pf_bench.total_return()

# Comparison table
comparison = pd.DataFrame({
    "Strategy": [
        f"{ret*100:.2f}%", f"{sharpe:.2f}", f"{sortino:.2f}",
        f"{max_dd*100:.2f}%", f"{win_rate*100:.1f}%", f"{n_trades}", f"{profit_factor:.2f}",
    ],
    "NIFTY BuyHold": [
        f"{bench_ret*100:.2f}%", f"{pf_bench.sharpe_ratio():.2f}", f"{pf_bench.sortino_ratio():.2f}",
        f"{pf_bench.max_drawdown()*100:.2f}%", "-", "-", "-",
    ],
}, index=["Total Return", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown",
          "Win Rate", "Total Trades", "Profit Factor"])
print(f"\n{comparison.to_string()}")

print(f"\n--- Plain Language ---")
print(f"* Return: {ret*100:.2f}% vs NIFTY {bench_ret*100:.2f}%")
print(f"* Max Drawdown: {max_dd*100:.2f}% (worst loss Rs {abs(max_dd)*INIT_CASH:,.0f})")
print(f"* {n_trades} trades with {win_rate*100:.1f}% win rate")
print(f"* Strategy trades only during ADX>=22 trending regimes")
print(f"* {DIRECTION} mode with {PROB_THRESHOLD} prob threshold")

# ============================================================
# EXPORT
# ============================================================
print(f"\n=== Saving outputs to {output_dir} ===")
trades_df = pf.trades.records_readable
trades_df.to_csv(output_dir / "trades.csv", index=False)

# Equity curve
equity = pf.value()
equity.name = "strategy"
bench_equity = pf_bench.value()
bench_equity.name = "benchmark"
equity_df = pd.concat([equity, bench_equity], axis=1)
equity_df.to_csv(output_dir / "equity.csv")

# Config
config = {
    "symbol": "NIFTY", "exchange": "NSE_INDEX", "interval": "1m",
    "data_source": "Historify_DuckDB", "data_path": HISTORIFY_DB,
    "lookback": LOOKBACK, "retrain_every": RETRAIN_EVERY,
    "prob_threshold": PROB_THRESHOLD, "adx_threshold": ADX_THRESHOLD,
    "direction": DIRECTION, "init_cash": INIT_CASH,
    "fees": FEES, "fixed_fees": FIXED_FEES,
    "sl_pct": sl_pct if use_pct_sl else f"atr*{ATR_MULT_SL}",
    "tp_pct": tp_pct if use_pct_sl else f"atr*{ATR_MULT_TP}",
    "total_return_pct": float(ret*100),
    "sharpe_ratio": float(sharpe) if sharpe else None,
    "sortino_ratio": float(sortino) if sortino else None,
    "max_drawdown_pct": float(max_dd*100),
    "win_rate_pct": float(win_rate*100),
    "total_trades": int(n_trades),
    "profit_factor": float(profit_factor) if profit_factor else None,
    "benchmark_return_pct": float(bench_ret*100),
    "models_trained": models_trained,
    "total_data_rows": len(data),
    "entry_signals": int(n_entries),
}
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print("Done!")
