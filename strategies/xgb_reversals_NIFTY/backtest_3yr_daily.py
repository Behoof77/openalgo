#!/usr/bin/env python3
"""V5X: Entry>40%, exit<20% or swing high, no ATR SL/TP — SREEL."""

import os, json, warnings, sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import vectorbt as vbt
from openalgo import ta

warnings.filterwarnings("ignore")

script_dir = Path(__file__).resolve().parent
output_dir = script_dir / "outputs" / "xgb_backtest_3yr_v5x_sreel"
output_dir.mkdir(parents=True, exist_ok=True)

# ============================================================
# CONFIG
# ============================================================
LOOKBACK = 250       # trading days to train each model
RETRAIN_EVERY = 50   # retrain days
PROB_THRESHOLD = 0.40
ATR_PERIOD = 14
INIT_CASH = 10_000_000

SYMBOL = "SREEL"
TICKER = "SREEL.NS"
BENCH_TICKER = "^NSEI"
START_DATE = "2023-08-01"
END_DATE = "2026-07-30"
FREQ = "1D"

# Indian delivery equity fees (STT 0.1% + stamp 0.015% + turnover 0.0001% + SEBI 0.0001% + GST 18% on brokerage)
# Approx all-in: 0.12% round-trip per leg
FEES = 0.0012
FIXED_FEES = 20  # brokerage per order

config = {
    "symbol": SYMBOL, "ticker": TICKER, "interval": FREQ,
    "start": START_DATE, "end": END_DATE,
    "lookback": LOOKBACK, "retrain_every": RETRAIN_EVERY,
    "prob_threshold": PROB_THRESHOLD,
    "atr_period": ATR_PERIOD,
    "fees": FEES, "fixed_fees": FIXED_FEES,
    "data_source": "yfinance",
    "version": "v5 (entry>40%, exit<20% or swing high, no ATR SL/TP)",
}

# ============================================================
# FETCH DATA - yfinance
# ============================================================
print(f"=== Fetching {SYMBOL} daily from yfinance ({START_DATE} to {END_DATE}) ===")
ticker = yf.Ticker(TICKER)
df = ticker.history(start=START_DATE, end=END_DATE, interval="1d")
df.index = pd.to_datetime(df.index).tz_localize(None)
df.columns = [c.lower() for c in df.columns]
df = df.rename(columns={"close": "close", "high": "high", "low": "low", "open": "open", "volume": "volume"})
print(f"  Rows: {len(df)}")
print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
price = df["close"]
high = df["high"]
low = df["low"]
volume = df["volume"]

# ============================================================
# FEATURE ENGINEERING (same as strategy.py)
# ============================================================
print("=== Feature engineering ===")
rsi = ta.rsi(price, 14)
rel_vol_ma = volume.rolling(20).mean().replace(0, np.nan)
rel_vol = volume / rel_vol_ma
rel_vol = rel_vol.fillna(1.0)
zscore = price.rolling(20).apply(lambda x: (x.iloc[-1] - x.mean()) / x.std() if x.std() > 0 else 0, raw=False)
adx_val = ta.adx(high.values, low.values, price.values, 14)
adx_line = adx_val[0] if isinstance(adx_val, (tuple, list)) else adx_val
adx_series = pd.Series(adx_line, index=price.index) if not isinstance(adx_line, pd.Series) else adx_line
atr_series = ta.atr(high.values, low.values, price.values, 14)
atr_s = pd.Series(atr_series, index=price.index) if not isinstance(atr_series, pd.Series) else atr_series

# Rolling std of close (20d)
std20 = price.rolling(20).std()

macd_result = ta.macd(price, 12, 26, 9)
if isinstance(macd_result, tuple):
    macd_line, macd_signal, macd_hist = macd_result
else:
    macd_hist = pd.Series(0.0, index=price.index)

bbands_result = ta.bbands(price, 20, 2)
if isinstance(bbands_result, tuple) and len(bbands_result) >= 3:
    bb_upper, bb_mid, bb_lower = bbands_result[0], bbands_result[1], bbands_result[2]
    bb_pct_b = (price - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)
else:
    bb_pct_b = pd.Series(0.5, index=price.index)
bb_pct_b = bb_pct_b.fillna(0.5).clip(0, 1)

vol_mom = (volume / volume.shift(5)).fillna(1.0)

# Pivot detection (simple: 5-bar swing high/low)
def pivot_high(s, left=2, right=2):
    peaks = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        if s.iloc[i] == max(s.iloc[i-left:i+right+1]):
            peaks.iloc[i] = True
    return peaks

def pivot_low(s, left=2, right=2):
    troughs = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        if s.iloc[i] == min(s.iloc[i-left:i+right+1]):
            troughs.iloc[i] = True
    return troughs

ph = pivot_high(price, 2, 2)
pl = pivot_low(price, 2, 2)

# Build feature matrix
features_df = pd.DataFrame({
    "rsi": rsi,
    "rel_vol": rel_vol,
    "zscore": zscore,
    "adx": adx_series,
    "atr_pct": atr_s / price * 100,
    "std20_pct": std20 / price * 100,
    "macd_hist": macd_hist,
    "bb_pct_b": bb_pct_b,
    "vol_mom": vol_mom,
    "is_swing_high": ph.astype(int),
    "is_swing_low": pl.astype(int),
    "close": price,
}, index=price.index)

feat_cols = ["rsi", "rel_vol", "zscore", "adx", "atr_pct", "std20_pct", "macd_hist", "bb_pct_b", "vol_mom", "is_swing_high", "is_swing_low"]
for c in feat_cols:
    features_df[c] = features_df[c].replace([np.inf, -np.inf], 0.0)

# Labels: 1 if price reverses UP in next 3 bars (long), 1 if DOWN (short)
fwd_ret_3 = price.shift(-3) / price - 1

label_long = (fwd_ret_3 > 0.005).astype(int)   # 0.5% up reversal
label_short = (fwd_ret_3 < -0.005).astype(int)  # 0.5% down reversal
reg_target = fwd_ret_3  # regression target (actual % return)

features_df["label_long"] = label_long
features_df["label_short"] = label_short
features_df["reg_target"] = reg_target

data_len = len(features_df)
print(f"  Features: {list(features_df.columns)}")
print(f"  {data_len} rows")

# Drop rows with NaN features
feat_cols = ["rsi", "rel_vol", "zscore", "adx", "atr_pct", "std20_pct", "macd_hist", "bb_pct_b", "vol_mom", "is_swing_high", "is_swing_low"]
features_clean = features_df.dropna(subset=feat_cols)
print(f"  After dropping NaN features: {len(features_clean)} rows")
print(f"  Long labels: {features_clean['label_long'].sum()} ({features_clean['label_long'].mean()*100:.1f}%)")
print(f"  Short labels: {features_clean['label_short'].sum()} ({features_clean['label_short'].mean()*100:.1f}%)")

# ============================================================
# XGBOOST WALK-FORWARD
# ============================================================
print("=== Walk-forward XGBoost training ===")
from xgboost import XGBClassifier, XGBRegressor

models = ["cls_long", "cls_short", "reg_long", "reg_short"]
model_map = {}

min_train_idx = LOOKBACK
train_indices = list(range(min_train_idx, data_len))

# Pre-generate all predictions
pred_cols = ["pred_prob_long", "pred_prob_short", "pred_ret_long", "pred_ret_short"]
all_preds = pd.DataFrame(0.0, index=df.index, columns=pred_cols)

pivot_indices = list(range(min_train_idx, data_len, RETRAIN_EVERY))
if pivot_indices[-1] != data_len - 1:
    pivot_indices.append(data_len - 1)

print(f"  Training windows: {len(pivot_indices)}")
for i, start_i in enumerate(pivot_indices):
    end_i = pivot_indices[i+1] if i+1 < len(pivot_indices) else data_len
    train_start = 0
    train_end = start_i
    raw_slice = features_clean.iloc[train_start:train_end]
    train_slice = raw_slice.dropna(subset=["label_long", "label_short", "reg_target"])
    pred_slice = features_df.iloc[train_end:end_i]

    if len(train_slice) < 100:
        print(f"    Window {i}: skip (train={len(train_slice)} < 100)")
        continue

    print(f"    Window {i}: train [{train_start}:{train_end}] pred [{train_end}:{end_i}]", end="")

    for mname in models:
        is_cls = "cls" in mname
        direction = "long" if "long" in mname else "short"
        label_col = f"label_{direction}"

        n_pos = train_slice[label_col].sum()
        if n_pos < 5:
            print(f" skip {mname} (pos={n_pos})", end="")
            continue

        scale_pos_weight = (len(train_slice) - n_pos) / n_pos if n_pos > 0 else 1

        cls = (XGBClassifier if is_cls else XGBRegressor)(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
        if is_cls:
            cls.set_params(scale_pos_weight=min(scale_pos_weight, 10))

        cls.fit(train_slice[feat_cols], train_slice[label_col] if is_cls else train_slice["reg_target"])

        # Predict
        pred_slice_local = features_df.reindex(pred_slice.index)

        if is_cls:
            prob = cls.predict_proba(pred_slice_local[feat_cols])[:, 1]
            col = f"pred_prob_{direction}"
        else:
            prob = cls.predict(pred_slice_local[feat_cols])
            col = f"pred_ret_{direction}"

        # Ensure index alignment
        all_preds.loc[pred_slice.index, col] = prob

    print()

print("  Predictions generated")

print("=== Signal generation ===")

long_prob = all_preds["pred_prob_long"]
long_signal_raw = (long_prob >= PROB_THRESHOLD)
exit_signal = (long_prob < 0.20) | ph

entries_long_arr = ta.exrem(long_signal_raw.fillna(False), exit_signal.fillna(False))
entries_long = pd.Series(entries_long_arr, index=price.index).fillna(False).astype(bool)
exits_long = exit_signal

size_series = pd.Series(0.0, index=price.index)
entry_idx = entries_long[entries_long].index
size_series[entry_idx] = INIT_CASH

print(f"  Long entries: {entries_long.sum()}")
if len(entry_idx) > 0:
    print(f"  Mean entry confidence: {long_prob[entry_idx].mean():.3f}")
    print(f"  Avg position size: Rs {size_series[entry_idx].mean():,.0f}")

# ============================================================
# VECTORBT BACKTEST
# ============================================================
print("=== Running VectorBT backtest ===")

pf = vbt.Portfolio.from_signals(
    price, entries_long, exits_long,
    init_cash=INIT_CASH,
    size=size_series, size_type="value",
    fees=FEES, fixed_fees=FIXED_FEES,
    direction="longonly",
    min_size=1, size_granularity=1,
    freq="1D",
    slippage=0.001,
)

# Benchmark: NIFTY
print(f"=== Fetching benchmark ({BENCH_TICKER}) from yfinance ===")
bench = yf.Ticker(BENCH_TICKER)
df_bench = bench.history(start=START_DATE, end=END_DATE, interval="1d")
df_bench.index = pd.to_datetime(df_bench.index).tz_localize(None)
bench_price = df_bench["Close"]
bench_price = bench_price.reindex(price.index).ffill().bfill()
pf_bench = vbt.Portfolio.from_holding(bench_price, init_cash=INIT_CASH, fees=0.0, freq="1D")

# ============================================================
# RESULTS
# ============================================================
print()
print("=" * 60)
print(f"  XGBoost Reversal - {SYMBOL} 3yr Daily")
print("=" * 60)

stats = pf.stats()
print()
print(stats)

# Comparison table
tr = pf.total_return()
sr = pf.sharpe_ratio()
sort = pf.sortino_ratio()
mdd = pf.max_drawdown()
wr = pf.trades.win_rate()
ntrades = pf.trades.count()
pf_ratio = pf.trades.profit_factor()

tr_b = pf_bench.total_return()
sr_b = pf_bench.sharpe_ratio()
sort_b = pf_bench.sortino_ratio()
mdd_b = pf_bench.max_drawdown()

print()
print("  Strategy vs NIFTY 50")
print(f"  {'Metric':<20} {'Strategy':<15} {'NIFTY 50':<15}")
print(f"  {'-'*20} {'-'*15} {'-'*15}")
print(f"  {'Total Return':<20} {tr*100:<15.2f}% {tr_b*100:<15.2f}%")
print(f"  {'Sharpe Ratio':<20} {sr:<15.2f} {sr_b:<15.2f}")
print(f"  {'Sortino Ratio':<20} {sort:<15.2f} {sort_b:<15.2f}")
print(f"  {'Max Drawdown':<20} {mdd*100:<15.2f}% {mdd_b*100:<15.2f}%")
print(f"  {'Win Rate':<20} {wr*100:<15.1f}% {'N/A':<15}")
print(f"  {'Total Trades':<20} {ntrades:<15} {'N/A':<15}")
print(f"  {'Profit Factor':<20} {pf_ratio:<15.2f} {'N/A':<15}")

# Plain language
print()
print("  Key Takeaways")
print(f"  - With Rs 1Cr capital, {ntrades} signals triggered over 3 years")
print(f"  - Strategy returned {tr*100:.2f}% vs NIFTY {tr_b*100:.2f}%")
print(f"  - Worst drawdown: {abs(mdd)*100:.2f}% (Rs {abs(mdd)*10_000_000:,.0f})")
print(f"  - Win rate: {wr*100:.1f}% (profitable trades out of {ntrades})")

# Export
pf.positions.records_readable.to_csv(output_dir / "trades.csv", index=False)
pd.DataFrame({"equity": pf.value()}).to_csv(output_dir / "equity.csv")
with open(output_dir / "config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"  Trades exported to {output_dir / 'trades.csv'}")
print(f"  Config exported to {output_dir / 'config.json'}")
