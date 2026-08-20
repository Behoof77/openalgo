#!/usr/bin/env python3
"""
1-year daily XGBoost reversal backtest on NIFTY index.

Uses NIFTY (NSE_INDEX) daily data via OpenAlgo API.
Feature: RSI(14), RelVol(20), ZScore(20), ADX(14)
Training: LOOKBACK=250, retrain every 50 bars
Signals: XGBoost probability + pivot detection + ADX filter
Risk: SL=2xATR, TP1=4xATR (50% exit), TP2=8xATR (remaining)
Execution: VectorBT Portfolio.from_signals()

Usage:
    uv run python strategies/xgb_reversals_NIFTY/backtest_1yr.py

Output: outputs/xgb_backtest_1yr/
"""

import os, json, warnings, sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=False)
warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
SYMBOL = "NIFTY"
EXCHANGE = "NSE_INDEX"
INTERVAL = "D"
START_DATE = "2025-08-01"
END_DATE = "2026-07-25"

WARMUP_BARS = 100
RETRAIN_EVERY = 30

PROB_THRESHOLD = 0.60
ADX_THRESHOLD = 22

SL_ATR = 2.0
TP1_ATR = 4.0
TP2_ATR = 8.0
TP1_SIZE = 0.5

MODEL_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

OUTPUT_DIR = "outputs/xgb_backtest_1yr"
os.makedirs(OUTPUT_DIR, exist_ok=True)
log = print


# ── Data ────────────────────────────────────────────────────────────
def fetch_data():
    """Fetch daily NIFTY index data via OpenAlgo API."""
    from openalgo import api

    client = api(
        api_key=os.environ["OPENALGO_API_KEY"],
        host=os.environ.get("HOST_SERVER") or os.environ.get("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    log(f"Fetching {SYMBOL} {EXCHANGE} {INTERVAL} {START_DATE} -> {END_DATE} ...")
    df = client.history(
        symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
        start_date=START_DATE, end_date=END_DATE,
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        log(f"ERROR: No data returned. Response: {df}")
        sys.exit(1)

    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "volume": "Volume",
    })

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        else:
            df.index = pd.to_datetime(df.index)

    df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    df = df.sort_index()
    log(f"Fetched {len(df)} rows, {df.index[0].date()} -> {df.index[-1].date()}")
    return df


# ── Features ────────────────────────────────────────────────────────
def make_features(df):
    """Compute feature columns in-place."""
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64)
    low = df["Low"].values.astype(np.float64)
    volume = df["Volume"].values.astype(np.float64)

    from openalgo import ta

    df["RSI"] = ta.rsi(close, 14)
    df["SMA20"] = ta.sma(close, 20)

    vol_sma = ta.sma(volume, 20)
    df["RelVol"] = np.where(vol_sma > 0, volume / vol_sma, 1.0)

    sma_val = ta.sma(close, 20)
    std_val = ta.stdev(close, 20)
    df["ZScore"] = np.where(std_val > 0, (close - sma_val) / std_val, 0.0)

    df["ADX"] = ta.adx(high, low, close, 14)[0]  # tuple: (ADX, +DI, -DI)
    df["ATR"] = ta.atr(high, low, close, 14)

    df["EMA9"] = ta.sma(close, 9)
    df["EMA50"] = ta.sma(close, 50)

    df["Close_lag1"] = df["Close"].shift(1)
    df["Volume_lag1"] = df["Volume"].shift(1)
    df["RSI_lag1"] = df["RSI"].shift(1)
    df["ADX_lag1"] = df["ADX"].shift(1)
    df["Ret_1"] = df["Close"].pct_change(1)
    df["Ret_5"] = df["Close"].pct_change(5)

    return df


def prepare_labels(df, horizon=3):
    """Create classification and regression labels."""
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1.0
    std_ret = df["Ret_1"].rolling(20).std().values

    labels_cls = np.full(len(df), np.nan)
    pos_thresh = 1.0 * std_ret
    neg_thresh = -1.0 * std_ret
    labels_cls[future_ret.values > pos_thresh] = 1
    labels_cls[future_ret.values < neg_thresh] = 0

    df["Label_cls"] = labels_cls
    df["Label_reg"] = future_ret.values
    return df


# ── Model Training ──────────────────────────────────────────────────
def train_models(train_df):
    """Train 4 XGBoost models on training subset."""
    from xgboost import XGBClassifier, XGBRegressor

    feat_cols = [
        "RSI", "SMA20", "RelVol", "ZScore", "ADX", "ATR",
        "EMA9", "EMA50", "Close_lag1", "Volume_lag1",
        "RSI_lag1", "ADX_lag1", "Ret_1", "Ret_5",
    ]

    train_df = train_df.dropna().copy()
    if len(train_df) < 50:
        return None

    X = train_df[feat_cols].values

    cls_mask = ~train_df["Label_cls"].isna()
    if cls_mask.sum() > 20:
        y_cls = train_df.loc[cls_mask, "Label_cls"].values
        cls_long = XGBClassifier(**MODEL_PARAMS)
        cls_long.fit(X[cls_mask.values], y_cls)
        y_cls_short = 1 - y_cls
        cls_short = XGBClassifier(**MODEL_PARAMS)
        cls_short.fit(X[cls_mask.values], y_cls_short)
    else:
        from sklearn.dummy import DummyClassifier
        cls_long = DummyClassifier(strategy="most_frequent")
        cls_long.fit(X[:2], np.array([0, 0]))
        cls_short = DummyClassifier(strategy="most_frequent")
        cls_short.fit(X[:2], np.array([1, 1]))

    reg_mask = ~train_df["Label_reg"].isna()
    if reg_mask.sum() > 20:
        y_reg = train_df.loc[reg_mask, "Label_reg"].values
        reg_long = XGBRegressor(**{**MODEL_PARAMS, "objective": "reg:squarederror"})
        reg_long.fit(X[reg_mask.values], y_reg)
        reg_short = XGBRegressor(**{**MODEL_PARAMS, "objective": "reg:squarederror"})
        reg_short.fit(X[reg_mask.values], -y_reg)
    else:
        reg_long = reg_short = None

    return {
        "cls_long": cls_long, "cls_short": cls_short,
        "reg_long": reg_long, "reg_short": reg_short,
        "feat_cols": feat_cols,
    }


# ── Pivot Detection ─────────────────────────────────────────────────
def detect_pivots(close, window=5):
    """Find swing highs/lows."""
    length = len(close)
    high = np.full(length, np.nan)
    low = np.full(length, np.nan)
    for i in range(window, length - window):
        if all(close[i] >= close[i - window:i]) and all(close[i] >= close[i + 1:i + window + 1]):
            high[i] = 1.0
        if all(close[i] <= close[i - window:i]) and all(close[i] <= close[i + 1:i + window + 1]):
            low[i] = 1.0
    return high, low


# ── Signal Generation ───────────────────────────────────────────────
def generate_signals(df, models):
    """Generate entry/exit signals from models."""
    if models is None:
        return None

    cls_long = models["cls_long"]
    cls_short = models["cls_short"]
    reg_long = models["reg_long"]
    reg_short = models["reg_short"]
    feat_cols = models["feat_cols"]

    df = df.copy()
    X = df[feat_cols].values

    if hasattr(cls_long, "predict_proba"):
        probL = cls_long.predict_proba(X)[:, 1]
        probS = cls_short.predict_proba(X)[:, 1]
    else:
        probL = np.full(len(df), 0.5)
        probS = np.full(len(df), 0.5)

    pred_ret_long = reg_long.predict(X) if reg_long else np.zeros(len(df))
    pred_ret_short = reg_short.predict(X) if reg_short else np.zeros(len(df))
    pred_ret_net = pred_ret_long - pred_ret_short

    pivots_high, pivots_low = detect_pivots(df["Close"].values, window=5)

    entry_long = (
        (probL >= PROB_THRESHOLD)
        & (df["ADX"].values >= ADX_THRESHOLD)
        & (pivots_low == 1.0)
        & (pred_ret_net > 0)
    )

    entry_short = (
        (probS >= PROB_THRESHOLD)
        & (df["ADX"].values >= ADX_THRESHOLD)
        & (pivots_high == 1.0)
        & (pred_ret_net < 0)
    )

    signals = pd.Series(0, index=df.index, dtype=int)
    signals[entry_long] = 1
    signals[entry_short] = 2

    log(f"  Signals: long={entry_long.sum()}, short={entry_short.sum()}")
    return signals


# ── Backtest ────────────────────────────────────────────────────────
def run_backtest(df, signals):
    """Run VectorBT Portfolio.from_signals()."""
    import vectorbt as vbt

    close = df["Close"]
    atr = df["ATR"].values

    entries = pd.Series(False, index=df.index)
    exits = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    short_exits = pd.Series(False, index=df.index)

    entries[signals == 1] = True
    short_entries[signals == 2] = True

    long_pos = entries.cumsum() - exits.cumsum()
    short_pos = short_entries.cumsum() - short_exits.cumsum()
    exits[long_pos.shift(1).fillna(0) > 0] = True
    short_exits[short_pos.shift(1).fillna(0) > 0] = True

    sl_dist = atr * SL_ATR
    tp_dist = atr * TP1_ATR

    sl = np.full(len(close), np.nan)
    tp = np.full(len(close), np.nan)
    for i in range(len(close)):
        if entries.iloc[i]:
            sl[i] = close.iloc[i] - sl_dist[i]
            tp[i] = close.iloc[i] + tp_dist[i]
        elif short_entries.iloc[i]:
            sl[i] = close.iloc[i] + sl_dist[i]
            tp[i] = close.iloc[i] - tp_dist[i]

    pf = vbt.Portfolio.from_signals(
        close.values,
        entries=entries.values,
        exits=exits.values,
        short_entries=short_entries.values,
        short_exits=short_exits.values,
        sl_stop=sl,
        tp_stop=tp,
        size=pd.Series(0.95, index=df.index),
        fees=0.0005,
        slippage=0.001,
        freq="D",
        init_cash=10_000_000,
    )

    return pf


# ── Main ────────────────────────────────────────────────────────────
def main():
    log("=" * 60)
    log("NIFTY 1-Year Daily XGBoost Reversal Backtest")
    log(f"Period: {START_DATE} to {END_DATE}")
    log("=" * 60)

    df = fetch_data()
    df = make_features(df)
    df = prepare_labels(df, horizon=3)
    log(f"Data shape: {df.shape}")

    all_signals = pd.Series(0, index=df.index, dtype=int)

    for i in range(WARMUP_BARS, len(df), max(1, RETRAIN_EVERY)):
        end_train = min(i, len(df))
        start_train = max(0, end_train - WARMUP_BARS)
        train_df = df.iloc[start_train:end_train]
        test_start = end_train
        test_end = min(end_train + RETRAIN_EVERY, len(df))

        if test_start >= test_end:
            break

        log(f"Window: train {start_train}-{end_train}, predict {test_start}-{test_end}")
        models = train_models(train_df)
        if models is None:
            continue

        test_df = df.iloc[test_start:test_end]
        sigs = generate_signals(test_df, models)
        if sigs is None:
            continue

        all_signals.iloc[test_start:test_end] = sigs

    total = (all_signals != 0).sum()
    log(f"\nTotal signals: {total} (long={(all_signals==1).sum()}, short={(all_signals==2).sum()})")

    if total == 0:
        log("No signals. Try lower PROB_THRESHOLD.")
        return

    log("Running VectorBT backtest ...")
    pf = run_backtest(df, all_signals)

    log("\n" + "=" * 60)
    log("RESULTS")
    log("=" * 60)
    stats = pf.stats()
    for k in ["Start", "End", "Period", "Start Value", "End Value",
              "Total Return [%]", "Max Drawdown [%]", "Sharpe Ratio",
              "Sortino Ratio", "Win Rate [%]", "Total Trades"]:
        if k in stats.index:
            log(f"  {k}: {stats[k]}")

    trades = pf.trades.records_readable
    equity = pf.value()
    trades.to_csv(f"{OUTPUT_DIR}/trades.csv", index=False)
    equity.to_csv(f"{OUTPUT_DIR}/equity.csv")
    all_signals.to_csv(f"{OUTPUT_DIR}/signals.csv")

    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump({
            "symbol": SYMBOL, "exchange": EXCHANGE, "interval": INTERVAL,
            "period": f"{START_DATE} to {END_DATE}",
            "prob_threshold": PROB_THRESHOLD, "adx_threshold": ADX_THRESHOLD,
            "sl_atr": SL_ATR, "tp1_atr": TP1_ATR, "tp2_atr": TP2_ATR,
        }, f, indent=2)

    log(f"\nOutputs in {OUTPUT_DIR}/")
    log("Done.")


if __name__ == "__main__":
    main()
