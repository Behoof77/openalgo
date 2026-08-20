#!/usr/bin/env python3
"""
XGBoost Lite: Reversals Strategy for NIFTY 1m
=============================================
Ported from TradingView "XG Boost Lite: Reversals | Gains Algo" by GainzAlgo.

A machine-learning strategy that trains 4 XGBoost models (long/short
classification + long/short regression) to predict reversal points on
NIFTY 1-minute data. Features: RSI(14), RelVol(20), Z-Score(20), ADX(14).

Usage (upload to OpenAlgo /python page, or run standalone):
  python strategy.py

Upload-ready for OpenAlgo /python self-hosted page.
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# ------------------------------------------------------------------
# 1. CONFIG  (all overridable via env vars or OpenAlgo /python params)
# ------------------------------------------------------------------

STRATEGY_NAME = os.getenv("STRATEGY_NAME", "xgb_reversals_NIFTY")

# Data
SYMBOL = os.getenv("SYMBOL", "NIFTY25AUG26FUT")
EXCHANGE = os.getenv("OPENALGO_STRATEGY_EXCHANGE", os.getenv("EXCHANGE", "NFO"))
INTERVAL = os.getenv("INTERVAL", "1m")
PRODUCT = os.getenv("PRODUCT", "NRML")
LOT_SIZE = int(os.getenv("LOT_SIZE", "65"))
QUANTITY = int(os.getenv("QUANTITY", str(LOT_SIZE)))

# XGBoost training
LOOKBACK = int(os.getenv("LOOKBACK", "250"))           # training window bars
RETRAIN_FREQ = int(os.getenv("RETRAIN_FREQ", "50"))    # retrain every N bars
N_ROUNDS = int(os.getenv("N_ROUNDS", "20"))            # boosting rounds
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "0.3"))
MAX_DEPTH = int(os.getenv("MAX_DEPTH", "4"))

# Signal thresholds
PROB_THRESHOLD = float(os.getenv("PROB_THRESHOLD", "0.50"))
ADX_FILTER = float(os.getenv("ADX_FILTER", "20"))

# Target parameters
FWD_BARS = int(os.getenv("FWD_BARS", "15"))
ATR_PERIOD = int(os.getenv("ATR_PERIOD", "14"))

# Risk management
TP1_R = float(os.getenv("TP1_R", "1.0"))               # TP1 = 1 * risk
TP3_FLOOR_R = float(os.getenv("TP3_FLOOR_R", "1.5"))   # TP3 floor = 1.5 * risk
SL_BUFFER_ATR = float(os.getenv("SL_BUFFER_ATR", "0.25"))
SL_MIN_ATR = float(os.getenv("SL_MIN_ATR", "0.5"))

# Pivot detection
PIVOT_LOOKBACK = int(os.getenv("PIVOT_LOOKBACK", "8"))

# Live mode
POLL_INTERVAL_SEC = int(os.getenv("POLL_INTERVAL_SEC", "15"))
WARMUP_BARS = int(os.getenv("WARMUP_BARS", "500"))

# ------------------------------------------------------------------
# 2. ENV + LOGGING
# ------------------------------------------------------------------

_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(_dotenv_path, override=False)

API_KEY = os.getenv("OPENALGO_API_KEY", "")
API_HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "https://skopaq.duckdns.org")
WS_URL = os.getenv("WEBSOCKET_URL") or (
    f"ws://{os.getenv('WEBSOCKET_HOST', '127.0.0.1')}:{os.getenv('WEBSOCKET_PORT', '8765')}"
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(STRATEGY_NAME)

# ------------------------------------------------------------------
# 3. FEATURE ENGINEERING
# ------------------------------------------------------------------


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 4 features: RSI(14), RelVol(20), ZScore(20), ADX(14).

    Returns DataFrame indexed like df, NaN-rows during indicator warmup.
    """
    from openalgo import ta

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float) if "volume" in df else pd.Series(0, index=df.index)

    feats = pd.DataFrame(index=df.index)

    # 1. RSI(14)
    feats["rsi_14"] = ta.rsi(close, 14)

    # 2. Relative Volume (volume / SMA volume 20)
    if volume.max() == 0:
        feats["rel_vol_20"] = 1.0
    else:
        vol_sma = ta.sma(volume, 20)
        if isinstance(vol_sma, np.ndarray):
            vol_sma = pd.Series(vol_sma, index=df.index).replace(0, np.nan)
        else:
            vol_sma = vol_sma.replace(0, np.nan)
        feats["rel_vol_20"] = volume / vol_sma

    # 3. Z-Score of Close(20)
    close_sma = ta.sma(close, 20)
    close_std = ta.stdev(close, 20)
    if isinstance(close_std, np.ndarray):
        close_std = pd.Series(close_std, index=df.index).replace(0, np.nan)
    else:
        close_std = close_std.replace(0, np.nan)
    feats["zscore_20"] = (close - close_sma) / close_std

    # 4. ADX(14)
    adx_result = ta.adx(high, low, close, 14)
    if isinstance(adx_result, tuple):
        feats["adx_14"] = pd.Series(adx_result[0], index=df.index)
    elif isinstance(adx_result, pd.DataFrame):
        feats["adx_14"] = adx_result.iloc[:, 0]
    elif isinstance(adx_result, pd.Series):
        feats["adx_14"] = adx_result
    else:
        try:
            feats["adx_14"] = adx_result["adx"]
        except (KeyError, TypeError):
            feats["adx_14"] = pd.Series(
                [a["adx"] if isinstance(a, dict) else np.nan for a in adx_result],
                index=df.index,
            )

    return feats.astype(float)


# ------------------------------------------------------------------
# 4. TARGET ENGINEERING
# ------------------------------------------------------------------


def make_targets(df: pd.DataFrame) -> dict:
    """Compute forward-15-bar classification and regression targets.

    Returns dict with cls_long, reg_long, cls_short, reg_short (Series, NaN
    on the last FWD_BARS rows where forward data is unavailable).
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    from openalgo import ta
    atr = ta.atr(high, low, close, ATR_PERIOD)

    fwd_max = high.rolling(FWD_BARS).max().shift(-FWD_BARS + 1)
    fwd_min = low.rolling(FWD_BARS).min().shift(-FWD_BARS + 1)

    gain = (fwd_max - close) / atr
    loss = (close - fwd_min) / atr

    cls_long = (gain >= 1.0).astype(float)
    cls_short = (loss >= 1.0).astype(float)
    reg_long = gain.clip(lower=0)
    reg_short = loss.clip(lower=0)

    return {
        "cls_long": cls_long,
        "reg_long": reg_long,
        "cls_short": cls_short,
        "reg_short": reg_short,
    }


# ------------------------------------------------------------------
# 5. MODEL TRAINING
# ------------------------------------------------------------------


def train_models(train_feats: pd.DataFrame, train_targets: dict) -> dict:
    """Train 4 XGBoost models on aligned features + targets.

    Returns dict {name: xgb_model or None if not enough samples}.
    """
    import xgboost as xgb

    models = {}
    X = train_feats.values

    for target_key, obj, eval_metric in [
        ("cls_long", "binary:logistic", "logloss"),
        ("cls_short", "binary:logistic", "logloss"),
        ("reg_long", "reg:squarederror", "rmse"),
        ("reg_short", "reg:squarederror", "rmse"),
    ]:
        y = train_targets[target_key].reindex(train_feats.index).values
        mask = ~np.isnan(y.astype(float))
        y_clean = y[mask]
        X_clean = X[mask]

        if len(y_clean) < 50:
            log.warning("Not enough samples for %s (%d), skipping", target_key, len(y_clean))
            models[target_key] = None
            continue

        dtrain = xgb.DMatrix(X_clean, label=y_clean.astype(float))
        params = {
            "objective": obj,
            "eval_metric": eval_metric,
            "learning_rate": LEARNING_RATE,
            "max_depth": MAX_DEPTH,
            "verbosity": 0,
            "nthread": 2,
        }
        model = xgb.train(params, dtrain, num_boost_round=N_ROUNDS)
        models[target_key] = model

    return models


# ------------------------------------------------------------------
# 6. PREDICTION
# ------------------------------------------------------------------


def predict(feats_row: np.ndarray, models: dict) -> dict:
    """Run a single feature row through all 4 models.

    Returns dict with probL, regL, probS, regS (float, NaN if model missing).
    """
    import xgboost as xgb

    result = {"probL": np.nan, "regL": np.nan, "probS": np.nan, "regS": np.nan}

    for src_key, dst_key in [("cls_long", "probL"), ("cls_short", "probS")]:
        model = models.get(src_key)
        if model is None:
            continue
        d = xgb.DMatrix(feats_row.reshape(1, -1))
        result[dst_key] = float(model.predict(d)[0])

    for src_key, dst_key in [("reg_long", "regL"), ("reg_short", "regS")]:
        model = models.get(src_key)
        if model is None:
            continue
        d = xgb.DMatrix(feats_row.reshape(1, -1))
        result[dst_key] = float(model.predict(d)[0])

    return result


# ------------------------------------------------------------------
# 7. PIVOT DETECTION
# ------------------------------------------------------------------


def detect_pivots(df: pd.DataFrame) -> dict:
    """Detect bull/bear pivot points for reversal entry timing.

    Bull pivot: low < lowest(low, PIVOT_LOOKBACK).shift(1) AND close > midpoint
    Bear pivot: high > highest(high, PIVOT_LOOKBACK).shift(1) AND close < midpoint

    Returns dict with bull_pivot, bear_pivot boolean Series.
    """
    from openalgo import ta

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    lowest_8 = ta.lowest(low, PIVOT_LOOKBACK)
    highest_8 = ta.highest(high, PIVOT_LOOKBACK)
    if isinstance(lowest_8, np.ndarray):
        lowest_8 = pd.Series(lowest_8, index=df.index)
    if isinstance(highest_8, np.ndarray):
        highest_8 = pd.Series(highest_8, index=df.index)

    bull_pivot_raw = low < lowest_8.shift(1)
    bear_pivot_raw = high > highest_8.shift(1)

    midpoint = (high + low) / 2
    bull_pivot = bull_pivot_raw & (close > midpoint)
    bear_pivot = bear_pivot_raw & (close < midpoint)

    return {"bull_pivot": bull_pivot, "bear_pivot": bear_pivot}


# ------------------------------------------------------------------
# 8. SIGNAL + RISK
# ------------------------------------------------------------------


def compute_signal(
    row_idx: int,
    df: pd.DataFrame,
    pivots: dict,
    predictions: np.ndarray,
    adx_series: pd.Series,
    atr_series: pd.Series,
) -> dict:
    """Evaluate signal at a given bar index.

    Returns dict with entry (bool), direction ("long"/"short"/None),
    sl_price, tp1_price, tp3_price.
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    atr = atr_series

    current_pred = predictions[row_idx]
    bull = pivots["bull_pivot"].iloc[row_idx]
    bear = pivots["bear_pivot"].iloc[row_idx]
    adx_val = adx_series.iloc[row_idx] if pd.notna(adx_series.iloc[row_idx]) else 0

    result = {"entry": False, "direction": None, "sl_price": None,
              "tp1_price": None, "tp3_price": None}

    # Long entry
    if bull and current_pred["probL"] >= PROB_THRESHOLD and adx_val >= ADX_FILTER:
        result["entry"] = True
        result["direction"] = "long"
        entry_price = close.iloc[row_idx]
        sl_distance = (entry_price - low.iloc[row_idx]) + SL_BUFFER_ATR * atr.iloc[row_idx]
        sl_distance = max(sl_distance, SL_MIN_ATR * atr.iloc[row_idx])
        risk = sl_distance
        result["sl_price"] = entry_price - sl_distance
        result["tp1_price"] = entry_price + risk * TP1_R
        reg = current_pred["regL"]
        if pd.notna(reg):
            tp3_price = entry_price + max(reg * atr.iloc[row_idx], risk * TP3_FLOOR_R)
        else:
            tp3_price = entry_price + risk * TP3_FLOOR_R
        result["tp3_price"] = tp3_price
        return result

    # Short entry
    if bear and current_pred["probS"] >= PROB_THRESHOLD and adx_val >= ADX_FILTER:
        result["entry"] = True
        result["direction"] = "short"
        entry_price = close.iloc[row_idx]
        sl_distance = (high.iloc[row_idx] - entry_price) + SL_BUFFER_ATR * atr.iloc[row_idx]
        sl_distance = max(sl_distance, SL_MIN_ATR * atr.iloc[row_idx])
        risk = sl_distance
        result["sl_price"] = entry_price + sl_distance
        result["tp1_price"] = entry_price - risk * TP1_R
        reg = current_pred["regS"]
        if pd.notna(reg):
            tp3_price = entry_price - max(reg * atr.iloc[row_idx], risk * TP3_FLOOR_R)
        else:
            tp3_price = entry_price - risk * TP3_FLOOR_R
        result["tp3_price"] = tp3_price
        return result

    return result


# ------------------------------------------------------------------
# 9. EXIT POSITION HELPER
# ------------------------------------------------------------------


def exit_position(client, state: dict):
    """Exit current open position."""
    try:
        action = "SELL" if state["position"] == "long" else "BUY"
        resp = client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=SYMBOL,
            exchange=EXCHANGE,
            action=action,
            price_type="MARKET",
            product=PRODUCT,
            quantity=str(QUANTITY),
            price=str(0),
        )
        log.info("Exit order: %s", resp)
    except Exception as e:
        log.exception("Exit order failed: %s", e)
    finally:
        state["position"] = None
        state["order_id"] = None


# ------------------------------------------------------------------
# 10. LIVE LOOP
# ------------------------------------------------------------------


def run_live():
    """Run the strategy live via OpenAlgo REST + WebSocket."""
    from openalgo import api, ta

    log.info("=== LIVE MODE ===")
    log.info("Symbol: %s on %s @ %s", SYMBOL, EXCHANGE, INTERVAL)
    log.info("Host: %s", API_HOST)

    client = api(api_key=API_KEY, host=API_HOST, ws_url=WS_URL)

    # --- Paper trading via analyzer mode ---
    try:
        status = client.analyzerstatus()
        if isinstance(status, dict) and status.get("data", {}).get("analyze_mode") is False:
            toggle = client.analyzertoggle(mode=True)
            log.info("Analyzer mode toggled: %s", toggle)
        else:
            log.info("Analyzer already active (paper trading)")
    except Exception as e:
        log.warning("Could not toggle analyzer mode (not broker-connected?): %s", e)

    # --- Fetch warmup data ---
    log.info("Fetching warmup data (%d bars)...", WARMUP_BARS)
    end_date = datetime.now()
    df = client.history(
        symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
        start_date=(end_date - timedelta(days=14)).strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if not isinstance(df, pd.DataFrame) or df.empty:
        log.error("No warmup data -- cannot start. Check API key and NFO data access.")
        sys.exit(1)

    log.info("Got %d warmup bars", len(df))
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    # --- Initial training ---
    models = {}
    feats = make_features(df)
    targets = make_targets(df)
    t_feats = feats.dropna()
    t_targets = {}
    for k in ["cls_long", "reg_long", "cls_short", "reg_short"]:
        t_targets[k] = targets[k].reindex(t_feats.index)
    models = train_models(t_feats, t_targets)
    model_status = {k: "OK" if v else "SKIP" for k, v in models.items()}
    log.info("Models: %s", model_status)

    # --- State ---
    state = {
        "position": None,
        "entry_price": None,
        "sl_price": None,
        "tp1_price": None,
        "tp3_price": None,
        "tp1_hit": False,
        "order_id": None,
        "bar_count": len(df),
        "last_retrain": len(df) - 1,
        "predictions": [None] * len(df),
    }

    # Pre-compute pivots + ADX + ATR for warmup data
    pivots = detect_pivots(df)
    adx_result = ta.adx(df["high"].astype(float), df["low"].astype(float),
                        df["close"].astype(float), 14)
    if isinstance(adx_result, pd.DataFrame):
        adx_series = adx_result.iloc[:, 0]
    elif isinstance(adx_result, pd.Series):
        adx_series = adx_result
    else:
        adx_series = pd.Series(index=df.index, dtype=float)
    atr_series = ta.atr(
        df["high"].astype(float), df["low"].astype(float),
        df["close"].astype(float), ATR_PERIOD,
    )

    # Predict for warmup bars
    for i in range(len(df)):
        feat_row = feats.iloc[i].values.astype(float)
        if np.any(np.isnan(feat_row)):
            continue
        state["predictions"][i] = predict(feat_row, models)

    # Count how many warmup predictions cross threshold (diagnostic)
    probL_vals = [p["probL"] for p in state["predictions"] if p is not None and not np.isnan(p["probL"])]
    probS_vals = [p["probS"] for p in state["predictions"] if p is not None and not np.isnan(p["probS"])]
    if probL_vals:
        log.info("Warmup probL: mean=%.3f max=%.3f >=%.2f=%d/%d",
                 np.mean(probL_vals), np.max(probL_vals),
                 PROB_THRESHOLD, sum(1 for v in probL_vals if v >= PROB_THRESHOLD), len(probL_vals))
    if probS_vals:
        log.info("Warmup probS: mean=%.3f max=%.3f >=%.2f=%d/%d",
                 np.mean(probS_vals), np.max(probS_vals),
                 PROB_THRESHOLD, sum(1 for v in probS_vals if v >= PROB_THRESHOLD), len(probS_vals))

    # --- Main bar processing ---
    def on_new_bar():
        nonlocal df, models, pivots, adx_series, atr_series
        try:
            new_df = client.history(
                symbol=SYMBOL, exchange=EXCHANGE, interval=INTERVAL,
                start_date=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                end_date=datetime.now().strftime("%Y-%m-%d"),
            )

            if not isinstance(new_df, pd.DataFrame) or new_df.empty:
                return

            if "timestamp" in new_df.columns:
                new_df["timestamp"] = pd.to_datetime(new_df["timestamp"])
                new_df = new_df.set_index("timestamp")
            else:
                new_df.index = pd.to_datetime(new_df.index)
            new_df = new_df.sort_index()
            if new_df.index.tz is not None:
                new_df.index = new_df.index.tz_convert(None)

            old_len = len(df)
            df = pd.concat([df, new_df])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if len(df) == old_len:
                return

            # Recompute features/pivots/indicators
            feats = make_features(df)
            pivots = detect_pivots(df)
            adx_result = ta.adx(df["high"].astype(float), df["low"].astype(float),
                                df["close"].astype(float), 14)
            if isinstance(adx_result, pd.DataFrame):
                adx_series = adx_result.iloc[:, 0]
            elif isinstance(adx_result, pd.Series):
                adx_series = adx_result
            else:
                adx_series = pd.Series(index=df.index, dtype=float)
            atr_series = ta.atr(
                df["high"].astype(float), df["low"].astype(float),
                df["close"].astype(float), ATR_PERIOD,
            )

            # Extend prediction array
            while len(state["predictions"]) < len(df):
                state["predictions"].append(None)

            new_bars = len(df) - old_len
            state["bar_count"] = len(df)

            # Process newly added bars (only the latest closed bar matters)
            for offset in range(new_bars):
                idx = old_len - new_bars + offset
                if idx < 0:
                    continue
                idx = min(idx, len(df) - 2)

                feat_row = feats.iloc[idx].values.astype(float)
                if np.any(np.isnan(feat_row)):
                    continue
                state["predictions"][idx] = predict(feat_row, models)

                current_signal = state["predictions"][idx]

                # Exit checks
                if state["position"] is not None:
                    # Check SL/TP via LTP
                    ltp_resp = client.quotes(symbol=SYMBOL, exchange=EXCHANGE)
                    ltp = float(ltp_resp.get("data", {}).get("ltp", 0))

                    if state["position"] == "long":
                        if not state["tp1_hit"] and state["tp1_price"] and ltp >= state["tp1_price"]:
                            log.info("TP1 HIT long @ %.2f", ltp)
                            state["tp1_hit"] = True
                            state["sl_price"] = state["entry_price"]

                        if state["sl_price"] and ltp <= state["sl_price"]:
                            log.info("SL HIT long @ %.2f (entry=%.2f, sl=%.2f)",
                                     ltp, state["entry_price"], state["sl_price"])
                            exit_position(client, state)
                            continue

                        if state["tp3_price"] and ltp >= state["tp3_price"]:
                            log.info("TP3 HIT long @ %.2f", ltp)
                            exit_position(client, state)
                            continue

                        if state["tp1_hit"] and state["tp1_price"] and state["tp3_price"]:
                            tp2_price = (state["tp1_price"] + state["tp3_price"]) / 2
                            if ltp >= tp2_price:
                                log.info("TP2 HIT long @ %.2f", ltp)
                                exit_position(client, state)
                                continue

                    elif state["position"] == "short":
                        if not state["tp1_hit"] and state["tp1_price"] and ltp <= state["tp1_price"]:
                            log.info("TP1 HIT short @ %.2f", ltp)
                            state["tp1_hit"] = True
                            state["sl_price"] = state["entry_price"]

                        if state["sl_price"] and ltp >= state["sl_price"]:
                            log.info("SL HIT short @ %.2f (entry=%.2f, sl=%.2f)",
                                     ltp, state["entry_price"], state["sl_price"])
                            exit_position(client, state)
                            continue

                        if state["tp3_price"] and ltp <= state["tp3_price"]:
                            log.info("TP3 HIT short @ %.2f", ltp)
                            exit_position(client, state)
                            continue

                        if state["tp1_hit"] and state["tp1_price"] and state["tp3_price"]:
                            tp2_price = (state["tp1_price"] + state["tp3_price"]) / 2
                            if ltp <= tp2_price:
                                log.info("TP2 HIT short @ %.2f", ltp)
                                exit_position(client, state)
                                continue

                # Entry check (no open position)
                if state["position"] is None:
                    sig = compute_signal(idx, df, pivots, state["predictions"], adx_series, atr_series)
                    if sig["entry"]:
                        action = "BUY" if sig["direction"] == "long" else "SELL"
                        entry_price = float(df["close"].iloc[idx])

                        log.info(
                            "SIGNAL %s @ bar %d (%s) entry=%.2f SL=%.2f TP1=%.2f TP3=%.2f "
                            "probL=%.3f probS=%.3f",
                            sig["direction"].upper(), idx,
                            str(df.index[idx])[:19], entry_price,
                            sig["sl_price"], sig["tp1_price"], sig["tp3_price"],
                            current_signal["probL"] if current_signal else 0,
                            current_signal["probS"] if current_signal else 0,
                        )

                        resp = client.placeorder(
                            strategy=STRATEGY_NAME,
                            symbol=SYMBOL,
                            exchange=EXCHANGE,
                            action=action,
                            price_type="MARKET",
                            product=PRODUCT,
                            quantity=str(QUANTITY),
                            price=str(entry_price),
                        )

                        log.info("Order response: %s", resp)
                        order_status = resp.get("status", "error") if isinstance(resp, dict) else "error"

                        if order_status == "success":
                            order_id = resp.get("data", {}).get("orderid", str(time.time()))
                            state["position"] = sig["direction"]
                            state["entry_price"] = entry_price
                            state["sl_price"] = sig["sl_price"]
                            state["tp1_price"] = sig["tp1_price"]
                            state["tp3_price"] = sig["tp3_price"]
                            state["tp1_hit"] = False
                            state["order_id"] = order_id
                            log.info("Position opened: %s @ %.2f", sig["direction"], entry_price)

            # Retrain check
            if state["bar_count"] - state["last_retrain"] >= RETRAIN_FREQ:
                train_end = state["bar_count"]
                train_start = max(0, train_end - LOOKBACK)
                train_df = df.iloc[train_start:train_end]
                train_feats = make_features(train_df)
                train_targets = make_targets(train_df)
                t_feats = train_feats.dropna()
                t_targets = {}
                for k in ["cls_long", "reg_long", "cls_short", "reg_short"]:
                    t_targets[k] = train_targets[k].reindex(t_feats.index)
                if len(t_feats) >= 50:
                    new_models = train_models(t_feats, t_targets)
                    for k, v in new_models.items():
                        if v is not None:
                            models[k] = v
                    state["last_retrain"] = state["bar_count"]
                    log.info("Retrained models at bar %d (%d samples)",
                             state["bar_count"], len(t_feats))

        except Exception as e:
            log.exception("Error in on_new_bar: %s", e)

    # --- SIGTERM / shutdown ---
    stop_event = threading.Event()

    def _shutdown(signum, frame):
        log.info("Signal %d received -- shutting down", signum)
        stop_event.set()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    log.info("Live loop started. Poll interval: %ds", POLL_INTERVAL_SEC)
    try:
        while not stop_event.is_set():
            on_new_bar()
            stop_event.wait(POLL_INTERVAL_SEC)
    except KeyboardInterrupt:
        log.info("Keyboard interrupt")
    finally:
        if state["position"] is not None:
            log.warning("Shutting down with open position -- manual cleanup needed")
        try:
            client.disconnect()
        except Exception:
            pass
        log.info("Shutdown complete")


# ------------------------------------------------------------------
# 11. DISPATCHER
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=f"{STRATEGY_NAME} -- XGBoost Reversals for NIFTY 1m"
    )
    parser.add_argument(
        "--mode",
        choices=["live"],
        default=os.getenv("MODE", "live"),
        help="Run mode (default: %(default)s)",
    )
    args = parser.parse_args()

    log.info("Strategy: %s  Mode: %s", STRATEGY_NAME, args.mode)
    log.info("Symbol: %s  Exchange: %s  Interval: %s", SYMBOL, EXCHANGE, INTERVAL)
    log.info("Training: lookback=%d  retrain=%d  rounds=%d  lr=%.2f",
             LOOKBACK, RETRAIN_FREQ, N_ROUNDS, LEARNING_RATE)
    log.info("Thresholds: prob=%.2f  adx=%.1f  fwd_bars=%d",
             PROB_THRESHOLD, ADX_FILTER, FWD_BARS)
    log.info("Risk: tp1R=%.1f  tp3FloorR=%.1f  slBuffer=%.2f*ATR  slMin=%.2f*ATR",
             TP1_R, TP3_FLOOR_R, SL_BUFFER_ATR, SL_MIN_ATR)

    run_live()


if __name__ == "__main__":
    main()
