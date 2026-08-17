"""
kronos_sizing.py — NIFTY futures trade engine with Kronos position-sizing overlay
==================================================================================

A single-file, upload-ready strategy for the OpenAlgo /python self-hosted runner.

ARCHITECTURE (two strictly isolated layers):

    BASE SIGNAL ENGINE (Layer 1)
        |
        +---> direction + entry + ATR
        |
        +---> SL = 1.0 ATR / TP = 2.5 ATR / max window = 180 min   (FROZEN)
        |
        +---> KRONOS POSITION-SIZING OVERLAY (Layer 2)
        |
        +---> strength = abs(forecast_return_pct)
        |
        +---> <0.050% -> 0.50x | 0.050-0.075% -> 0.75x | >=0.075% -> 1.00x  (FROZEN)
        |
        +---> qty = BASE_QTY * weight, floor to lot-size multiple
        |
        +---> OpenAlgo order

Layer 1 (BASE SIGNAL ENGINE) generates: direction, signal, entry, ATR.
Layer 2 (KRONOS OVERLAY) determines ONLY the position_size_multiplier at entry.

Kronos NEVER:
  - determines trade direction (no agreement/direction filter),
  - creates or cancels entries,
  - alters entry / SL / TP / ATR / exit mechanics,
  - is evaluated after entry (entry-only evaluation, no continuous resizing).

SIGNAL PROVENANCE — READ BEFORE LIVE TRADING
---------------------------------------------
The base signal logic below is **C: NEWLY DESIGNED**. It is NOT:
  - (A) recovered from the historically validated (612/773) strategy, nor
  - (B) reconstructed from any existing source file.
The frozen execution contract (SL=1.0 ATR, TP=2.5 ATR, 180-min window) was
preserved verbatim, but the historical validation does not encode the original
signal-generation formula. Therefore MODE=live is GATED behind
BASE_ENGINE_VALIDATED=true and MUST NOT be enabled until the base engine has
been demonstrated to match the historical trade-generation logic. Default
MODE=validate logs every decision and NEVER places an order.

FROZEN EXECUTION PARAMETERS (env-overridable, defaults are frozen):
  SL_ATR=1.0, TP_ATR=2.5, TRAILING_SL_ATR=0, TRAILING_ACTIVATION_ATR=0,
  MAX_EXECUTION_WINDOW_MIN=180, KRONOS_TIMEOUT_SECONDS=5.

FROZEN KRONOS SIZING:
  kronos_strength_weight(strength_pct):
      strength_pct <  0.050        -> 0.50x
      0.050 <= strength_pct < 0.075 -> 0.75x
      strength_pct >= 0.075         -> 1.00x
  Fail-closed: any Kronos failure (timeout, malformed, stale, empty
  raw_predictions) -> NO TRADE. NEVER falls back to 1.00x.

Kronos server contract (reused, not invented): POST /predict on the existing
FastAPI server (kronos_server/main.py). Request: {"data": [OHLCV records, most
recent last, >= 512 rows], "freq": "...", "profile": "fast"|"normal"|"accurate"}.
Response: {"prediction": 1|-1|0, "confidence": float, "raw_predictions": [float],
"inference_ms": float, "rows_received": int}. raw_predictions are PRICE
forecasts (same units as the last close sent to the server), NOT percentage
returns. forecast_price = mean(raw_predictions);
forecast_return_pct = (forecast_price - last_close) / last_close * 100;
strength_pct = abs(forecast_return_pct). This conversion keeps the frozen
sizing tiers (0.050 / 0.075) in the same units as the validated research —
never use mean(raw_predictions) directly as the return. The server returns no
timestamp field, so the stale check is client-side wall-clock.

ENVIRONMENT (all optional unless noted):
  OPENALGO_API_KEY        required (injected by the /python platform)
  HOST_SERVER             preferred host (default https://skopaq.duckdns.org)
  OPENALGO_HOST           fallback host
  WEBSOCKET_URL           default ws://127.0.0.1:8765
  STRATEGY_NAME           default kronos_sizing
  MODE                    validate (default) | live  (live requires BASE_ENGINE_VALIDATED)
  BASE_ENGINE_VALIDATED   true|false (default false) — live gate
  UNDERLYING              default NIFTY
  EXCHANGE                default NFO (futures)
  BASE_INTERVAL           default 5m
  BASE_LOOKBACK_DAYS      default 10 (enough for >= 512 context rows at 5m)
  KRONOS_CONTEXT_ROWS     default 512 (minimum rows sent to Kronos)
  KRONOS_URL              default http://127.0.0.1:8000
  KRONOS_TIMEOUT_SECONDS  default 5 (frozen)
  KRONOS_FREQ             default 5min
  KRONOS_PROFILE          default fast
  BASE_QTY                default 75 (normal 1.00x position, must be >= 1 lot)
  PRODUCT                 default MIS
  PRICE_TYPE              default MARKET
  ATR_PERIOD              default 14
  SL_ATR                  default 1.0 (frozen)
  TP_ATR                  default 2.5 (frozen)
  TRAILING_SL_ATR         default 0 (frozen)
  TRAILING_ACTIVATION_ATR default 0 (frozen)
  MAX_EXECUTION_WINDOW_MIN default 180 (frozen)
  CADENCE_SECONDS         default 60 (main loop period)
  STALE_MARKET_SECONDS    default 300 (halt new entries if market data stale)
  MAX_ENTRY_RETRIES       default 3
  DATA_DIR                default ./kronos_sizing_data (decisions log + state)
  SQUARE_OFF_HHMM         optional (e.g. 15:15) — force-exit open positions

SIGTERM/SIGINT: graceful stop (log state; square off if a position is open and
SQUARE_OFF_HHMM/SHUTDOWN policy allows). stdout-only logging (no emojis).
"""

from __future__ import annotations

import json
import math
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from strategies.shared import (
    ExecutionMode,
    PositionOwner,
    Reconciler,
    RecoveryManager,
    StatePersister,
    StopMode,
)
from strategies.shared.execution_modes import register_strategy_mode

# ---------------------------------------------------------------------------
# FROZEN CONSTANTS — do not edit (spec sections 1 and 3)
# ---------------------------------------------------------------------------
STRENGTH_LOW: float = 0.050   # percent
STRENGTH_HIGH: float = 0.075  # percent
WEIGHT_LOW: float = 0.50
WEIGHT_MID: float = 0.75
WEIGHT_HIGH: float = 1.00

SL_ATR_DEFAULT: float = 1.0
TP_ATR_DEFAULT: float = 2.5
TRAILING_SL_ATR_DEFAULT: float = 0.0
TRAILING_ACTIVATION_ATR_DEFAULT: float = 0.0
MAX_EXECUTION_WINDOW_MIN_DEFAULT: int = 180
KRONOS_TIMEOUT_SECONDS_DEFAULT: float = 5.0
MIN_KRONOS_ROWS: int = 512

STRATEGY_ID: str = "kronos_sizing"


# ===========================================================================
# LAYER 2 — KRONOS POSITION-SIZING OVERLAY (pure functions first)
# ===========================================================================
def kronos_strength_weight(strength_pct: float) -> float:
    """Frozen mapping: absolute forecast strength (%) -> position size weight.

    strength_pct < 0.050         -> 0.50x
    0.050 <= strength_pct < 0.075 -> 0.75x
    strength_pct >= 0.075         -> 1.00x

    Pure and unit-testable. Raises ValueError on non-finite input.
    """
    if strength_pct is None or not math.isfinite(strength_pct) or strength_pct < 0:
        raise ValueError(f"invalid strength_pct: {strength_pct!r}")
    if strength_pct < STRENGTH_LOW:
        return WEIGHT_LOW
    if strength_pct < STRENGTH_HIGH:
        return WEIGHT_MID
    return WEIGHT_HIGH


def forecast_price_from_raw(raw_predictions: List[float]) -> Optional[float]:
    """forecast_price = mean(raw_predictions), in the same units as last_close.

    Kronos raw_predictions are PRICE forecasts (e.g. 24512.25 for NIFTY), not
    percentage returns. Returns None when raw_predictions is empty/None ->
    strength cannot be computed -> NO TRADE (fail-closed). Non-finite entries
    -> None.
    """
    if not raw_predictions:
        return None
    vals = [v for v in raw_predictions if v is not None and math.isfinite(v)]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def forecast_return_pct_from_raw(
    raw_predictions: List[float], last_close: Optional[float]
) -> Optional[float]:
    """forecast_return_pct = (forecast_price - last_close) / last_close * 100.

    raw_predictions are PRICE forecasts (same units as last_close), so the
    percentage return is anchored on the last close Kronos saw, NOT the entry
    price. Never use mean(raw_predictions) directly as the return.
    Returns None when raw_predictions is empty/None or last_close is
    missing/non-finite/<= 0 -> strength cannot be computed -> NO TRADE
    (fail-closed).
    """
    forecast_price = forecast_price_from_raw(raw_predictions)
    if forecast_price is None:
        return None
    if last_close is None or not math.isfinite(last_close) or last_close <= 0:
        return None
    return (forecast_price - last_close) / last_close * 100.0


def strength_from_raw(
    raw_predictions: List[float], last_close: Optional[float]
) -> Optional[float]:
    """strength_pct = abs(forecast_return_pct); None -> cannot compute -> NO TRADE."""
    forecast = forecast_return_pct_from_raw(raw_predictions, last_close)
    if forecast is None:
        return None
    return abs(forecast)


@dataclass
class Forecast:
    """Result of one Kronos forecast call, with causality timestamps."""

    success: bool
    last_close: Optional[float] = None
    forecast_price: Optional[float] = None
    forecast_return_pct: Optional[float] = None
    strength_pct: Optional[float] = None
    signal: Optional[int] = None
    confidence: Optional[float] = None
    inference_ms: Optional[float] = None
    request_ts: Optional[str] = None
    response_ts: Optional[str] = None
    error: Optional[str] = None


class KronosClient:
    """Reuses the existing Kronos server contract: POST /predict.

    Contract mirrors kronos/client.py (same payload, same response parsing)
    but with the frozen KRONOS_TIMEOUT_SECONDS=5 default and a client-side
    staleness check (the server returns no timestamp field).
    """

    def __init__(self, base_url: str, timeout_seconds: float = KRONOS_TIMEOUT_SECONDS_DEFAULT):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def get_forecast(
        self,
        ohlcv: pd.DataFrame,
        freq: Optional[str] = None,
        profile: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> Forecast:
        """POST /predict with OHLCV records (most recent last).

        Only data available at signal time is sent — the caller passes the
        exact df already used for the base signal. Never future data.
        Fail-closed: timeout/malformed/stale -> success=False.
        """
        request_ts = (now or datetime.now()).isoformat(timespec="milliseconds")

        # ---- build payload exactly like kronos/client.py predict() ----
        cols = ["open", "high", "low", "close"]
        df = ohlcv[cols + (["volume"] if "volume" in ohlcv.columns else [])].copy()
        df = df.dropna(subset=cols)
        last_close = float(df["close"].iloc[-1])
        if len(df) < MIN_KRONOS_ROWS:
            return Forecast(
                success=False,
                request_ts=request_ts,
                error=f"insufficient rows for Kronos: {len(df)} < {MIN_KRONOS_ROWS}",
            )
        records = df.to_dict(orient="records")
        body: Dict[str, Any] = {"data": records}
        if freq:
            body["freq"] = freq
        if profile:
            body["profile"] = profile

        # ---- call ----
        started = time.monotonic()
        try:
            resp = requests.post(
                f"{self.base_url}/predict",
                json=body,
                timeout=(self.timeout_seconds, self.timeout_seconds),
            )
            elapsed_s = time.monotonic() - started
            response_ts = datetime.now().isoformat(timespec="milliseconds")
        except requests.RequestException as exc:
            return Forecast(
                success=False,
                request_ts=request_ts,
                error=f"Kronos unreachable/timeout: {exc.__class__.__name__}: {exc}",
            )

        # ---- staleness (client-side wall-clock; spec 7) ----
        if elapsed_s > self.timeout_seconds:
            return Forecast(
                success=False,
                request_ts=request_ts,
                response_ts=response_ts,
                error=f"Kronos stale response: {elapsed_s:.2f}s > {self.timeout_seconds}s timeout",
            )
        if resp.status_code != 200:
            return Forecast(
                success=False,
                request_ts=request_ts,
                response_ts=response_ts,
                error=f"Kronos HTTP {resp.status_code}: {resp.text[:200]}",
            )

        # ---- validate response shape ----
        try:
            data = resp.json()
        except ValueError:
            return Forecast(
                success=False,
                request_ts=request_ts,
                response_ts=response_ts,
                error="Kronos malformed response: not JSON",
            )
        if not isinstance(data, dict) or data.get("prediction") is None:
            return Forecast(
                success=False,
                request_ts=request_ts,
                response_ts=response_ts,
                error="Kronos malformed response: missing prediction field",
            )

        raw = data.get("raw_predictions", [])
        forecast_price = forecast_price_from_raw(raw)
        forecast = forecast_return_pct_from_raw(raw, last_close)
        strength = strength_from_raw(raw, last_close)
        if forecast_price is None or forecast is None or strength is None:
            return Forecast(
                success=False,
                last_close=last_close,
                request_ts=request_ts,
                response_ts=response_ts,
                error="Kronos empty raw_predictions: cannot compute strength (fail-closed)",
            )

        return Forecast(
            success=True,
            last_close=last_close,
            forecast_price=round(forecast_price, 6),
            forecast_return_pct=round(forecast, 6),
            strength_pct=round(strength, 6),
            signal=int(data["prediction"]),
            confidence=float(data.get("confidence", 0.0)),
            inference_ms=float(data.get("inference_ms", 0.0)),
            request_ts=request_ts,
            response_ts=response_ts,
        )


# ===========================================================================
# LAYER 1 — BASE SIGNAL ENGINE
# PROVENANCE: C (NEWLY DESIGNED). See module docstring before going live.
# ===========================================================================
def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder-smoothed Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


@dataclass
class BaseSignal:
    """Output of the base signal engine: direction, entry, ATR (and provenance)."""

    direction: int          # +1 long, -1 short, 0 none
    entry: Optional[float]  # reference price at signal time
    atr: Optional[float]    # ATR at signal time (drives SL/TP)
    signal_time: Optional[str] = None  # causality: when the signal fired
    reason: str = ""


def base_signal_engine(
    df: pd.DataFrame,
    atr_period: int = 14,
    ema_fast: int = 9,
    ema_slow: int = 21,
    signal_time: Optional[str] = None,
) -> BaseSignal:
    """NEWLY DESIGNED base engine (provenance C).

    EMA fast/slow trend filter with a crossover entry on the last closed bar:
      - close crosses above EMA_fast while EMA_fast > EMA_slow  -> long (+1)
      - close crosses below EMA_fast while EMA_fast < EMA_slow  -> short (-1)
      - otherwise -> flat (0)
    entry = last close; atr = ATR(last bar). Frozen SL/TP are applied by the
    execution layer (1.0 ATR / 2.5 ATR / 180-min window) — NOT here.
    """
    if df is None or len(df) < max(atr_period + 1, ema_slow + 1):
        return BaseSignal(direction=0, entry=None, atr=None, reason="insufficient bars")

    close = df["close"]
    fast = close.ewm(span=ema_fast, adjust=False).mean()
    slow = close.ewm(span=ema_slow, adjust=False).mean()
    atr = compute_atr(df, atr_period)

    prev_fast, prev_slow = fast.iloc[-2], slow.iloc[-2]
    cur_fast, cur_slow = fast.iloc[-1], slow.iloc[-1]
    cur_close = float(close.iloc[-1])
    cur_atr = float(atr.iloc[-1])

    if cur_atr is None or not math.isfinite(cur_atr) or cur_atr <= 0:
        return BaseSignal(direction=0, entry=None, atr=None, reason="invalid ATR")

    crossed_up = prev_fast <= prev_slow and cur_fast > cur_slow
    crossed_dn = prev_fast >= prev_slow and cur_fast < cur_slow

    if crossed_up:
        return BaseSignal(
            direction=+1,
            entry=cur_close,
            atr=cur_atr,
            signal_time=signal_time,
            reason="ema fast cross above slow, bullish filter",
        )
    if crossed_dn:
        return BaseSignal(
            direction=-1,
            entry=cur_close,
            atr=cur_atr,
            signal_time=signal_time,
            reason="ema fast cross below slow, bearish filter",
        )
    return BaseSignal(direction=0, entry=None, atr=None, reason="no cross on last bar")


# ===========================================================================
# EXECUTION LAYER — frozen SL/TP/window, sizing, state machine, order safety
# ===========================================================================
class State:
    IDLE = "IDLE"
    SIGNAL_DETECTED = "SIGNAL_DETECTED"
    KRONOS_PENDING = "KRONOS_PENDING"
    ORDER_PENDING = "ORDER_PENDING"
    POSITION_OPEN = "POSITION_OPEN"
    EXIT_PENDING = "EXIT_PENDING"
    FLAT = "FLAT"
    ERROR = "ERROR"


@dataclass
class Position:
    direction: int                # +1 long, -1 short
    symbol: str
    exchange: str
    qty: int
    entry_price: float
    atr: float
    sl: float                     # frozen: entry -/+ 1.0 ATR
    tp: float                     # frozen: entry +/- 2.5 ATR
    entry_ts: str                 # causality / window start
    entry_epoch: float
    order_id: Optional[str] = None
    weight: float = WEIGHT_HIGH
    strength_pct: Optional[float] = None
    forecast_return_pct: Optional[float] = None


def round_down_to_lot(qty_float: float, lot_size: int) -> int:
    """Floor to lot-size multiple. Returns 0 when below one lot (invalid qty)."""
    if lot_size <= 0 or qty_float is None or not math.isfinite(qty_float) or qty_float <= 0:
        return 0
    return int(math.floor(qty_float / lot_size)) * lot_size


class KronosSizingStrategy:
    """Main strategy: reconciles state machine with the OpenAlgo position book."""

    def __init__(self, client: Any, env: Dict[str, str] | None = None, owner: PositionOwner | None = None):
        self.client = client
        env = os.environ if env is None else {**os.environ, **env}
        self.env = env

        self._owner = owner
        self._owned_position_id: Optional[str] = None

        # platform/host contract
        self.api_key = env.get("OPENALGO_API_KEY", "")
        self.host = env.get("HOST_SERVER") or env.get("OPENALGO_HOST") or "https://skopaq.duckdns.org"
        self.ws_url = env.get("WEBSOCKET_URL") or env.get("WS_URL") or "ws://127.0.0.1:8765"

        self.strategy_name = env.get("STRATEGY_NAME", "kronos_sizing")
        self.mode = env.get("MODE", "validate").lower()
        self.base_engine_validated = env.get("BASE_ENGINE_VALIDATED", "false").lower() == "true"

        self.underlying = env.get("UNDERLYING", "NIFTY")
        self.exchange = env.get("EXCHANGE", "NFO")
        self.base_interval = env.get("BASE_INTERVAL", "5m")
        self.lookback_days = int(env.get("BASE_LOOKBACK_DAYS", "10"))
        self.kronos_context_rows = int(env.get("KRONOS_CONTEXT_ROWS", str(MIN_KRONOS_ROWS)))
        self.kronos_url = env.get("KRONOS_URL", "http://127.0.0.1:8000")
        self.kronos_timeout = float(env.get("KRONOS_TIMEOUT_SECONDS", str(KRONOS_TIMEOUT_SECONDS_DEFAULT)))
        self.kronos_freq = env.get("KRONOS_FREQ", "5min")
        self.kronos_profile = env.get("KRONOS_PROFILE", "fast")

        self.base_qty = int(env.get("BASE_QTY", "75"))
        self.product = env.get("PRODUCT", "MIS")
        self.price_type = env.get("PRICE_TYPE", "MARKET")
        self.atr_period = int(env.get("ATR_PERIOD", "14"))
        self.sl_atr = float(env.get("SL_ATR", str(SL_ATR_DEFAULT)))
        self.tp_atr = float(env.get("TP_ATR", str(TP_ATR_DEFAULT)))
        self.trailing_sl_atr = float(env.get("TRAILING_SL_ATR", str(TRAILING_SL_ATR_DEFAULT)))
        self.trailing_activation_atr = float(env.get("TRAILING_ACTIVATION_ATR", str(TRAILING_ACTIVATION_ATR_DEFAULT)))
        self.max_window_min = int(env.get("MAX_EXECUTION_WINDOW_MIN", str(MAX_EXECUTION_WINDOW_MIN_DEFAULT)))
        self.cadence_s = float(env.get("CADENCE_SECONDS", "60"))
        self.stale_market_s = float(env.get("STALE_MARKET_SECONDS", "300"))
        self.max_entry_retries = int(env.get("MAX_ENTRY_RETRIES", "3"))

        self.data_dir = env.get("DATA_DIR", "./kronos_sizing_data")
        os.makedirs(self.data_dir, exist_ok=True)
        self.decision_log = os.path.join(self.data_dir, "decisions.jsonl")
        self.state_file = os.path.join(self.data_dir, "state.json")

        self.kronos = KronosClient(self.kronos_url, self.kronos_timeout)

        # runtime state
        self.state = State.IDLE
        self.symbol: Optional[str] = None
        self.lot_size: Optional[int] = None
        self.position: Optional[Position] = None
        self.last_signal: Optional[BaseSignal] = None
        self.last_forecast: Optional[Forecast] = None
        self.inflight_order_id: Optional[str] = None
        self.entry_retries = 0
        self._order_seq = 0
        self._stop = False

    # ---------------- lifecycle ----------------
    def start(self) -> None:
        if not self.api_key:
            print("FATAL: OPENALGO_API_KEY is required (platform injects it). Exiting.", flush=True)
            sys.exit(1)
        if self.mode == "live" and not self.base_engine_validated:
            print(
                "FATAL: MODE=live requires BASE_ENGINE_VALIDATED=true. The base signal engine "
                "is provenance C (newly designed) and has NOT been demonstrated to match the "
                "historically validated trade-generation logic. Refusing to trade. "
                "Run MODE=validate until the engine is validated.",
                flush=True,
            )
            sys.exit(1)
        print(
            f"[{self.strategy_name}] mode={self.mode} host={self.host} underlying={self.underlying} "
            f"exchange={self.exchange} interval={self.base_interval} "
            f"SL={self.sl_atr}ATR TP={self.tp_atr}ATR window={self.max_window_min}min "
            f"kronos={self.kronos_url} timeout={self.kronos_timeout}s",
            flush=True,
        )
        self._load_state()

    def stop(self) -> None:
        self._stop = True

    # ---------------- instrument resolution ----------------
    def resolve_symbol_and_lot(self) -> bool:
        """Resolve nearest NIFTY future symbol + lot size from instrument metadata.

        Uses client.symbol() (instrument metadata) per spec 4 — never hardcoded.
        """
        try:
            expiry_resp = self.client.expiry(symbol=self.underlying, exchange=self.exchange, instrumenttype="futures")
            expiries = expiry_resp.get("data", expiry_resp) if isinstance(expiry_resp, dict) else expiry_resp
            if isinstance(expiries, dict):
                expiries = expiries.get("expiry", []) or expiries.get("data", [])
            if not expiries:
                print(f"[{self.strategy_name}] ERROR: no futures expiry for {self.underlying}", flush=True)
                return False
            expiries = sorted(expiries)
            near = expiries[0]
            near_str = str(near).replace("-", "")
            self.symbol = f"{self.underlying}{near_str}FUT"
        except Exception as exc:
            print(f"[{self.strategy_name}] ERROR: expiry lookup failed: {exc}", flush=True)
            return False

        try:
            info = self.client.symbol(symbol=self.symbol, exchange=self.exchange)
            info = info.get("data", info) if isinstance(info, dict) else info
            lot = int(info.get("lotsize", 0) or 0)
            if lot <= 0:
                print(f"[{self.strategy_name}] ERROR: lot size missing/zero for {self.symbol}", flush=True)
                return False
            self.lot_size = lot
            print(f"[{self.strategy_name}] resolved symbol={self.symbol} lotsize={lot}", flush=True)
            return True
        except Exception as exc:
            print(f"[{self.strategy_name}] ERROR: symbol lookup failed: {exc}", flush=True)
            return False

    # ---------------- data ----------------
    def fetch_ohlcv(self) -> Optional[pd.DataFrame]:
        end = datetime.now()
        start = end - timedelta(days=self.lookback_days)
        try:
            df = self.client.history(
                symbol=self.symbol,
                exchange=self.exchange,
                interval=self.base_interval,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            print(f"[{self.strategy_name}] ERROR: history fetch failed: {exc}", flush=True)
            return None
        if df is None or len(df) < MIN_KRONOS_ROWS:
            print(
                f"[{self.strategy_name}] ERROR: history rows {0 if df is None else len(df)} < {MIN_KRONOS_ROWS} "
                f"(increase BASE_LOOKBACK_DAYS)",
                flush=True,
            )
            return None
        return df.tail(self.kronos_context_rows)

    def live_price(self) -> Optional[float]:
        try:
            q = self.client.quotes(symbol=self.symbol, exchange=self.exchange)
            data = q.get("data", q) if isinstance(q, dict) else q
            if isinstance(data, dict):
                return float(data.get("ltp") or data.get("last_price") or 0.0) or None
            return float(data) if data else None
        except Exception:
            return None

    # ---------------- position reconciliation (spec 11) ----------------
    def actual_net_qty(self) -> int:
        try:
            pb = self.client.positionbook()
            rows = pb.get("data", pb) if isinstance(pb, dict) else pb
            if not isinstance(rows, list):
                return 0
            for r in rows:
                if isinstance(r, dict) and r.get("symbol") == self.symbol and self._is_ours(r):
                    return int(r.get("netqty") or r.get("net_quantity") or r.get("quantity") or 0)
            return 0
        except Exception:
            return 0

    def _is_ours(self, row: Dict[str, Any]) -> bool:
        strat = str(row.get("strategy") or "").lower()
        sym = str(row.get("symbol") or "").lower()
        return (strat == self.strategy_name.lower() or not strat) and sym == str(self.symbol).lower()

    def reconcile(self) -> bool:
        """Local state vs OpenAlgo actual position. Mismatch -> ERROR halt.

        Returns True when consistent.

        In validate mode orders are simulated, so the broker position book is
        intentionally unchanged — the strict mismatch check is skipped and the
        full entry/exit lifecycle still runs for validation.
        """
        if self.mode == "validate":
            return True
        actual = self.actual_net_qty()
        expect_open = self.position is not None and self.state in (
            State.POSITION_OPEN, State.EXIT_PENDING, State.ORDER_PENDING
        )
        if expect_open and actual == 0:
            print(
                f"[{self.strategy_name}] ERROR STATE: local state={self.state} expects position "
                f"on {self.symbol} but OpenAlgo net qty=0. Mismatch -> halting entries.",
                flush=True,
            )
            self.state = State.ERROR
            return False
        if not expect_open and actual != 0:
            print(
                f"[{self.strategy_name}] ERROR STATE: local state={self.state} expects flat "
                f"but OpenAlgo net qty={actual} on {self.symbol}. Mismatch -> halting entries.",
                flush=True,
            )
            self.state = State.ERROR
            return False
        return True

    # ---------------- order safety (spec 12) ----------------
    def _next_order_id(self, kind: str) -> str:
        self._order_seq += 1
        return f"{self.strategy_name}-{kind}-{int(time.time() * 1000)}-{self._order_seq}"

    def _order_is_settled(self, order_id: str) -> Optional[str]:
        """Return 'open', 'complete' or None if unknown — check BEFORE resending."""
        try:
            st = self.client.orderstatus(order_id=order_id, strategy=self.strategy_name)
            data = st.get("data", st) if isinstance(st, dict) else st
            status = str((data.get("order_status") if isinstance(data, dict) else data) or "").upper()
            if "COMPLETE" in status or "FILLED" in status or status in ("FILLED", "COMPLETE"):
                return "complete"
            if status in ("OPEN", "PENDING", "TRIGGER_PENDING", "NEW"):
                return "open"
            if status in ("REJECTED", "CANCELLED", "CANCELED", "EXPIRED"):
                return status.lower()
            return "open" if status else None
        except Exception:
            return None

    def place_entry(self, direction: int, qty: int, entry_price: float, atr: float) -> Optional[str]:
        action = "BUY" if direction == +1 else "SELL"
        order_id = self._next_order_id("entry")
        try:
            resp = self.client.placeorder(
                strategy=self.strategy_name,
                symbol=self.symbol,
                action=action,
                exchange=self.exchange,
                price_type=self.price_type,
                product=self.product,
                quantity=qty,
            )
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            returned_id = data.get("orderid") if isinstance(data, dict) else None
            print(
                f"[{self.strategy_name}] ENTRY ORDER id={returned_id or order_id} {action} qty={qty} "
                f"{self.symbol} ref_entry={entry_price:.2f}",
                flush=True,
            )
            return returned_id or order_id
        except Exception as exc:
            print(f"[{self.strategy_name}] ERROR: entry order failed: {exc}", flush=True)
            return None

    def place_exit(self, qty: int, reason: str) -> Optional[str]:
        if self.position is None:
            return None
        action = "SELL" if self.position.direction == +1 else "BUY"
        order_id = self._next_order_id("exit")
        try:
            resp = self.client.placeorder(
                strategy=self.strategy_name,
                symbol=self.symbol,
                action=action,
                exchange=self.exchange,
                price_type=self.price_type,
                product=self.product,
                quantity=qty,
            )
            data = resp.get("data", resp) if isinstance(resp, dict) else resp
            returned_id = data.get("orderid") if isinstance(data, dict) else None
            print(
                f"[{self.strategy_name}] EXIT ORDER id={returned_id or order_id} {action} qty={qty} "
                f"{self.symbol} reason={reason}",
                flush=True,
            )
            return returned_id or order_id
        except Exception as exc:
            print(f"[{self.strategy_name}] ERROR: exit order failed: {exc}", flush=True)
            return None

    # ---------------- state machine ----------------
    def run_once(self) -> None:
        """One cadence of the state machine (spec 9/10/11/12)."""

        # market-open sanity: refuse entries after square-off time if configured
        so = self.env.get("SQUARE_OFF_HHMM")
        if so:
            hh, mm = (int(x) for x in so.split(":"))
            if datetime.now().hour > hh or (datetime.now().hour == hh and datetime.now().minute >= mm):
                if self.position and self.state == State.POSITION_OPEN:
                    self._trigger_exit(f"square-off {so}")
                return

        if not self.reconcile():
            return
        if self.state == State.ERROR:
            return

        if self.state == State.IDLE:
            self._on_idle()
        elif self.state == State.SIGNAL_DETECTED:
            self._on_signal_detected()
        elif self.state == State.KRONOS_PENDING:
            self._on_kronos_pending()
        elif self.state == State.ORDER_PENDING:
            self._on_order_pending()
        elif self.state == State.POSITION_OPEN:
            self._on_position_open()
        elif self.state == State.EXIT_PENDING:
            self._on_exit_pending()
        elif self.state == State.FLAT:
            self.state = State.IDLE

    def _on_idle(self) -> None:
        if not self.reconcile():
            return
        if self.symbol is None or self.lot_size is None:
            if not self.resolve_symbol_and_lot():
                self.state = State.ERROR
                return
        df = self.fetch_ohlcv()
        if df is None:
            return  # stay IDLE; retry next cadence
        # market-data freshness guard
        last_ts = getattr(df.index, "max", None)
        if last_ts is not None:
            try:
                last_dt = pd.Timestamp(last_ts).to_pydatetime()
                if datetime.now() - last_dt > timedelta(seconds=self.stale_market_s):
                    print(
                        f"[{self.strategy_name}] WARN: market data stale ({last_dt}), "
                        f"skipping entries this cadence",
                        flush=True,
                    )
                    return
            except Exception:
                pass

        sig = base_signal_engine(
            df, atr_period=self.atr_period, signal_time=datetime.now().isoformat(timespec="milliseconds")
        )
        if sig.direction == 0:
            return
        # entry-only evaluation: Kronos is consulted only at entry (spec 10)
        self.last_signal = sig
        self.state = State.SIGNAL_DETECTED
        print(
            f"[{self.strategy_name}] SIGNAL dir={sig.direction:+d} entry={sig.entry:.2f} "
            f"atr={sig.atr:.2f} t={sig.signal_time} reason={sig.reason}",
            flush=True,
        )
        self._log_decision({"stage": "signal", **self._signal_dict(sig)})

    def _on_signal_detected(self) -> None:
        """Transition SIGNAL_DETECTED -> KRONOS_PENDING (forecast is next)."""
        if self.last_signal is None:
            self.state = State.IDLE
            return
        self.state = State.KRONOS_PENDING

    def _on_kronos_pending(self) -> None:
        """Kronos forecast at entry only. Fail-closed: any failure -> NO TRADE -> IDLE."""
        sig = self.last_signal
        if sig is None:
            self.state = State.IDLE
            return
        df = self.fetch_ohlcv()
        if df is None:
            self.state = State.IDLE
            return
        forecast = self.kronos.get_forecast(
            df, freq=self.kronos_freq, profile=self.kronos_profile, now=datetime.now()
        )
        self.last_forecast = forecast
        if not forecast.success:
            self._log_decision({
                "stage": "forecast",
                "signal_time": sig.signal_time,
                "forecast_request_time": forecast.request_ts,
                "forecast_response_time": forecast.response_ts,
                "last_close": forecast.last_close,
                "forecast_price": forecast.forecast_price,
                "forecast_return_pct": forecast.forecast_return_pct,
                "strength_pct": forecast.strength_pct,
                "selected_weight": None,
                "success": forecast.success,
                "error": forecast.error,
            })
            print(
                f"[{self.strategy_name}] NO TRADE (fail-closed): {forecast.error} "
                f"req={forecast.request_ts} resp={forecast.response_ts}",
                flush=True,
            )
            self.state = State.IDLE
            return
        weight = kronos_strength_weight(forecast.strength_pct)  # type: ignore[arg-type]
        qty = round_down_to_lot(self.base_qty * weight, self.lot_size or 0)
        self._log_decision({
            "stage": "forecast",
            "signal_time": sig.signal_time,
            "forecast_request_time": forecast.request_ts,
            "forecast_response_time": forecast.response_ts,
            "last_close": forecast.last_close,
            "forecast_price": forecast.forecast_price,
            "forecast_return_pct": forecast.forecast_return_pct,
            "strength_pct": forecast.strength_pct,
            "selected_weight": weight,
            "success": forecast.success,
            "error": forecast.error,
        })
        print(
            f"[{self.strategy_name}] KRONOS last_close={forecast.last_close:.2f} "
            f"forecast_price={forecast.forecast_price:.2f} "
            f"forecast_return_pct={forecast.forecast_return_pct:.6f} "
            f"strength_pct={forecast.strength_pct:.6f} "
            f"selected_weight={weight:.2f} qty={qty} "
            f"(base={self.base_qty} lot={self.lot_size})",
            flush=True,
        )
        if qty <= 0:
            print(f"[{self.strategy_name}] NO TRADE: qty {qty} below one lot ({self.lot_size})", flush=True)
            self.state = State.IDLE
            return
        self.position = Position(
            direction=sig.direction,
            symbol=self.symbol or "",
            exchange=self.exchange,
            qty=qty,
            entry_price=sig.entry or 0.0,
            atr=sig.atr or 0.0,
            sl=(sig.entry or 0.0) - sig.direction * self.sl_atr * (sig.atr or 0.0),
            tp=(sig.entry or 0.0) + sig.direction * self.tp_atr * (sig.atr or 0.0),
            entry_ts=sig.signal_time or "",
            entry_epoch=time.time(),
            weight=weight,
            strength_pct=forecast.strength_pct,
            forecast_return_pct=forecast.forecast_return_pct,
        )
        self.entry_retries = 0
        self.state = State.ORDER_PENDING
        self._save_state()

    def _on_order_pending(self) -> None:
        """Idempotent entry: check existing order before resending (spec 12)."""
        pos = self.position
        if pos is None:
            self.state = State.IDLE
            return
        if pos.order_id:
            settled = self._order_is_settled(pos.order_id)
            if settled == "complete" or self.actual_net_qty() != 0:
                print(f"[{self.strategy_name}] ENTRY FILLED order={pos.order_id} qty={pos.qty}", flush=True)
                self.state = State.POSITION_OPEN
                if self._owner:
                    try:
                        pid = self._owner.register_position(
                            STRATEGY_ID, pos.symbol, pos.exchange, pos.qty, self.product,
                            entry_order_id=pos.order_id or "",
                        )
                        self._owned_position_id = pid
                        print(f"[{self.strategy_name}] registered ownership {pos.symbol} -> {pid}", flush=True)
                    except Exception as exc:
                        print(f"[{self.strategy_name}] ownership registration failed ({exc})", flush=True)
                self._save_state()
                return
            if settled in ("rejected", "cancelled", "expired"):
                print(f"[{self.strategy_name}] ENTRY {settled} order={pos.order_id}", flush=True)
                pos.order_id = None
                self.entry_retries += 1
                if self.entry_retries > self.max_entry_retries:
                    self.state = State.ERROR
                    return
                return  # retry next cadence (idempotent — no duplicate send)
            if settled == "open":
                return  # still working — do NOT resend
        if self.mode == "validate":
            print(
                f"[{self.strategy_name}] [VALIDATE] would place ENTRY dir={pos.direction:+d} "
                f"qty={pos.qty} {pos.symbol} sl={pos.sl:.2f} tp={pos.tp:.2f}",
                flush=True,
            )
            self.state = State.POSITION_OPEN  # simulate open in validate mode
            self._save_state()
            return
        oid = self.place_entry(pos.direction, pos.qty, pos.entry_price, pos.atr)
        if oid:
            pos.order_id = oid
            self.entry_retries += 1
            self._save_state()
        # else: retried next cadence

    def _on_position_open(self) -> None:
        """Frozen exits only: SL / TP / max window (spec 3, 10). No trailing unless configured."""
        pos = self.position
        if pos is None:
            self.state = State.IDLE
            return
        price = self.live_price()
        if price is None:
            df = self.fetch_ohlcv()
            if df is not None and len(df):
                price = float(df["close"].iloc[-1])
        if price is None:
            print(f"[{self.strategy_name}] WARN: no price for exit check, retrying", flush=True)
            return
        elapsed_min = (time.time() - pos.entry_epoch) / 60.0
        # frozen: SL/TP in ATR terms, no trailing by default
        sl = pos.sl
        tp = pos.tp
        if pos.direction == +1:
            hit_sl = price <= sl
            hit_tp = price >= tp
        else:
            hit_sl = price >= sl
            hit_tp = price <= tp
        reason = None
        if hit_sl:
            reason = f"SL {sl:.2f} (price {price:.2f})"
        elif hit_tp:
            reason = f"TP {tp:.2f} (price {price:.2f})"
        elif elapsed_min >= self.max_window_min:
            reason = f"max window {self.max_window_min}min elapsed"
        if reason:
            self._trigger_exit(reason)

    def _trigger_exit(self, reason: str) -> None:
        self.state = State.EXIT_PENDING
        print(f"[{self.strategy_name}] EXIT TRIGGER: {reason}", flush=True)

    def _on_exit_pending(self) -> None:
        pos = self.position
        if pos is None:
            self.state = State.IDLE
            return
        if self.mode == "validate":
            print(f"[{self.strategy_name}] [VALIDATE] would EXIT qty={pos.qty} {pos.symbol}", flush=True)
            self._clear_position()
            self.state = State.FLAT
            return
        if pos.order_id is None:
            oid = self.place_exit(pos.qty, reason="exit")
            if oid:
                pos.order_id = oid
                self._save_state()
            return
        settled = self._order_is_settled(pos.order_id)
        if settled == "complete" or self.actual_net_qty() == 0:
            print(f"[{self.strategy_name}] EXIT FILLED order={pos.order_id}", flush=True)
            self._clear_position()
            self.state = State.FLAT
            return
        if settled in ("rejected", "cancelled", "expired"):
            print(f"[{self.strategy_name}] EXIT {settled} order={pos.order_id} retrying", flush=True)
            pos.order_id = None
            self._save_state()

    # ---------------- persistence / logging ----------------
    def _signal_dict(self, sig: BaseSignal) -> Dict[str, Any]:
        return {
            "direction": sig.direction,
            "entry": sig.entry,
            "atr": sig.atr,
            "signal_time": sig.signal_time,
            "reason": sig.reason,
        }

    def _log_decision(self, record: Dict[str, Any]) -> None:
        record["ts"] = datetime.now().isoformat(timespec="milliseconds")
        try:
            with open(self.decision_log, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            print(f"[{self.strategy_name}] WARN: decision log write failed: {exc}", flush=True)

    def _save_state(self) -> None:
        pos = self.position
        payload = {
            "state": self.state,
            "symbol": self.symbol,
            "lot_size": self.lot_size,
            "inflight_order_id": self.inflight_order_id,
            "position": None if pos is None else {
                "direction": pos.direction, "symbol": pos.symbol, "exchange": pos.exchange,
                "qty": pos.qty, "entry_price": pos.entry_price, "atr": pos.atr,
                "sl": pos.sl, "tp": pos.tp, "entry_ts": pos.entry_ts,
                "entry_epoch": pos.entry_epoch, "order_id": pos.order_id,
                "weight": pos.weight, "strength_pct": pos.strength_pct,
                "forecast_return_pct": pos.forecast_return_pct,
            },
        }
        try:
            with open(self.state_file, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            print(f"[{self.strategy_name}] WARN: state save failed: {exc}", flush=True)

    def _load_state(self) -> None:
        """Restore position/state across restarts; reconcile will verify."""
        try:
            with open(self.state_file, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, ValueError):
            return
        pos_data = payload.get("position")
        if isinstance(pos_data, dict):
            self.position = Position(
                direction=int(pos_data["direction"]),
                symbol=pos_data["symbol"],
                exchange=pos_data["exchange"],
                qty=int(pos_data["qty"]),
                entry_price=float(pos_data["entry_price"]),
                atr=float(pos_data["atr"]),
                sl=float(pos_data["sl"]),
                tp=float(pos_data["tp"]),
                entry_ts=pos_data["entry_ts"],
                entry_epoch=float(pos_data["entry_epoch"]),
                order_id=pos_data.get("order_id"),
                weight=float(pos_data.get("weight", WEIGHT_HIGH)),
                strength_pct=pos_data.get("strength_pct"),
                forecast_return_pct=pos_data.get("forecast_return_pct"),
            )
            self.state = payload.get("state", State.POSITION_OPEN)
            self.symbol = payload.get("symbol")
            self.lot_size = payload.get("lot_size")
            self.inflight_order_id = payload.get("inflight_order_id")
            print(
                f"[{self.strategy_name}] restored state={self.state} "
                f"pos={self.position.direction:+d}x{self.position.qty} {self.position.symbol}",
                flush=True,
            )

    def _clear_position(self) -> None:
        if self._owner and self._owned_position_id:
            try:
                self._owner.release_position(self._owned_position_id, STRATEGY_ID)
                print(f"[{self.strategy_name}] released ownership {self._owned_position_id}", flush=True)
            except Exception as exc:
                print(f"[{self.strategy_name}] ownership release failed ({exc})", flush=True)
            self._owned_position_id = None
        self.position = None
        self.last_signal = None
        self.last_forecast = None
        self.entry_retries = 0
        self._save_state()

    def shutdown_square_off(self) -> None:
        """On SIGTERM/SIGINT: square off an open position (best-effort)."""
        if self.position is None:
            print(f"[{self.strategy_name}] shutdown: flat, exiting", flush=True)
            return
        print(f"[{self.strategy_name}] shutdown with open position, squaring off", flush=True)
        if self.mode == "validate":
            print(f"[{self.strategy_name}] [VALIDATE] would EXIT qty={self.position.qty}", flush=True)
            return
        oid = self.place_exit(self.position.qty, reason="shutdown")
        if oid:
            time.sleep(2)
            print(f"[{self.strategy_name}] shutdown exit order={oid}", flush=True)


# ===========================================================================
# MAIN — /python self-hosted entry point
# ===========================================================================
def _signal_handler(strategy: KronosSizingStrategy, signum: int, frame: Any) -> None:
    print(f"[{strategy.strategy_name}] received signal {signum}, stopping", flush=True)
    strategy.stop()


def main() -> None:
    from openalgo import api

    register_strategy_mode(STRATEGY_ID, ExecutionMode.INTRADAY)
    stop_mode = StopMode.STOP_AND_CLOSE
    persister = StatePersister()
    owner = PositionOwner(persister)
    reconciler = Reconciler(persister, owner)

    strategy = KronosSizingStrategy(client=None, owner=owner)
    strategy.start()

    try:
        strategy.client = api(api_key=strategy.api_key, host=strategy.host, ws_url=strategy.ws_url)
    except Exception as exc:
        print(f"FATAL: OpenAlgo client init failed: {exc}", flush=True)
        sys.exit(1)

    # Startup reconciliation — block if orphans found
    try:
        broker_positions = strategy.client.positionbook().get("data", [])
    except Exception as exc:
        print(f"[{strategy.strategy_name}] positionbook failed ({exc}) - skipping reconciliation", flush=True)
        broker_positions = []
    result = reconciler.reconcile(STRATEGY_ID, broker_positions)
    if not result.is_clean:
        print(f"[{strategy.strategy_name}] reconciliation blocked: {result.blocked_reason}", flush=True)
        print(f"[{strategy.strategy_name}] resolve orphan positions before trading", flush=True)
    else:
        print(
            f"[{strategy.strategy_name}] reconciliation clean: {len(result.owned)} owned, "
            f"{len(result.stale)} stale",
            flush=True,
        )
    RecoveryManager(STRATEGY_ID, persister, owner, reconciler).startup(
        lambda: strategy.client.positionbook().get("data", []),
    )

    try:
        strategy.analyzer = strategy.client.analyzerstatus()
    except Exception:
        pass

    def _sigterm_handler(signum: int, frame: Any) -> None:
        print(f"[{strategy.strategy_name}] SIGTERM received (stop_mode={stop_mode.value})", flush=True)
        if stop_mode == StopMode.STOP_AND_CLOSE:
            strategy.stop()
            strategy.shutdown_square_off()
        else:
            print(f"[{strategy.strategy_name}] STOP_TRADING_ONLY - positions left open", flush=True)
            strategy.stop()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, _sigterm_handler)

    print(f"[{strategy.strategy_name}] starting cadence loop ({strategy.cadence_s}s)", flush=True)
    while not strategy._stop:
        try:
            strategy.run_once()
        except Exception as exc:
            print(f"[{strategy.strategy_name}] ERROR in cadence: {exc}", flush=True)
            strategy.state = State.ERROR
        time.sleep(strategy.cadence_s)

    strategy.shutdown_square_off()
    try:
        strategy.client.disconnect()
    except Exception:
        pass
    print(f"[{strategy.strategy_name}] stopped cleanly", flush=True)


if __name__ == "__main__":
    main()
