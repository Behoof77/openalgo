"""
Kronos ATM directional strategy (NIFTY options).

Trades ATM CE/PE purely on the Kronos regime vote:
    vote up     -> BUY ATM CE  (long, bull)
    vote down   -> BUY ATM PE  (long, bear)
    vote neutral-> no entry; any open position is exited
    vote change -> exit the current leg, then enter the new one
                   (only one entry open at any time)

No ML model, no feature engineering. The full signal chain is:

    KronosFilter (regime vote) -> DecisionEngine (state machine)
        -> RiskManager (funds / breaker) -> Execution (entry + SL/TARGET legs)

Host contract (OpenAlgo /python self-hosted page):
  - HOST_SERVER is honoured first, then OPENALGO_HOST.
  - OPENALGO_STRATEGY_EXCHANGE overrides the options exchange.
  - SIGTERM/SIGINT handlers stop the loop cleanly (square off in intraday).
  - Logging is stdout-only (the host captures stdout into per-run log files).
  - No asyncio: the OpenAlgo SDK is synchronous.

Config (all env-driven, sensible defaults):
  HOST, WS_URL, API_KEY, STRATEGY_NAME, UNDERLYING, SPOT_EXCHANGE,
  OPTIONS_EXCHANGE, EXCHANGE, EXPIRY_DATE, PRODUCT, PRICE_TYPE,
  QUANTITY, FALLBACK_LOT, TARGET_POINTS, STOP_POINTS, ROLL_THRESHOLD_PTS,
  ROLL_HYSTERESIS_PTS, KRONOS_URL, KRONOS_CADENCE_SECONDS, KRONOS_FREQ,
  KRONOS_CONTEXT_ROWS, KRONOS_BACKFILL_DAYS, KRONOS_PROFILE,
  DECISION_CADENCE_SECONDS, STALE_MARKET_SECONDS, SQUARE_OFF_HHMM,
  MAX_CONSECUTIVE_REJECTIONS, DAILY_LOSS_LIMIT_PTS, DATA_DIR,
  DECISION_LOG, STATE_FILE, MODE.

Components: DataAcquisition, KronosFilter, DecisionEngine, CircuitBreaker,
RiskManager, Execution.
"""

import json
import os
import signal
import sys
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dotenv import find_dotenv, load_dotenv
from openalgo import api

from strategies.shared import (
    ExecutionMode,
    PositionOwner,
    Reconciler,
    RecoveryManager,
    StatePersister,
    StopMode,
)
from strategies.shared.execution_modes import register_strategy_mode

load_dotenv(find_dotenv(), override=False)

# ---------------------------------------------------------------------------
# Single-instance guard (fail fast, BEFORE any network work).
# A second instance launched via UI, manual nohup, or scheduler exits in ~1s.
# The kernel releases the flock automatically when the holding process dies,
# so no stale-lock cleanup is ever needed.
# ---------------------------------------------------------------------------
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX host
    fcntl = None

_LOCK_PATH = os.path.join(
    os.environ.get("DATA_DIR") or "strategies/kronos_vote", ".instance.lock"
)
if fcntl is not None:
    os.makedirs(os.path.dirname(_LOCK_PATH), exist_ok=True)
    _lock_fd = open(_LOCK_PATH, "a+")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        _lock_fd.seek(0)
        _pid = _lock_fd.read().strip() or "unknown"
        print(
            f"[LOCK] another instance already running (pid={_pid}); exiting",
            flush=True,
        )
        sys.exit(0)
    _lock_fd.seek(0)
    _lock_fd.truncate()
    _lock_fd.write(str(os.getpid()))
    _lock_fd.flush()
else:
    _lock_fd = None

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST_SERVER") or os.environ.get("OPENALGO_HOST") or "http://127.0.0.1:5000"
WS_URL = os.environ.get("OPENALGO_WS_URL") or "ws://127.0.0.1:8765"
API_KEY = os.environ.get("OPENALGO_API_KEY") or ""

STRATEGY_NAME = os.environ.get("STRATEGY_NAME") or "kronos_vote"
STRATEGY_ID = os.environ.get("STRATEGY_ID") or "kronos_vote_1"
UNDERLYING = os.environ.get("UNDERLYING") or "NIFTY"
SPOT_EXCHANGE = os.environ.get("SPOT_EXCHANGE") or "NSE_INDEX"
OPTIONS_EXCHANGE = os.environ.get("OPTIONS_EXCHANGE") or "NFO"
EXCHANGE = os.environ.get("OPENALGO_STRATEGY_EXCHANGE") or OPTIONS_EXCHANGE
EXPIRY_DATE = os.environ.get("EXPIRY_DATE") or "11AUG26"

PRODUCT = os.environ.get("PRODUCT") or "MIS"
PRICE_TYPE = os.environ.get("PRICE_TYPE") or "LIMIT"
QUANTITY = int(os.environ.get("QUANTITY") or 65)
FALLBACK_LOT = int(os.environ.get("FALLBACK_LOT") or 65)  # NIFTY options lot (broker-validated)
TARGET_POINTS = float(os.environ.get("TARGET_POINTS") or 20)
STOP_POINTS = float(os.environ.get("STOP_POINTS") or 10)

ROLL_THRESHOLD_PTS = float(os.environ.get("ROLL_THRESHOLD_PTS") or 25)
ROLL_HYSTERESIS_PTS = float(os.environ.get("ROLL_HYSTERESIS_PTS") or 40)

KRONOS_URL = os.environ.get("KRONOS_URL") or "http://127.0.0.1:8000/predict"
KRONOS_CADENCE_SECONDS = float(os.environ.get("KRONOS_CADENCE_SECONDS") or 900)
KRONOS_FREQ = os.environ.get("KRONOS_FREQ") or "5m"
KRONOS_CONTEXT_ROWS = int(os.environ.get("KRONOS_CONTEXT_ROWS") or 512)
KRONOS_BACKFILL_DAYS = int(os.environ.get("KRONOS_BACKFILL_DAYS") or 14)
KRONOS_PROFILE = os.environ.get("KRONOS_PROFILE") or "fast"

DECISION_CADENCE_SECONDS = float(os.environ.get("DECISION_CADENCE_SECONDS") or 60)
STALE_MARKET_SECONDS = float(os.environ.get("STALE_MARKET_SECONDS") or 5)
SQUARE_OFF_HHMM = os.environ.get("SQUARE_OFF_HHMM") or "15:15"

MAX_CONSECUTIVE_REJECTIONS = int(os.environ.get("MAX_CONSECUTIVE_REJECTIONS") or 3)
DAILY_LOSS_LIMIT_PTS = float(os.environ.get("DAILY_LOSS_LIMIT_PTS") or 100)

DATA_DIR = os.environ.get("DATA_DIR") or "strategies/kronos_vote"
DECISION_LOG = os.environ.get("DECISION_LOG") or os.path.join(DATA_DIR, "decisions.jsonl")
STATE_FILE = os.environ.get("STATE_FILE") or os.path.join(DATA_DIR, "state.json")
MODE = os.environ.get("MODE") or ""
STOP_MODE = os.environ.get("STOP_MODE") or "STOP_AND_CLOSE"


# ---------------------------------------------------------------------------
# Live market snapshot (thread-shared, updated by the WS callback)
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    spot: float = 0.0
    ce_symbol: str = ""
    pe_symbol: str = ""
    ce_ltp: float = 0.0
    pe_ltp: float = 0.0
    atm_strike: int = 0
    lotsize: int = 0
    last_update: float = 0.0

    def fresh(self) -> bool:
        return time.time() - self.last_update <= STALE_MARKET_SECONDS


# ---------------------------------------------------------------------------
# Data acquisition: resolve ATM symbols, stream LTP, roll on drift
# ---------------------------------------------------------------------------
class DataAcquisition:
    def __init__(self, client, snapshot: Snapshot):
        self.client = client
        self.snapshot = snapshot
        self.instruments = [{"exchange": SPOT_EXCHANGE, "symbol": UNDERLYING}]
        self.lock = threading.Lock()
        # ATM roll hysteresis state: 0 = unset, +1 last roll up, -1 last roll down.
        self._roll_dir = 0
        self._hyst_logged = False

    @staticmethod
    def _extract_symbol(resp, what):
        if not resp:
            return None
        if isinstance(resp, dict) and resp.get("status") == "success" and resp.get("symbol"):
            return resp["symbol"]
        data = resp.get("data") if isinstance(resp, dict) else None
        if isinstance(data, dict):
            return data.get("symbol")
        print(f"[DATA] unexpected {what} response: {resp}", flush=True)
        return None

    def resolve_atm(self):
        with self.lock:
            try:
                ce = self.client.optionsymbol(
                    underlying=UNDERLYING,
                    exchange=SPOT_EXCHANGE,
                    expiry_date=EXPIRY_DATE,
                    offset="ATM",
                    option_type="CE",
                )
                pe = self.client.optionsymbol(
                    underlying=UNDERLYING,
                    exchange=SPOT_EXCHANGE,
                    expiry_date=EXPIRY_DATE,
                    offset="ATM",
                    option_type="PE",
                )
            except Exception as exc:
                print(f"[DATA] ATM resolve failed ({exc})", flush=True)
                return
            ce_sym = self._extract_symbol(ce, "CE")
            pe_sym = self._extract_symbol(pe, "PE")
            if not ce_sym or not pe_sym:
                print("[DATA] ATM resolve incomplete - retrying next cycle", flush=True)
                return
            self.snapshot.ce_symbol = ce_sym
            self.snapshot.pe_symbol = pe_sym
            self.snapshot.atm_strike = int(ce_sym[-7:-2])
            self.snapshot.lotsize = int(
                (ce.get("data") or {}).get("lotsize")
                or (ce.get("lotsize"))
                or FALLBACK_LOT
            )
            spot = (ce.get("data") or {}).get("underlying_ltp") or 0.0
            if spot:
                self.snapshot.spot = float(spot)
            self._roll_dir = 0
            self._hyst_logged = False
            print(
                f"[DATA] ATM {self.snapshot.atm_strike} | CE {ce_sym} | PE {pe_sym} | "
                f"lotsize {self.snapshot.lotsize} | spot {self.snapshot.spot:.2f}",
                flush=True,
            )
            self.instruments = [
                {"exchange": SPOT_EXCHANGE, "symbol": UNDERLYING},
                {"exchange": OPTIONS_EXCHANGE, "symbol": ce_sym},
                {"exchange": OPTIONS_EXCHANGE, "symbol": pe_sym},
            ]

    def _on_ltp(self, msg):
        symbol = getattr(msg, "symbol", None)
        data = getattr(msg, "data", None)
        if not symbol or data is None:
            return
        ltp = data.get("ltp")
        if ltp is None:
            return
        ts = data.get("timestamp") or 0
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0.0
        if ts > 1e12:  # ms epoch -> s
            ts = ts / 1000.0
        snap = self.snapshot
        if symbol == UNDERLYING:
            snap.spot = float(ltp)
        elif symbol == snap.ce_symbol:
            snap.ce_ltp = float(ltp)
        elif symbol == snap.pe_symbol:
            snap.pe_ltp = float(ltp)
        else:
            return
        snap.last_update = time.time()
        # ATM roll with hysteresis: same-direction drift rolls at ROLL_THRESHOLD,
        # a reversal must exceed ROLL_HYSTERESIS before we roll back.
        roll = False
        hyst_hold = False
        if snap.spot and snap.atm_strike:
            direction = 1 if snap.spot > snap.atm_strike else -1
            deviation = abs(snap.spot - snap.atm_strike)
            if self._roll_dir == 0 or direction == self._roll_dir:
                roll = deviation > ROLL_THRESHOLD_PTS
            else:
                roll = deviation > ROLL_HYSTERESIS_PTS
                hyst_hold = not roll
        if roll:
            self._roll_dir = 1 if snap.spot > snap.atm_strike else -1
            self._hyst_logged = False
            print(
                f"[DATA] spot {snap.spot:.2f} rolled past ATM {snap.atm_strike} -> re-resolving",
                flush=True,
            )
            self.resolve_atm()
            self.client.subscribe_ltp(self.instruments, self._on_ltp)
        elif hyst_hold and not self._hyst_logged:
            print(
                f"[DATA] roll hysteresis: spot {snap.spot:.2f} vs ATM {snap.atm_strike} "
                f"dev {abs(snap.spot - snap.atm_strike):.2f} < {ROLL_HYSTERESIS_PTS:.0f} "
                "reversal bound - holding ATM",
                flush=True,
            )
            self._hyst_logged = True

    def start(self):
        self.resolve_atm()
        self.client.connect()
        self.client.subscribe_ltp(self.instruments, self._on_ltp)


# ---------------------------------------------------------------------------
# Kronos regime vote (the ONLY signal source)
# ---------------------------------------------------------------------------
@dataclass
class Regime:
    direction: str = "neutral"  # up | down | neutral
    confidence: float = 0.0
    fetched_at: float = 0.0


class KronosFilter:
    def __init__(self, client=None):
        import requests  # lazy: not needed until first regime call

        self._requests = requests
        self.client = client
        self.url = KRONOS_URL
        self._cache = Regime("neutral", 0.0, 0.0)

    def _rows(self):
        now = datetime.now()
        start = now - timedelta(days=KRONOS_BACKFILL_DAYS)
        df = self.client.history(
            symbol=UNDERLYING,
            exchange=SPOT_EXCHANGE,
            interval=KRONOS_FREQ,
            start_date=start.strftime("%Y-%m-%d"),
            end_date=now.strftime("%Y-%m-%d"),
        )
        tail = df.tail(KRONOS_CONTEXT_ROWS) if hasattr(df, "tail") else df
        rows = []
        for _, r in tail.iterrows():
            rows.append(
                {
                    "open": float(r["open"]),
                    "high": float(r["high"]),
                    "low": float(r["low"]),
                    "close": float(r["close"]),
                    "volume": float(r.get("volume", 0) or 0),
                }
            )
        return rows

    def regime(self) -> Regime:
        now = time.time()
        if now - self._cache.fetched_at < KRONOS_CADENCE_SECONDS:
            return self._cache
        if self.client is None:
            raise RuntimeError("kronos requires an api client for history backfill")
        try:
            rows = self._rows()
            if len(rows) < KRONOS_CONTEXT_ROWS:
                print(
                    f"[KRONOS] {len(rows)} rows < {KRONOS_CONTEXT_ROWS} required -> neutral",
                    flush=True,
                )
                self._cache = Regime("neutral", 0.0, now)
                return self._cache
            resp = self._requests.post(
                self.url,
                json={
                    "data": rows,
                    "freq": KRONOS_FREQ,
                    "profile": KRONOS_PROFILE,
                },
                timeout=120,
            )
            body = resp.json()
            prediction = int(body.get("prediction", 0))
            if prediction == 1:
                direction = "up"
            elif prediction == -1:
                direction = "down"
            else:
                direction = "neutral"
            confidence = float(body.get("confidence", 0.0))
            self._cache = Regime(direction, confidence, now)
            print(
                f"[KRONOS] regime={direction} confidence={confidence:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[KRONOS] unavailable ({exc}) -> neutral", flush=True)
            self._cache = Regime("neutral", 0.0, now)
        return self._cache


# ---------------------------------------------------------------------------
# Decision: pure kronos-vote state machine
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    timestamp: float
    side: str            # "CE" | "PE" | ""
    action: str          # BUY | HOLD | EXIT | FLIP
    confidence: float
    reason: str
    vote: str = "neutral"


class DecisionEngine:
    def __init__(self, kronos: KronosFilter):
        self.kronos = kronos

    def evaluate(self, snap: Snapshot, owned_side) -> Decision:
        now = time.time()
        regime = self.kronos.regime()
        vote = regime.direction
        confidence = regime.confidence
        if vote == "up":
            target = "CE"
        elif vote == "down":
            target = "PE"
        else:
            target = None
        if owned_side is None:
            if target is None:
                return Decision(now, "", "HOLD", confidence, "vote-neutral", vote)
            return Decision(now, target, "BUY", confidence, "vote-entry", vote)
        if owned_side == target:
            return Decision(now, owned_side, "HOLD", confidence, "vote-hold", vote)
        if target is None:
            return Decision(now, owned_side, "EXIT", confidence, "vote-neutral-exit", vote)
        return Decision(now, target, "FLIP", confidence, "vote-flip", vote)

    @staticmethod
    def log(decision: Decision):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            record = {
                "timestamp": datetime.fromtimestamp(decision.timestamp).isoformat(),
                "symbol": f"{UNDERLYING}{EXPIRY_DATE}",
                "vote": decision.vote,
                "confidence": round(decision.confidence, 4),
                "side": decision.side or None,
                "action": decision.action,
                "decision": decision.reason,
                "reason": decision.reason,
            }
            with open(DECISION_LOG, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except Exception as exc:
            print(f"[LOG] decision log error ({exc})", flush=True)


# ---------------------------------------------------------------------------
# Circuit breaker: block entries after repeated failures
# ---------------------------------------------------------------------------
class CircuitBreaker:
    TRIP_REASONS = ("too-many-rejected-orders", "daily-loss-exceeded")

    def __init__(self):
        self.tripped = False
        self.trip_reason = ""
        self.trip_date = ""
        self.consecutive_rejections = 0
        self.live_pnl_pts = 0.0

    def check(self) -> bool:
        today = datetime.now().strftime("%Y-%m-%d")
        if self.trip_date != today:
            self.tripped = False
            self.trip_reason = ""
            self.trip_date = today
            self.consecutive_rejections = 0
        return self.tripped

    def trip(self, reason: str):
        self.tripped = True
        self.trip_reason = reason
        self.trip_date = datetime.now().strftime("%Y-%m-%d")
        print(f"[BREAKER] CIRCUIT OPEN: {reason}", flush=True)

    def on_rejected_order(self):
        self.consecutive_rejections += 1
        if self.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
            self.trip("too-many-rejected-orders")

    def on_order_ok(self):
        self.consecutive_rejections = 0


# ---------------------------------------------------------------------------
# Risk management (pre-entry gates)
# ---------------------------------------------------------------------------
class RiskManager:
    def __init__(self, client, breaker: CircuitBreaker):
        self.client = client
        self.breaker = breaker

    def allow(self, snap: Snapshot):
        if not snap.fresh():
            return "market-data-stale"
        if self.breaker.check():
            return "circuit-open"
        try:
            funds = self.client.funds().get("data", {})
            if float(funds.get("availablecash") or 0) <= 0:
                return "margin-unavailable"
        except Exception as exc:
            print(f"[RISK] funds check failed ({exc})", flush=True)
            return "margin-unavailable"
        return None


# ---------------------------------------------------------------------------
# Execution: single-position lifecycle with SL/TARGET legs
# ---------------------------------------------------------------------------
class Execution:
    def __init__(self, client, breaker: CircuitBreaker,
                 owner: PositionOwner | None = None):
        self.client = client
        self.breaker = breaker
        self._owner = owner
        self._owned_position_id: str | None = None

    def _wait_fill(self, order_id: str, timeout: float = 90.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                status = self.client.orderstatus(order_id=order_id, strategy=STRATEGY_NAME)
                data = status.get("data", {})
                if data.get("order_status") == "complete":
                    return float(
                        data.get("average_price")
                        or data.get("averageprice")
                        or data.get("price")
                        or 0.0
                    )
            except Exception as exc:
                print(f"[EXEC] orderstatus error ({exc})", flush=True)
            time.sleep(0.5)
        return None

    def _position_fill(self, symbol: str):
        """If the entry filled after _wait_fill gave up, recover the average
        fill price from the position book so SL/TARGET legs still get placed."""
        try:
            book = self.client.positionbook()
            for pos in book.get("data", []):
                if pos.get("symbol") != symbol:
                    continue
                qty = (
                    pos.get("quantity")
                    or pos.get("netqty")
                    or pos.get("netquantity")
                    or pos.get("daynetqty")
                    or 0
                )
                avg = (
                    pos.get("average_price")
                    or pos.get("averageprice")
                    or pos.get("avgprice")
                    or pos.get("price")
                    or 0
                )
                if float(qty or 0) != 0 and float(avg or 0) > 0:
                    return float(avg)
        except Exception as exc:
            print(f"[EXEC] positionbook recovery error ({exc})", flush=True)
        return None

    @staticmethod
    def _entry_qty(snap: Snapshot) -> int:
        lots = snap.lotsize or FALLBACK_LOT
        return max(lots, int(QUANTITY / lots) * lots)

    # ---------------------------- state persistence ----------------------
    def _save_state(self, positions, extra=None):
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            prev = {}
            try:
                with open(STATE_FILE, "r") as fh:
                    prev = json.load(fh)
            except Exception:
                pass
            payload = {"positions": positions, "updated": time.time()}
            iid = prev.get("instance_id") or (extra or {}).get("instance_id")
            if iid:
                payload["instance_id"] = iid
            with open(STATE_FILE, "w") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            print(f"[EXEC] state save error ({exc})", flush=True)

    def _load_state(self):
        try:
            with open(STATE_FILE, "r") as fh:
                data = json.load(fh)
            pos = data.get("positions") or {}
            return {s: p for s, p in pos.items() if int(p.get("qty") or 0) != 0}
        except Exception:
            return {}

    def _resolve_instance_id(self) -> str:
        try:
            with open(STATE_FILE, "r") as fh:
                raw = json.load(fh)
            iid = raw.get("instance_id")
        except Exception:
            iid = None
        if not iid:
            iid = str(uuid.uuid4())
            self._save_state(self._load_state(), extra={"instance_id": iid})
            print(f"[EXEC] generated instance_id {iid}", flush=True)
        return iid

    def _tradebook_net_positions(self):
        """Aggregate tradebook rows tagged with our strategy into net
        quantities per symbol. Returns {symbol: net_qty} for nonzero nets;
        {} on any error (safe default)."""
        nets = {}
        try:
            tb = self.client.tradebook()
            rows = tb.get("data", []) if isinstance(tb, dict) else tb
            for r in rows:
                if (r.get("strategy") or "") != STRATEGY_NAME:
                    continue
                sym = str(r.get("symbol") or "")
                if not sym:
                    continue
                qty = int(r.get("quantity") or 0)
                act = str(r.get("action") or "").upper()
                if act == "BUY":
                    nets[sym] = nets.get(sym, 0) + qty
                elif act == "SELL":
                    nets[sym] = nets.get(sym, 0) - qty
        except Exception as exc:
            print(
                f"[EXEC] tradebook ownership error ({exc}) -> treating as no positions",
                flush=True,
            )
            return {}
        return {s: n for s, n in nets.items() if n != 0}

    def _owned_positions(self):
        """Resolve positions owned by THIS strategy: state.json is
        authoritative; tradebook (strategy-tagged rows) fills the gap when
        state is missing (e.g. first run after an upgrade)."""
        state = self._load_state()
        tb = self._tradebook_net_positions()
        owned = dict(state)
        for sym, net in tb.items():
            if sym not in owned:
                print(f"[EXEC] tradebook shows owned position {sym} not in state - adopting", flush=True)
                owned[sym] = {"qty": net, "entry_price": 0.0}
        self._save_state(owned)
        return owned

    def owned_side(self):
        """The side (CE/PE) of the single owned position, or None."""
        for sym in self._owned_positions():
            upper = str(sym).upper()
            if upper.endswith("CE"):
                return "CE"
            if upper.endswith("PE"):
                return "PE"
        return None

    def _note_entry(self, symbol, side, qty, entry_price):
        state = self._load_state()
        state[symbol] = {
            "side": side,
            "qty": qty,
            "entry_price": entry_price,
            "ts": time.time(),
        }
        self._save_state(state)

    def _clear_position(self, symbol):
        if self._owner is not None and self._owned_position_id is not None:
            try:
                self._owner.release(self._owned_position_id, reason="position_closed")
                print(f"[EXEC] PositionOwner released {symbol}", flush=True)
            except Exception as exc:
                print(f"[EXEC] PositionOwner release failed ({exc})", flush=True)
            self._owned_position_id = None
        state = self._load_state()
        state.pop(symbol, None)
        self._save_state(state)

    # ---------------------------- order actions --------------------------
    def buy_leg(self, side: str, snap: Snapshot):
        symbol = snap.ce_symbol if side == "CE" else snap.pe_symbol
        q = self.client.quotes(symbol=symbol, exchange=OPTIONS_EXCHANGE)["data"]
        ltp = q["ltp"]
        limit_price = round(ltp * 1.001, 2)
        qty = self._entry_qty(snap)
        print(f"[EXEC] {side} entry qty={qty} lotsize={snap.lotsize}", flush=True)
        entry = self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=OPTIONS_EXCHANGE,
            action="BUY",
            price_type=PRICE_TYPE,
            product=PRODUCT,
            quantity=str(qty),
            price=str(limit_price),
        )
        order_id = entry.get("orderid")
        if not order_id or entry.get("status") != "success":
            print(f"[EXEC] entry rejected ({entry.get('message', 'unknown')})", flush=True)
            self.breaker.on_rejected_order()
            return
        self.breaker.on_order_ok()
        fill = self._wait_fill(order_id)
        if fill is None:
            fill = self._position_fill(symbol)
        if fill is None:
            print(f"[EXEC] fill timeout for {symbol} - leaving broker legs", flush=True)
            return
        trigger = round(fill - STOP_POINTS, 2)
        self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=OPTIONS_EXCHANGE,
            action="SELL",
            price_type="SL",
            product=PRODUCT,
            quantity=str(qty),
            trigger_price=str(trigger),
            price=str(round(trigger * 0.995, 2)),
        )
        self.client.placeorder(
            strategy=STRATEGY_NAME,
            symbol=symbol,
            exchange=OPTIONS_EXCHANGE,
            action="SELL",
            price_type="LIMIT",
            product=PRODUCT,
            quantity=str(qty),
            price=str(round(fill + TARGET_POINTS, 2)),
        )
        print(
            f"[EXEC] {side} {symbol} filled @ {fill} | SL {trigger} | TARGET {fill + TARGET_POINTS:.2f}",
            flush=True,
        )
        self._note_entry(symbol, side, qty, fill)
        # Register with shared PositionOwner (best-effort)
        if self._owner is not None:
            try:
                self._owned_position_id = self._owner.register(
                    strategy_id=STRATEGY_ID,
                    strategy_mode=ExecutionMode.INTRADAY,
                    instrument=symbol,
                    symbol=symbol,
                    quantity=qty,
                    entry_order_id=order_id,
                )
                print(f"[EXEC] PositionOwner registered {symbol} -> {self._owned_position_id}", flush=True)
            except Exception as exc:
                print(f"[EXEC] PositionOwner register failed ({exc})", flush=True)

    def exit_position(self):
        """Flatten the owned position: cancel legs, close, clear state."""
        owned = self._owned_positions()
        if not owned:
            print("[EXEC] no owned position to exit", flush=True)
            return
        try:
            self.client.cancelallorder(strategy=STRATEGY_NAME)
        except Exception as exc:
            print(f"[EXEC] cancelallorder error ({exc})", flush=True)
        for symbol in owned:
            try:
                self.client.closeposition(
                    strategy=STRATEGY_NAME,
                    symbol=symbol,
                    exchange=OPTIONS_EXCHANGE,
                )
                print(f"[EXEC] closed owned position {symbol}", flush=True)
            except Exception as exc:
                print(f"[EXEC] closeposition {symbol} error ({exc})", flush=True)
            self._clear_position(symbol)

    def square_off(self):
        print("[EXEC] square-off window", flush=True)
        owned = self._owned_positions()
        if not owned:
            print("[EXEC] no owned positions to square off", flush=True)
            return
        try:
            self.client.cancelallorder(strategy=STRATEGY_NAME)
        except Exception as exc:
            print(f"[EXEC] cancelallorder error ({exc})", flush=True)
        for symbol in owned:
            try:
                self.client.closeposition(
                    strategy=STRATEGY_NAME,
                    symbol=symbol,
                    exchange=OPTIONS_EXCHANGE,
                )
                print(f"[EXEC] closed owned position {symbol}", flush=True)
            except Exception as exc:
                print(f"[EXEC] closeposition {symbol} error ({exc})", flush=True)
            self._clear_position(symbol)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _hhmm_to_epoch(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return (int(h) * 3600 + int(m) * 60) % 86400


def _shutdown_cleanly(client):
    """Close the WS feed gracefully, then drop the instance lock."""
    try:
        if getattr(client, "disconnect", None):
            client.disconnect()
    except Exception:
        pass
    try:
        if _lock_fd is not None:
            _lock_fd.close()
    except Exception:
        pass


# Per-side in-flight entry guard: never spawn a second buy_leg thread for a
# side that already has an entry being placed/fill-polled.
_ENTRY_INFLIGHT = {"CE": False, "PE": False}
_ENTRY_LOCK = threading.Lock()


def _buy_leg_wrapper(execr, side: str, snap):
    try:
        execr.buy_leg(side, snap)
    finally:
        with _ENTRY_LOCK:
            _ENTRY_INFLIGHT[side] = False


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(
        f"START mode={MODE or 'live'} expiry={EXPIRY_DATE} strategy={STRATEGY_NAME}",
        flush=True,
    )
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)
    try:
        status = client.analyzerstatus()
        if status.get("data", {}).get("analyze_mode"):
            print(f"ANALYZER simulated ({status['data'].get('total_logs')} logs)", flush=True)
        else:
            print("LIVE mode", flush=True)
    except Exception:
        print("analyzer status unavailable", flush=True)

    try:
        ex = client.expiry(symbol=UNDERLYING, exchange=OPTIONS_EXCHANGE, instrumenttype="options")
    except TypeError:
        ex = client.expiry(symbol=UNDERLYING, exchange=OPTIONS_EXCHANGE)
    except Exception as exc:
        print(f"expiry validation skipped ({exc})", flush=True)
        ex = None
    if ex is not None:
        if isinstance(ex, list):
            expiries = ex
        elif isinstance(ex, dict) and ex.get("status") == "success":
            expiries = ex.get("data", [])
        else:
            print("expiry validation skipped (unexpected response)", flush=True)
            expiries = None
        if isinstance(expiries, list):
            normalized = [str(e).replace("-", "").upper() for e in expiries]
            if EXPIRY_DATE.replace("-", "").upper() not in normalized:
                raise SystemExit(f"expiry {EXPIRY_DATE} not found for {UNDERLYING}")

    # --- Shared position-ownership components (best-effort, alongside state.json) ---
    _stop_mode = StopMode(STOP_MODE) if STOP_MODE in [m.value for m in StopMode] else StopMode.STOP_AND_CLOSE
    _persister: StatePersister | None = None
    _owner: PositionOwner | None = None
    _reconciler: Reconciler | None = None
    _recovery: RecoveryManager | None = None
    try:
        register_strategy_mode(STRATEGY_ID, ExecutionMode.INTRADAY)
        _persister = StatePersister()
        _owner = PositionOwner(persister=_persister)
        _reconciler = Reconciler(persister=_persister, owner=_owner)
        _recovery = RecoveryManager(STRATEGY_ID, persister=_persister, owner=_owner, reconciler=_reconciler)
        print(f"[EXEC] shared position-ownership initialized (stop_mode={_stop_mode.value})", flush=True)
    except Exception as exc:
        print(f"[EXEC] shared init failed - using state.json only ({exc})", flush=True)
        _persister = _owner = _reconciler = _recovery = None

    snapshot = Snapshot()
    data = DataAcquisition(client, snapshot)
    data.start()
    kronos = KronosFilter(client)
    engine = DecisionEngine(kronos)
    breaker = CircuitBreaker()
    risk = RiskManager(client, breaker)
    execr = Execution(client, breaker, owner=_owner)
    instance_id = execr._resolve_instance_id()
    print(f"START instance_id={instance_id}", flush=True)

    stop = threading.Event()
    eod_squared = False
    _sq_lock = threading.Lock()

    def _run_square_off_once(tag):
        # Called from main loop (eod/stop) AND a signal-handler daemon thread;
        # lock+flag guarantee square_off runs at most once, never concurrently.
        nonlocal eod_squared
        with _sq_lock:
            if eod_squared:
                return
            eod_squared = True
        try:
            execr.square_off()
        except Exception as exc:
            print(f"[EXEC] {tag} square-off error ({exc})", flush=True)

    def _on_stop_signal(signum, frame):
        stop.set()
        print(f"received signal {signum} (stop_mode={_stop_mode.value})", flush=True)
        if _stop_mode == StopMode.STOP_TRADING_ONLY:
            print("[EXEC] STOP_TRADING_ONLY - skipping square-off", flush=True)
            return
        print("[EXEC] closing open positions on signal", flush=True)
        threading.Thread(
            target=_run_square_off_once,
            args=("signal",),
            daemon=True,
        ).start()

    signal.signal(signal.SIGTERM, _on_stop_signal)
    signal.signal(signal.SIGINT, _on_stop_signal)

    # Startup reconciliation: a previous run may have been killed before its
    # square-off finished, leaving an open position. Flatten leftovers.
    print(f"[EXEC] startup reconciliation (instance_id={instance_id})", flush=True)
    try:
        owned = execr._owned_positions()
        if owned:
            print(
                f"[EXEC] startup reconciliation - {len(owned)} owned open position(s) "
                "from previous run, squaring off",
                flush=True,
            )
            execr.square_off()
        else:
            print("[EXEC] startup reconciliation - no owned positions", flush=True)
    except Exception as exc:
        print(f"[EXEC] startup reconciliation skipped ({exc})", flush=True)

    if _recovery is not None:
        try:
            def _fetch_broker_positions():
                return client.positionbook().get("data", [])
            rec = _recovery.startup(_fetch_broker_positions)
            print(f"[EXEC] shared recovery state={rec.state.value} stale={rec.stale_count}", flush=True)
        except RuntimeError as exc:
            print(f"[EXEC] shared recovery REQUIRES MANUAL INTERVENTION: {exc}", flush=True)
        except Exception as exc:
            print(f"[EXEC] shared recovery skipped ({exc})", flush=True)

    last_decided = 0.0
    last_resolve = 0.0
    while not stop.is_set():
        now = time.time()
        if not (snapshot.ce_symbol and snapshot.pe_symbol):
            if now - last_resolve > 5:
                last_resolve = now
                data.resolve_atm()
                if snapshot.ce_symbol and snapshot.pe_symbol:
                    data.client.subscribe_ltp(data.instruments, data._on_ltp)
            stop.wait(1)
            continue
        if now % 86400 > _hhmm_to_epoch(SQUARE_OFF_HHMM):
            _run_square_off_once("eod")
            stop.set()
            break
        if (
            snapshot.fresh()
            and now - last_decided > DECISION_CADENCE_SECONDS
            and snapshot.ce_symbol
            and snapshot.pe_symbol
        ):
            try:
                owned_side = execr.owned_side()
                decision = engine.evaluate(snapshot, owned_side)
            except Exception as exc:
                print(f"VOTE error ({exc})", flush=True)
                last_decided = now
                stop.wait(5)
                continue
            DecisionEngine.log(decision)
            last_decided = now
            if decision.action == "BUY":
                block = risk.allow(snapshot)
                if block:
                    print(f"BLOCKED {decision.side} ({block})", flush=True)
                elif execr.owned_side():
                    print("[EXEC] position already open - skipping re-entry", flush=True)
                else:
                    with _ENTRY_LOCK:
                        if _ENTRY_INFLIGHT.get(decision.side):
                            print(
                                f"[EXEC] {decision.side} entry already in flight - skipping",
                                flush=True,
                            )
                        else:
                            _ENTRY_INFLIGHT[decision.side] = True
                            threading.Thread(
                                target=_buy_leg_wrapper,
                                args=(execr, decision.side, snapshot),
                                daemon=True,
                            ).start()
            elif decision.action == "EXIT":
                execr.exit_position()
            elif decision.action == "FLIP":
                execr.exit_position()
                block = risk.allow(snapshot)
                if block:
                    print(f"BLOCKED {decision.side} after flip ({block})", flush=True)
                else:
                    with _ENTRY_LOCK:
                        if _ENTRY_INFLIGHT.get(decision.side):
                            print(
                                f"[EXEC] {decision.side} entry already in flight - skipping",
                                flush=True,
                            )
                        else:
                            _ENTRY_INFLIGHT[decision.side] = True
                            threading.Thread(
                                target=_buy_leg_wrapper,
                                args=(execr, decision.side, snapshot),
                                daemon=True,
                            ).start()
        stop.wait(5)
    if not eod_squared:
        print("[EXEC] stop requested - squaring off open positions", flush=True)
        _run_square_off_once("stop")
    _shutdown_cleanly(client)
    print("STOP clean", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Kronos ATM directional strategy (NIFTY options)")
    parser.add_argument(
        "--mode",
        default=None,
        help="Execution mode: live | paper | backtest (default: $MODE env or live)",
    )
    args = parser.parse_args()
    if args.mode:
        globals()["MODE"] = args.mode
    main()
