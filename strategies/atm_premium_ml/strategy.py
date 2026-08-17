"""ATM Premium ML strategy - live execution host.

7-component architecture:

1. DataAcquisition      - OpenAlgo WS LTP feed for spot + ATM options; dynamic
                          ATM re-resolution on spot roll; per-minute OHLCV
                          window buffers per option side (backfilled from the
                          history API at startup / after rolls).
2. FeatureEngineering   - tick -> per-side 60x7 window of
                          [open, high, low, close, iv, volume, oi] matching the
                          trained OptionTransformer's channel order.
3. ModelInference      - interface; LocalInference loads the CE/PE
                          OptionTransformer checkpoints
                          (models/{ce,pe}/option_transformer_{ce,pe}_complete.pt,
                          which embed model_state_dict + feature_mean/std +
                          sequence_length=60 + threshold=0.8), z-scores the
                          window and softmaxes a binary probability;
                          RemoteInference POSTs the windows to a VM2 endpoint.
                          Selected via INFERENCE_MODE.
4. KronosFilter         - low-cadence regime veto via Kronos-small
                          (POST {KRONOS_URL} with >=512 rows of spot OHLCV ->
                          prediction 1/-1/0 mapped to up/down/neutral);
                          neutral on failure (never blocks the loop).
5. DecisionEngine       - confidence gates (CE_THRESHOLD / PE_THRESHOLD),
                          signal TTL, conflicting-signal rule, kronos veto,
                          JSONL decision log.
6. RiskManagement       - circuit breaker (daily loss, order rejections, stale
                          market, inference latency, model errors), margin
                          check, ATM-still-current.
7. Execution            - entry -> fill -> SL + target legs -> square-off.
                          Analyzer-aware (orders simulated in analyzer mode).

Run:
    MODE=live python strategies/atm_premium_ml/strategy.py

Host contract:
    - HOST_SERVER is honoured first, then OPENALGO_HOST, else the local default.
    - OPENALGO_STRATEGY_EXCHANGE is honoured for the strategy exchange.
    - SIGTERM / SIGINT handlers stop the loop cleanly.
    - stdout-only logging (no files, no libraries) so the host captures logs.
    - no asyncio: the OpenAlgo SDK is synchronous.
"""

import json
import os
import signal
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from dotenv import find_dotenv, load_dotenv

from openalgo import api

load_dotenv(find_dotenv(), override=False)

from strategies.shared import (  # noqa: E402
    ExecutionMode,
    PositionOwner,
    Reconciler,
    RecoveryManager,
    StatePersister,
    StopMode,
)
from strategies.shared.execution_modes import register_strategy_mode  # noqa: E402

try:
    import torch
except ImportError:  # pragma: no cover - torch-less host (fail fast in LocalInference)
    torch = None

# ---------------------------------------------------------------------------
# Configuration (env-driven; every value has a sane default)
# ---------------------------------------------------------------------------
HOST = os.environ.get("HOST_SERVER") or os.environ.get("OPENALGO_HOST") or "http://127.0.0.1:5000"
WS_URL = os.environ.get("OPENALGO_WS_URL") or "ws://127.0.0.1:8765"
API_KEY = os.environ.get("OPENALGO_API_KEY") or ""

STRATEGY_ID = "atm_premium_ml"
STRATEGY_NAME = os.environ.get("STRATEGY_NAME") or STRATEGY_ID
UNDERLYING = os.environ.get("UNDERLYING") or "NIFTY"
SPOT_EXCHANGE = os.environ.get("SPOT_EXCHANGE") or "NSE_INDEX"
OPTIONS_EXCHANGE = os.environ.get("OPTIONS_EXCHANGE") or "NFO"
EXCHANGE = os.environ.get("OPENALGO_STRATEGY_EXCHANGE") or OPTIONS_EXCHANGE
EXPIRY_DATE = os.environ.get("EXPIRY_DATE") or "11AUG26"

PRODUCT = os.environ.get("PRODUCT") or "MIS"
PRICE_TYPE = os.environ.get("PRICE_TYPE") or "LIMIT"
QUANTITY = int(os.environ.get("QUANTITY") or 75)
TARGET_POINTS = float(os.environ.get("TARGET_POINTS") or 20)
STOP_POINTS = float(os.environ.get("STOP_POINTS") or 10)

# Model confidence thresholds (match the checkpoint's calibrated threshold 0.8).
CE_THRESHOLD = float(os.environ.get("CE_THRESHOLD") or 0.80)
PE_THRESHOLD = float(os.environ.get("PE_THRESHOLD") or 0.80)

SIGNAL_TTL_SECONDS = float(os.environ.get("SIGNAL_TTL_SECONDS") or 15)
STALE_MARKET_SECONDS = float(os.environ.get("STALE_MARKET_SECONDS") or 5)

DAILY_LOSS_LIMIT_PTS = float(os.environ.get("DAILY_LOSS_LIMIT_PTS") or 100)
MAX_CONSECUTIVE_REJECTIONS = int(os.environ.get("MAX_CONSECUTIVE_REJECTIONS") or 3)
MAX_INFERENCE_LATENCY_MS = float(os.environ.get("MAX_INFERENCE_LATENCY_MS") or 2000)
MAX_MODEL_ERRORS = int(os.environ.get("MAX_MODEL_ERRORS") or 5)

ROLL_THRESHOLD_PTS = float(os.environ.get("ROLL_THRESHOLD_PTS") or 25)

INFERENCE_MODE = os.environ.get("INFERENCE_MODE") or "local"
CE_MODEL_PATH = (
    os.environ.get("CE_MODEL_PATH")
    or "strategies/atm_premium_ml/models/ce/option_transformer_complete.pt"
)
PE_MODEL_PATH = (
    os.environ.get("PE_MODEL_PATH")
    or "strategies/atm_premium_ml/models/pe/option_transformer_pe_complete.pt"
)
VM2_INFERENCE_URL = os.environ.get("VM2_INFERENCE_URL") or "http://127.0.0.1:8000/infer"

# Feature-window configuration for the OptionTransformer (sequence_length=60).
FEATURE_NAMES = ["open", "high", "low", "close", "iv", "volume", "oi"]
FEATURE_VERSION = "1.0.0"
WINDOW_BARS = int(os.environ.get("WINDOW_BARS") or 60)
MIN_WINDOW_ROWS = int(os.environ.get("MIN_WINDOW_ROWS") or WINDOW_BARS)
BACKFILL_DAYS = int(os.environ.get("BACKFILL_DAYS") or 7)
IV_CADENCE_SECONDS = float(os.environ.get("IV_CADENCE_SECONDS") or 60)

# Kronos regime filter (Kronos-small inference server).
KRONOS_URL = os.environ.get("KRONOS_URL") or "http://127.0.0.1:8000/predict"
KRONOS_CADENCE_SECONDS = float(os.environ.get("KRONOS_CADENCE_SECONDS") or 900)
KRONOS_FREQ = os.environ.get("KRONOS_FREQ") or "5m"
KRONOS_CONTEXT_ROWS = int(os.environ.get("KRONOS_CONTEXT_ROWS") or 512)
KRONOS_BACKFILL_DAYS = int(os.environ.get("KRONOS_BACKFILL_DAYS") or 14)
KRONOS_PROFILE = os.environ.get("KRONOS_PROFILE") or "fast"

SQUARE_OFF_HHMM = os.environ.get("SQUARE_OFF_HHMM") or "15:15"
DATA_DIR = os.environ.get("DATA_DIR") or "strategies/atm_premium_ml"
DECISION_LOG = os.environ.get("DECISION_LOG") or os.path.join(DATA_DIR, "decisions.jsonl")
STATE_FILE = os.environ.get("STATE_FILE") or os.path.join(DATA_DIR, "state.json")
MODE = os.environ.get("MODE") or ""


@dataclass
class Snapshot:
    spot: float = 0.0
    ce_symbol: str = ""
    pe_symbol: str = ""
    ce_ltp: float = 0.0
    pe_ltp: float = 0.0
    atm_strike: int = 0
    lotsize: int = 0
    iv_ce: float = 0.0
    iv_pe: float = 0.0
    last_update: float = 0.0

    def fresh(self) -> bool:
        return time.time() - self.last_update <= STALE_MARKET_SECONDS


# ---------------------------------------------------------------------------
# 1. Data acquisition + per-minute window buffers
# ---------------------------------------------------------------------------
class WindowBuffer:
    """Thread-safe per-symbol minute-bar buffer.

    A bar is a tuple (epoch_minute, open, high, low, close, iv, volume, oi).
    Live ticks roll open bars on minute boundaries; backfill inserts closed
    historical bars. windows() returns rows in FEATURE_NAMES order:
    [open, high, low, close, iv, volume, oi].
    """

    def __init__(self, max_bars: int = WINDOW_BARS * 3):
        self._max_bars = max_bars
        self._bars = {}
        self._lock = threading.Lock()

    def reset(self, symbols):
        with self._lock:
            self._bars = {s: [] for s in symbols}

    def update(self, symbol, ts, ltp, iv=None, volume=None, oi=None):
        with self._lock:
            bars = self._bars.get(symbol)
            if bars is None:
                return
            bucket = int(ts // 60)
            if bars and bars[-1][0] == bucket:
                b = bars[-1]
                bars[-1] = (
                    bucket,
                    b[1],
                    max(b[2], ltp),
                    min(b[3], ltp),
                    ltp,
                    iv if iv is not None else b[5],
                    volume if volume is not None else b[6],
                    oi if oi is not None else b[7],
                )
            else:
                bars.append(
                    (
                        bucket,
                        ltp,
                        ltp,
                        ltp,
                        ltp,
                        iv if iv is not None else 0.0,
                        volume if volume is not None else 0.0,
                        oi if oi is not None else 0.0,
                    )
                )
                if len(bars) > self._max_bars:
                    del bars[: len(bars) - self._max_bars]

    def backfill(self, symbol, rows, iv):
        """rows: list of (epoch_minute, open, high, low, close, volume)."""
        with self._lock:
            bars = self._bars.setdefault(symbol, [])
            bars.extend((m, o, h, l, c, iv, v, 0.0) for m, o, h, l, c, v in rows)
            bars.sort(key=lambda r: r[0])
            if len(bars) > self._max_bars:
                del bars[: len(bars) - self._max_bars]

    def window(self, symbol, n):
        with self._lock:
            bars = self._bars.get(symbol) or []
            rows = bars[-n:]
            return [[b[1], b[2], b[3], b[4], b[5], b[6], b[7]] for b in rows]

    def count(self, symbol) -> int:
        with self._lock:
            return len(self._bars.get(symbol) or [])


class DataAcquisition:
    def __init__(self, client, snapshot: Snapshot, buffers: WindowBuffer):
        self.client = client
        self.snapshot = snapshot
        self.buffers = buffers
        self.instruments = [{"exchange": SPOT_EXCHANGE, "symbol": UNDERLYING}]
        self.lock = threading.Lock()

    @staticmethod
    def _extract_symbol(resp, what):
        """Extract the option symbol from SDK response (flat or nested shape)."""
        if isinstance(resp, dict):
            if resp.get("status") == "success" and resp.get("symbol"):
                return resp["symbol"]
            data = resp.get("data")
            if isinstance(data, dict) and data.get("symbol"):
                return data["symbol"]
        raise ValueError(f"{what}: optionsymbol unexpected response: {str(resp)[:160]}")

    def resolve_atm(self):
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
            ce_symbol = self._extract_symbol(ce, "CE")
            pe_symbol = self._extract_symbol(pe, "PE")
        except Exception as exc:
            print(f"[DATA] ATM resolve failed ({exc}); retrying next cycle", flush=True)
            return
        with self.lock:
            self.snapshot.ce_symbol = ce_symbol
            self.snapshot.pe_symbol = pe_symbol
            self.snapshot.atm_strike = int(ce_symbol[-7:-2])
            lots = ce.get("lotsize") or pe.get("lotsize")
            if lots:
                self.snapshot.lotsize = int(lots)
            spot = ce.get("underlying_ltp") or pe.get("underlying_ltp")
            if spot:
                self.snapshot.spot = float(spot)
            self.instruments = [
                {"exchange": SPOT_EXCHANGE, "symbol": UNDERLYING},
                {"exchange": OPTIONS_EXCHANGE, "symbol": self.snapshot.ce_symbol},
                {"exchange": OPTIONS_EXCHANGE, "symbol": self.snapshot.pe_symbol},
            ]
        # New symbols => start live accumulation immediately; the main loop
        # backfills history for them on its next pass.
        self.buffers.reset([self.snapshot.ce_symbol, self.snapshot.pe_symbol])

    def _on_ltp(self, msg):
        symbol = msg.get("symbol", "")
        data = msg.get("data") or {}
        ltp = data.get("ltp")
        if ltp is None:
            return
        ts = float(data.get("timestamp") or time.time())
        if ts > 1e12:
            ts /= 1000.0
        with self.lock:
            snap = self.snapshot
            if symbol == UNDERLYING:
                snap.spot = ltp
            elif symbol == snap.ce_symbol:
                snap.ce_ltp = ltp
            elif symbol == snap.pe_symbol:
                snap.pe_ltp = ltp
            snap.last_update = time.time()
            roll = snap.spot and snap.atm_strike and abs(snap.spot - snap.atm_strike) > ROLL_THRESHOLD_PTS
        if symbol == snap.ce_symbol:
            self.buffers.update(symbol, ts, ltp, iv=snap.iv_ce)
        elif symbol == snap.pe_symbol:
            self.buffers.update(symbol, ts, ltp, iv=snap.iv_pe)
        if roll:
            print(f"[DATA] spot {snap.spot:.2f} rolled past ATM {snap.atm_strike} -> re-resolving", flush=True)
            self.resolve_atm()
            self.client.subscribe_ltp(self.instruments, self._on_ltp)

    def start(self):
        self.resolve_atm()
        self.client.connect()
        self.client.subscribe_ltp(self.instruments, self._on_ltp)


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
def make_features(snap: Snapshot, buffers: WindowBuffer) -> dict:
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_version": FEATURE_VERSION,
        "ce": {"symbol": snap.ce_symbol, "rows": buffers.window(snap.ce_symbol, WINDOW_BARS)},
        "pe": {"symbol": snap.pe_symbol, "rows": buffers.window(snap.pe_symbol, WINDOW_BARS)},
    }


def _features_summary(features: dict) -> dict:
    out = {"feature_version": features.get("feature_version", FEATURE_VERSION)}
    for side in ("ce", "pe"):
        seg = features.get(side) or {}
        rows = seg.get("rows") or []
        out[side] = {
            "symbol": seg.get("symbol", ""),
            "rows": len(rows),
            "last_close": rows[-1][3] if rows else None,
            "last_iv": rows[-1][4] if rows else None,
        }
    return out


# ---------------------------------------------------------------------------
# 3. Model inference (local OptionTransformer checkpoints / remote VM2)
# ---------------------------------------------------------------------------
@dataclass
class InferenceResult:
    prob_ce: float
    prob_pe: float
    model_version: str
    feature_version: str
    training_dataset: str
    prediction_time: float
    latency_ms: float
    source: str

    def confidence_for(self, side: str) -> float:
        return self.prob_ce if side == "CE" else self.prob_pe


class ModelInference(ABC):
    @abstractmethod
    def infer(self, features: dict) -> InferenceResult:
        ...

    @abstractmethod
    def healthy(self) -> bool:
        ...


def _load_bundle(path: str):
    """Load an OptionTransformer *complete.pt checkpoint.

    Returns a ModelBundle with the verified reconstruction of the trained
    architecture (strict load_state_dict) plus the embedded z-score stats and
    the calibrated decision threshold.
    """
    if torch is None:
        raise RuntimeError(
            "torch is required for local inference. Install it into the "
            "strategy venv (e.g. uv add torch, or pip install torch "
            "--index-url https://download.pytorch.org/whl/cpu)."
        )

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        sd = payload["model_state_dict"]
        if "feature_mean" not in payload or "feature_std" not in payload:
            raise RuntimeError(
                f"{path}: checkpoint has no feature_mean/feature_std - use a *_complete.pt artifact"
            )
        mean = torch.as_tensor(payload["feature_mean"], dtype=torch.float32)
        std = torch.as_tensor(payload["feature_std"], dtype=torch.float32)
        seq = int(payload.get("sequence_length", WINDOW_BARS))
        names = list(payload.get("feature_names", FEATURE_NAMES))
        threshold = float(payload.get("threshold", CE_THRESHOLD))
        cfg = dict(payload.get("model_config") or {})
    elif isinstance(payload, dict) and "input_proj.weight" in payload:
        raise RuntimeError(
            f"{path} is a bare state_dict without normalization stats; use the *_complete.pt artifact"
        )
    else:
        raise RuntimeError(
            f"{path}: unrecognized checkpoint format (type={type(payload).__name__})"
        )
    if names != FEATURE_NAMES:
        raise RuntimeError(f"{path}: feature_names {names} != expected {FEATURE_NAMES}")
    cfg.setdefault("input_dim", len(names))
    cfg.setdefault("d_model", 64)
    cfg.setdefault("nhead", 4)
    cfg.setdefault("num_layers", 2)
    cfg.setdefault("dropout", 0.1)
    cfg.setdefault("max_seq", seq)

    model = OptionTransformer(**cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"{path}: state dict mismatch missing={missing} unexpected={unexpected}"
        )
    model.eval()
    print(
        f"[MODEL] loaded {os.path.basename(path)} (seq={seq} cfg={cfg} threshold={threshold})",
        flush=True,
    )
    return {
        "model": model,
        "feature_mean": mean,
        "feature_std": std,
        "sequence_length": seq,
        "feature_names": names,
        "threshold": threshold,
        "model_config": cfg,
    }


class OptionTransformer(torch.nn.Module):
    """Verified reconstruction of the trained binary classifier.

    Linear(input_dim -> d_model) + learnable positional embedding
    -> TransformerEncoder(d_model, nhead, dim_feedforward=128, dropout)
    -> mean-pool over time -> LayerNorm -> [Linear(d,d), ReLU, Dropout, Linear(d,2)].
    Verified with strict load_state_dict (missing=[] unexpected=[]) against
    both the CE and PE checkpoints.
    """

    def __init__(self, input_dim=7, d_model=64, nhead=4, num_layers=2, dropout=0.1, max_seq=60):
        super().__init__()
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_seq, d_model))
        layer = torch.nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers)
        self.norm = torch.nn.LayerNorm(d_model)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, 2),
        )

    def forward(self, x):
        t = x.size(1)
        x = self.input_proj(x) + self.pos_embedding[:, :t, :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.norm(x)
        return self.head(x)


class LocalInference(ModelInference):
    def __init__(self):
        if torch is None:
            raise RuntimeError(
                "torch is required for local inference. Install it into the "
                "strategy venv (e.g. uv add torch, or pip install torch "
                "--index-url https://download.pytorch.org/whl/cpu)."
            )
        self.torch = torch
        self.ce = _load_bundle(CE_MODEL_PATH)
        self.pe = _load_bundle(PE_MODEL_PATH)
        self.model_version = os.environ.get("MODEL_VERSION") or "option-transformer-20260806"
        self.training_dataset = os.environ.get("TRAINING_DATASET") or "chainst-option-premium-60m"

    def _predict(self, bundle, rows: list) -> float:
        torch = self.torch
        if not rows:
            return 0.0
        x = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)  # (1,T,7)
        seq = bundle["sequence_length"]
        if x.size(1) < seq:
            pad = x[:, :1, :].expand(-1, seq - x.size(1), -1)
            x = torch.cat([pad, x], dim=1)
        elif x.size(1) > seq:
            x = x[:, -seq:, :]
        x = (x - bundle["feature_mean"]) / bundle["feature_std"]
        with torch.no_grad():
            logits = bundle["model"](x)
        return float(torch.softmax(logits, dim=-1)[0, 1].item())

    def infer(self, features: dict) -> InferenceResult:
        t0 = time.time()
        prob_ce = self._predict(self.ce, features["ce"]["rows"])
        prob_pe = self._predict(self.pe, features["pe"]["rows"])
        return InferenceResult(
            prob_ce=prob_ce,
            prob_pe=prob_pe,
            model_version=self.model_version,
            feature_version=features.get("feature_version", FEATURE_VERSION),
            training_dataset=self.training_dataset,
            prediction_time=time.time(),
            latency_ms=(time.time() - t0) * 1000.0,
            source="local",
        )

    def healthy(self) -> bool:
        return True


class RemoteInference(ModelInference):
    def __init__(self):
        import requests

        self._requests = requests
        self.url = VM2_INFERENCE_URL

    def infer(self, features: dict) -> InferenceResult:
        t0 = time.time()
        try:
            resp = self._requests.post(
                self.url,
                json={
                    "features": features,
                    "feature_version": features.get("feature_version", FEATURE_VERSION),
                    "requested_at": datetime.now().isoformat(),
                },
                timeout=max(1.0, MAX_INFERENCE_LATENCY_MS / 1000.0),
            )
            resp.raise_for_status()
            body = resp.json()
            return InferenceResult(
                prob_ce=float(body["prob_ce"]),
                prob_pe=float(body["prob_pe"]),
                model_version=body.get("model_version", "remote"),
                feature_version=body.get("feature_version", features.get("feature_version", FEATURE_VERSION)),
                training_dataset=body.get("training_dataset", "unknown"),
                prediction_time=time.time(),
                latency_ms=(time.time() - t0) * 1000.0,
                source="remote",
            )
        except Exception as exc:
            raise RuntimeError(f"remote inference failed: {exc}") from exc

    def healthy(self) -> bool:
        try:
            return self._requests.get(self.url + "/health", timeout=1).ok
        except Exception:
            return False


def make_inference() -> ModelInference:
    if INFERENCE_MODE == "remote":
        return RemoteInference()
    return LocalInference()


# ---------------------------------------------------------------------------
# 4. Kronos regime filter (Kronos-small)
# ---------------------------------------------------------------------------
@dataclass
class Regime:
    direction: str  # up / down / neutral
    confidence: float
    fetched_at: float


class KronosFilter:
    def __init__(self, client=None):
        import requests

        self._requests = requests
        self.url = KRONOS_URL
        self.client = client
        self._cache = Regime("neutral", 0.0, 0.0)

    def _rows(self) -> list:
        """Most-recent-last OHLCV rows for the spot underlying."""
        import pandas as pd

        end = datetime.now()
        start = end - timedelta(days=KRONOS_BACKFILL_DAYS)
        try:
            df = self.client.history(
                symbol=UNDERLYING,
                exchange=SPOT_EXCHANGE,
                interval=KRONOS_FREQ,
                start_date=start.strftime("%Y-%m-%d"),
                end_date=end.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            print(f"[KRONOS] history failed ({exc})", flush=True)
            return []
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        return [
            {
                "open": float(r["open"]),
                "high": float(r["high"]),
                "low": float(r["low"]),
                "close": float(r["close"]),
                "volume": float(r.get("volume", 0) or 0),
            }
            for _, r in df.tail(KRONOS_CONTEXT_ROWS).iterrows()
        ]

    def regime(self) -> Regime:
        if time.time() - self._cache.fetched_at < KRONOS_CADENCE_SECONDS:
            return self._cache
        try:
            if self.client is None:
                raise RuntimeError("kronos requires an api client for history backfill")
            rows = self._rows()
            if len(rows) < KRONOS_CONTEXT_ROWS:
                print(
                    f"[KRONOS] {len(rows)} rows < {KRONOS_CONTEXT_ROWS} required -> neutral",
                    flush=True,
                )
                self._cache = Regime("neutral", 0.0, time.time())
                return self._cache
            resp = self._requests.post(
                self.url,
                json={"data": rows, "freq": KRONOS_FREQ, "profile": KRONOS_PROFILE},
                timeout=120,
            )
            resp.raise_for_status()
            body = resp.json()
            pred = int(body.get("prediction", 0))
            direction = {1: "up", -1: "down"}.get(pred, "neutral")
            self._cache = Regime(direction, float(body.get("confidence", 0.0)), time.time())
            print(
                f"[KRONOS] regime={direction} confidence={self._cache.confidence:.3f}",
                flush=True,
            )
        except Exception as exc:
            print(f"[KRONOS] unavailable ({exc}) -> neutral", flush=True)
            self._cache = Regime("neutral", 0.0, time.time())
        return self._cache


# ---------------------------------------------------------------------------
# 5. Decision engine
# ---------------------------------------------------------------------------
@dataclass
class Decision:
    timestamp: float
    side: str  # CE / PE / NONE
    action: str  # BUY / HOLD
    confidence: float
    reason: str
    result: InferenceResult | None = None
    kronos_direction: str = "neutral"


class DecisionEngine:
    def __init__(self, kronos: KronosFilter):
        self.kronos = kronos

    @staticmethod
    def log(decision: Decision, features: dict):
        os.makedirs(DATA_DIR, exist_ok=True)
        rec = {
            "timestamp": datetime.fromtimestamp(decision.timestamp).isoformat(),
            "symbol": f"{UNDERLYING}{EXPIRY_DATE}",
            "features": _features_summary(features),
            "side": decision.side,
            "action": decision.action,
            "confidence": decision.confidence,
            "kronos_state": {"direction": decision.kronos_direction},
            "decision": decision.reason,
            "reason": decision.reason,
        }
        with open(DECISION_LOG, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def evaluate(self, snap: Snapshot, inf: InferenceResult) -> Decision:
        now = time.time()
        if now - inf.prediction_time > SIGNAL_TTL_SECONDS:
            return Decision(now, "NONE", "HOLD", 0.0, "signal-stale")
        if inf.latency_ms > MAX_INFERENCE_LATENCY_MS:
            return Decision(now, "NONE", "HOLD", 0.0, "inference-latency")
        regime = self.kronos.regime()
        conf_ce = inf.confidence_for("CE")
        conf_pe = inf.confidence_for("PE")
        if regime.direction == "down" and conf_ce >= CE_THRESHOLD:
            return Decision(now, "CE", "HOLD", conf_ce, "kronos-veto", kronos_direction=regime.direction)
        if regime.direction == "up" and conf_pe >= PE_THRESHOLD:
            return Decision(now, "PE", "HOLD", conf_pe, "kronos-veto", kronos_direction=regime.direction)
        ce_ok = conf_ce >= CE_THRESHOLD
        pe_ok = conf_pe >= PE_THRESHOLD
        if ce_ok and pe_ok:
            return Decision(now, "NONE", "HOLD", max(conf_ce, conf_pe), "conflicting-signals", kronos_direction=regime.direction)
        if ce_ok:
            return Decision(now, "CE", "BUY", conf_ce, "threshold-met", kronos_direction=regime.direction)
        if pe_ok:
            return Decision(now, "PE", "BUY", conf_pe, "threshold-met", kronos_direction=regime.direction)
        return Decision(now, "NONE", "HOLD", max(conf_ce, conf_pe), "confidence-low", kronos_direction=regime.direction)


# ---------------------------------------------------------------------------
# 6. Risk management
# ---------------------------------------------------------------------------
class CircuitBreaker:
    TRIP_REASONS = (
        "daily-loss-exceeded",
        "too-many-rejected-orders",
        "broker-disconnected",
        "market-data-stale",
        "inference-latency",
        "consecutive-model-errors",
    )

    def __init__(self):
        self.tripped = False
        self.trip_date = None
        self.consecutive_rejections = 0
        self.consecutive_model_errors = 0
        self.live_pnl_pts = 0.0

    def check(self):
        if self.tripped and self.trip_date != datetime.now().date():
            print("[BREAKER] circuit reset (new day)", flush=True)
            self.tripped = False
            self.trip_date = None

    def trip(self, reason):
        self.tripped = True
        self.trip_date = datetime.now().date()
        print(f"[BREAKER] CIRCUIT OPEN: {reason}", flush=True)

    def on_rejected_order(self):
        self.consecutive_rejections += 1
        if self.consecutive_rejections >= MAX_CONSECUTIVE_REJECTIONS:
            self.trip("too-many-rejected-orders")

    def on_model_error(self):
        self.consecutive_model_errors += 1
        if self.consecutive_model_errors >= MAX_MODEL_ERRORS:
            self.trip("consecutive-model-errors")

    def on_model_ok(self):
        self.consecutive_model_errors = 0

    def on_daily_loss(self, pts):
        self.live_pnl_pts = pts
        if pts <= -DAILY_LOSS_LIMIT_PTS:
            self.trip("daily-loss-exceeded")


class RiskManager:
    def __init__(self, client, breaker: CircuitBreaker):
        self.client = client
        self.breaker = breaker

    def allow(self, decision: Decision, snap: Snapshot):
        if not snap.fresh():
            return "market-data-stale"
        self.breaker.check()
        if self.breaker.tripped:
            return "circuit-open"
        try:
            funds = self.client.funds()
            if funds.get("data", {}).get("availablecash", 1) <= 0:
                return "margin-unavailable"
        except Exception as exc:
            print(f"[RISK] funds check failed ({exc}) -> blocking", flush=True)
            return "margin-unavailable"
        return None


# ---------------------------------------------------------------------------
# 7. Execution
# ---------------------------------------------------------------------------
class Execution:
    def __init__(self, client, breaker: CircuitBreaker, owner: PositionOwner | None = None):
        self.client = client
        self.breaker = breaker
        self._entered_symbols: set[str] = set()
        self._lock = threading.Lock()
        self._owner = owner
        self._owned_positions: dict[str, str] = {}  # symbol -> position_id

    def _wait_fill(self, order_id: str, timeout: float = 20.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            try:
                status = self.client.orderstatus(order_id, strategy=STRATEGY_NAME)
                data = status.get("data", {})
                if data.get("status") == "complete":
                    return float(data.get("averageprice") or data.get("price") or 0.0)
            except Exception as exc:
                print(f"[EXEC] orderstatus error ({exc})", flush=True)
            time.sleep(0.5)
        return None

    @staticmethod
    def _entry_qty(snap: Snapshot) -> int:
        lots = snap.lotsize or QUANTITY
        return max(lots, int(QUANTITY / lots) * lots)

    def buy_and_leg(self, side: str, snap: Snapshot):
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
        self.breaker.on_model_ok()
        fill = self._wait_fill(order_id)
        if fill is None:
            print(f"[EXEC] fill timeout for {symbol} - leaving broker legs", flush=True)
            return
        with self._lock:
            self._entered_symbols.add(symbol)
        if self._owner:
            try:
                pid = self._owner.register_position(
                    STRATEGY_ID, symbol, OPTIONS_EXCHANGE, qty, PRODUCT,
                    entry_order_id=order_id,
                )
                with self._lock:
                    self._owned_positions[symbol] = pid
                print(f"[EXEC] registered ownership {symbol} -> {pid}", flush=True)
            except Exception as exc:
                print(f"[EXEC] ownership registration failed ({exc})", flush=True)
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
        print(f"[EXEC] {side} {symbol} filled @ {fill} | SL {trigger} | TARGET {fill + TARGET_POINTS:.2f}", flush=True)

    def square_off(self):
        print("[EXEC] square-off window", flush=True)
        try:
            self.client.cancelallorder(strategy=STRATEGY_NAME)
        except Exception as exc:
            print(f"[EXEC] cancelallorder error ({exc})", flush=True)
        with self._lock:
            owned = set(self._entered_symbols)
        if not owned:
            print("[EXEC] no owned symbols to square off", flush=True)
            return
        try:
            positions = self.client.positionbook()
            for p in positions.get("data", []):
                sym = p.get("symbol", "")
                if p.get("netqty") and sym in owned:
                    self.client.closeposition(
                        strategy=STRATEGY_NAME,
                        symbol=sym,
                        exchange=OPTIONS_EXCHANGE,
                    )
                    print(f"[EXEC] closed position {sym}", flush=True)
                    if self._owner and sym in self._owned_positions:
                        try:
                            self._owner.release_position(
                                self._owned_positions[sym], STRATEGY_ID,
                            )
                            del self._owned_positions[sym]
                        except Exception as exc:
                            print(f"[EXEC] ownership release failed ({sym}: {exc})", flush=True)
        except Exception as exc:
            print(f"[EXEC] square-off error ({exc})", flush=True)
        with self._lock:
            self._entered_symbols.clear()


# ---------------------------------------------------------------------------
# Backfill / IV helpers (main-thread only)
# ---------------------------------------------------------------------------
_last_iv_fetch = 0.0


def _refresh_iv(client, snap: Snapshot):
    global _last_iv_fetch
    now = time.time()
    if now - _last_iv_fetch < IV_CADENCE_SECONDS:
        return
    _last_iv_fetch = now
    for side, sym in (("ce", snap.ce_symbol), ("pe", snap.pe_symbol)):
        if not sym:
            continue
        try:
            greeks = client.optiongreeks(symbol=sym, exchange=OPTIONS_EXCHANGE)
            iv = greeks.get("implied_volatility")
            if iv is not None:
                if side == "ce":
                    snap.iv_ce = float(iv)
                else:
                    snap.iv_pe = float(iv)
        except Exception as exc:
            print(f"[DATA] optiongreeks failed for {sym} ({exc})", flush=True)


def _backfill_1m(client, symbol: str, exchange: str, days: int = BACKFILL_DAYS) -> list:
    import pandas as pd

    end = datetime.now()
    start = end - timedelta(days=days)
    try:
        df = client.history(
            symbol=symbol,
            exchange=exchange,
            interval="1m",
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
        )
    except Exception as exc:
        print(f"[DATA] history failed for {symbol} ({exc})", flush=True)
        return []
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    rows = []
    for ts, r in df.iterrows():
        rows.append(
            (
                int(ts.timestamp() // 60),
                float(r["open"]),
                float(r["high"]),
                float(r["low"]),
                float(r["close"]),
                float(r.get("volume", 0) or 0),
            )
        )
    return rows


def _resync_windows(client, snap: Snapshot, buffers: WindowBuffer):
    symbols = [snap.ce_symbol, snap.pe_symbol]
    buffers.reset(symbols)
    for side, sym in (("ce", snap.ce_symbol), ("pe", snap.pe_symbol)):
        rows = _backfill_1m(client, sym, OPTIONS_EXCHANGE)
        if rows:
            iv = snap.iv_ce if side == "ce" else snap.iv_pe
            buffers.backfill(sym, rows, iv)
            print(f"[DATA] backfilled {len(rows)} 1m bars for {sym} (iv={iv:.2f})", flush=True)
        else:
            print(f"[DATA] no history for {sym} - live accumulation only", flush=True)


def _hhmm_to_epoch(hhmm: str) -> int:
    h, m = hhmm.split(":")
    return (int(h) * 3600 + int(m) * 60) % 86400


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(
        f"START mode={MODE or 'live'} inference={INFERENCE_MODE} expiry={EXPIRY_DATE}",
        flush=True,
    )
    client = api(api_key=API_KEY, host=HOST, ws_url=WS_URL)

    register_strategy_mode(STRATEGY_ID, ExecutionMode.SWING)
    stop_mode = StopMode.STOP_AND_CLOSE
    persister = StatePersister()
    owner = PositionOwner(persister)
    reconciler = Reconciler(persister, owner)

    try:
        broker_positions = client.positionbook().get("data", [])
    except Exception as exc:
        print(f"[MAIN] positionbook failed ({exc}) - skipping reconciliation", flush=True)
        broker_positions = []
    result = reconciler.reconcile(STRATEGY_ID, broker_positions)
    if not result.is_clean:
        print(f"[MAIN] reconciliation blocked: {result.blocked_reason}", flush=True)
        print("[MAIN] resolve orphan positions before trading", flush=True)
    else:
        print(f"[MAIN] reconciliation clean: {len(result.owned)} owned, {len(result.stale)} stale", flush=True)
    RecoveryManager(STRATEGY_ID, persister, owner, reconciler).startup(
        lambda: client.positionbook().get("data", []),
    )

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

    snapshot = Snapshot()
    buffers = WindowBuffer()
    data = DataAcquisition(client, snapshot, buffers)
    data.start()
    inference = make_inference()
    kronos = KronosFilter(client)
    engine = DecisionEngine(kronos)
    breaker = CircuitBreaker()
    risk = RiskManager(client, breaker)
    execr = Execution(client, breaker, owner=owner)

    stop = threading.Event()

    def _sigterm_handler(*_):
        print(f"[MAIN] SIGTERM received (stop_mode={stop_mode.value})", flush=True)
        if stop_mode == StopMode.STOP_AND_CLOSE:
            try:
                execr.square_off()
            except Exception as exc:
                print(f"[MAIN] SIGTERM square-off error ({exc})", flush=True)
        else:
            print("[MAIN] STOP_TRADING_ONLY - positions left open", flush=True)
        stop.set()

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    last_decided = 0.0
    synced_symbols = set()
    last_resolve = 0.0
    while not stop.is_set():
        now = time.time()
        if not (snapshot.ce_symbol and snapshot.pe_symbol):
            if now - last_resolve > 5:
                last_resolve = now
                data.resolve_atm()
                if snapshot.ce_symbol and snapshot.pe_symbol:
                    data.client.subscribe_ltp(data.instruments, data._on_ltp)
            time.sleep(1)
            continue
        if now % 86400 > _hhmm_to_epoch(SQUARE_OFF_HHMM):
            execr.square_off()
            stop.set()
            break
        _refresh_iv(client, snapshot)
        current = {snapshot.ce_symbol, snapshot.pe_symbol}
        if current != synced_symbols:
            _resync_windows(client, snapshot, buffers)
            synced_symbols = current
        if snapshot.fresh() and now - last_decided > 1.0 and snapshot.ce_symbol and snapshot.pe_symbol:
            try:
                features = make_features(snapshot, buffers)
                if (
                    buffers.count(snapshot.ce_symbol) < MIN_WINDOW_ROWS
                    or buffers.count(snapshot.pe_symbol) < MIN_WINDOW_ROWS
                ):
                    decision = Decision(now, "NONE", "HOLD", 0.0, "window-warming")
                    DecisionEngine.log(decision, features)
                    last_decided = now
                    continue
                inf = inference.infer(features)
                if not inference.healthy():
                    breaker.on_model_error()
                    time.sleep(1)
                    continue
            except Exception as exc:
                print(f"MODEL error ({exc})", flush=True)
                breaker.on_model_error()
                time.sleep(1)
                continue
            decision = engine.evaluate(snapshot, inf)
            DecisionEngine.log(decision, features)
            if decision.action == "BUY":
                block = risk.allow(decision, snapshot)
                if block:
                    print(f"BLOCKED {decision.side} ({block})", flush=True)
                else:
                    threading.Thread(
                        target=execr.buy_and_leg,
                        args=(decision.side, snapshot),
                        daemon=True,
                    ).start()
            last_decided = now
        time.sleep(0.5)
    print("STOP clean", flush=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ATM premium ML strategy (NIFTY options)")
    parser.add_argument(
        "--mode",
        default=None,
        help="Execution mode: live | paper | backtest (default: $MODE env or live)",
    )
    args = parser.parse_args()
    if args.mode:
        globals()["MODE"] = args.mode
    main()
