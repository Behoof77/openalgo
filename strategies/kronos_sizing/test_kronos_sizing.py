"""Unit tests for kronos_sizing.py.

Covers the frozen Kronos tier boundaries, fail-closed behaviour, quantity
rounding, the base signal engine, and the full validate-mode state machine
lifecycle (signal -> Kronos -> sized order -> open -> SL/TP/window exit).

Run from the repo root:
    .venv\\Scripts\\python.exe -m pytest strategies/kronos_sizing/test_kronos_sizing.py -v
"""

import json
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

import kronos_sizing as ks


# ---------------------------------------------------------------------------
# test doubles
# ---------------------------------------------------------------------------
def make_ohlcv(n: int = 512, spike: float = 130.0) -> pd.DataFrame:
    """Build a 5m OHLCV frame that ends with a decisive signal bar.

    Closes decline linearly (100 -> 80) then spike on the last bar, which
    forces EMA9 above EMA21 on the final row (bullish cross). A DatetimeIndex
    ending now is required so the strategy's stale-market guard does not skip.
    """
    index = pd.date_range(end=pd.Timestamp.now().floor("min"), periods=n, freq="5min")
    closes = np.linspace(100.0, 80.0, n)
    closes[-1] = spike
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 1.0
    lows = np.minimum(opens, closes) - 1.0
    volume = np.full(n, 1000.0)
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=index,
    )


class FakeClient:
    """Minimal OpenAlgo SDK stand-in for state-machine tests."""

    def __init__(self):
        self.history_df = make_ohlcv()
        self.position_rows = []
        self.order_status_map = {}
        self.quotes_price = None
        self.placed = []

    def history(self, symbol=None, exchange=None, interval=None, start_date=None, end_date=None):
        return self.history_df

    def quotes(self, symbol=None, exchange=None):
        return {"data": {"ltp": self.quotes_price}} if self.quotes_price is not None else {"data": {}}

    def positionbook(self):
        return {"data": self.position_rows}

    def orderstatus(self, order_id=None, strategy=None):
        return {"data": {"order_status": self.order_status_map.get(order_id, "")}}

    def placeorder(self, **kwargs):
        self.placed.append(kwargs)
        return {"data": {"orderid": f"FAKE-{len(self.placed)}"}}


class StubKronos:
    """KronosClient stand-in returning a canned forecast; counts calls."""

    def __init__(self, forecast: ks.Forecast):
        self.forecast = forecast
        self.calls = 0

    def get_forecast(self, ohlcv, freq=None, profile=None, now=None):
        self.calls += 1
        return self.forecast


class FakeResp:
    def __init__(self, status_code=200, text="", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self):
        return self._payload


# ---------------------------------------------------------------------------
# Layer 2 pure functions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("strength", "expected"),
    [
        (0.0499, 0.50),
        (0.050, 0.75),
        (0.0749, 0.75),
        (0.075, 1.00),
        (0.1, 1.00),
    ],
)
def test_strength_weight_tiers(strength, expected):
    assert ks.kronos_strength_weight(strength) == expected


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf"), float("-inf"), None])
def test_strength_weight_invalid_inputs(bad):
    with pytest.raises(ValueError):
        ks.kronos_strength_weight(bad)


def test_forecast_return_converted_from_price():
    # raw_predictions are PRICE forecasts; return = (forecast - last_close)/last_close*100
    assert ks.forecast_return_pct_from_raw([100.01], 100.0) == pytest.approx(0.01)
    assert ks.forecast_return_pct_from_raw([99.9], 100.0) == pytest.approx(-0.1)
    assert ks.forecast_return_pct_from_raw([100.05, 100.15, 99.95], 100.0) == pytest.approx(0.05)


def test_forecast_return_empty_is_none():
    assert ks.forecast_return_pct_from_raw([], 100.0) is None
    assert ks.forecast_return_pct_from_raw(None, 100.0) is None
    assert ks.forecast_return_pct_from_raw([None, float("nan")], 100.0) is None
    assert ks.forecast_return_pct_from_raw([100.0], None) is None
    assert ks.forecast_return_pct_from_raw([100.0], 0.0) is None
    assert ks.forecast_return_pct_from_raw([100.0], float("nan")) is None


def test_strength_is_abs():
    assert ks.strength_from_raw([100.02, 100.04, 99.97], 100.0) == pytest.approx(0.01)
    assert ks.strength_from_raw([99.9], 100.0) == pytest.approx(0.1)
    assert ks.strength_from_raw([], 100.0) is None


def test_forecast_price_user_example_24500():
    # last_close=24500, Kronos forecasts 24512.25 -> 0.05% -> mid tier 0.75x
    forecast = ks.forecast_return_pct_from_raw([24512.25], 24500.0)
    assert forecast == pytest.approx(0.05)
    # production rounds to 6dp before the tier lookup (boundary float precision)
    assert ks.kronos_strength_weight(round(forecast, 6)) == 0.75


def test_round_down_to_lot():
    assert ks.round_down_to_lot(56.25, 25) == 50
    assert ks.round_down_to_lot(37.5, 25) == 25
    assert ks.round_down_to_lot(12, 25) == 0
    assert ks.round_down_to_lot(10, 0) == 0
    assert ks.round_down_to_lot(0, 25) == 0


# ---------------------------------------------------------------------------
# Layer 1 base signal engine
# ---------------------------------------------------------------------------
def test_base_signal_long_on_spike():
    df = make_ohlcv(512, spike=130.0)
    sig = ks.base_signal_engine(df)
    assert sig.direction == +1
    assert sig.entry == pytest.approx(130.0)
    assert sig.atr is not None and sig.atr > 0
    assert sig.reason == "ema fast cross above slow, bullish filter"


def test_base_signal_short_on_drop():
    df = make_ohlcv(512, spike=60.0)
    df["close"] = np.linspace(80.0, 100.0, len(df))
    df["close"].iloc[-1] = 60.0
    df["open"] = np.roll(df["close"], 1)
    df["open"].iloc[0] = df["close"].iloc[0]
    df["high"] = np.maximum(df["open"], df["close"]) + 1.0
    df["low"] = np.minimum(df["open"], df["close"]) - 1.0
    sig = ks.base_signal_engine(df)
    assert sig.direction == -1
    assert sig.entry == pytest.approx(60.0)
    assert sig.reason == "ema fast cross below slow, bearish filter"


def test_base_signal_insufficient_bars():
    sig = ks.base_signal_engine(make_ohlcv(5))
    assert sig.direction == 0
    assert sig.reason == "insufficient bars"


def test_base_signal_no_cross_flat():
    df = make_ohlcv(512)
    df["close"] = np.full(len(df), 100.0)
    df["open"] = np.full(len(df), 100.0)
    df["high"] = np.full(len(df), 101.0)
    df["low"] = np.full(len(df), 99.0)
    sig = ks.base_signal_engine(df)
    assert sig.direction == 0
    assert sig.reason == "no cross on last bar"


def test_compute_atr_positive():
    atr = ks.compute_atr(make_ohlcv(100))
    assert atr.notna().all()
    assert (atr > 0).all()


# ---------------------------------------------------------------------------
# KronosClient (HTTP + parsing)
# ---------------------------------------------------------------------------
def test_get_forecast_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["body"] = json
        captured["timeout"] = timeout
        return FakeResp(
            payload={
                "prediction": 1,
                "confidence": 0.8,
                "raw_predictions": [130.1, 130.05, 130.15],
                "inference_ms": 12.5,
                "rows_received": 512,
            }
        )

    monkeypatch.setattr(ks.requests, "post", fake_post)
    client = ks.KronosClient("http://x:8000")
    forecast = client.get_forecast(make_ohlcv(600), freq="5min", profile="fast")

    assert forecast.success
    assert forecast.last_close == pytest.approx(130.0)
    assert forecast.forecast_price == pytest.approx(130.1)
    expected_ret = round((130.1 - 130.0) / 130.0 * 100.0, 6)
    assert forecast.forecast_return_pct == pytest.approx(expected_ret, abs=1e-12)
    assert forecast.strength_pct == pytest.approx(expected_ret, abs=1e-12)
    assert forecast.signal == 1
    assert forecast.confidence == pytest.approx(0.8)
    assert captured["url"] == "http://x:8000/predict"
    assert len(captured["body"]["data"]) == 600
    assert captured["body"]["freq"] == "5min"
    assert captured["body"]["profile"] == "fast"
    assert "volume" in captured["body"]["data"][0]
    assert captured["timeout"] == (5.0, 5.0)


def test_get_forecast_stale(monkeypatch):
    monkeypatch.setattr(ks.requests, "post", lambda *a, **k: FakeResp(payload={"prediction": 1}))
    monotonic_ticks = [0.0, 100.0]
    monkeypatch.setattr(ks.time, "monotonic", lambda: monotonic_ticks.pop(0))
    forecast = ks.KronosClient("http://x:8000").get_forecast(make_ohlcv(600))
    assert not forecast.success
    assert "stale" in forecast.error


def test_get_forecast_insufficient_rows(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["called"] = True
        return FakeResp(payload={"prediction": 1})

    monkeypatch.setattr(ks.requests, "post", fake_post)
    forecast = ks.KronosClient("http://x:8000").get_forecast(make_ohlcv(100))
    assert not forecast.success
    assert "insufficient rows" in forecast.error
    assert "called" not in captured  # never called the server


def test_get_forecast_empty_raw_fail_closed(monkeypatch):
    monkeypatch.setattr(
        ks.requests,
        "post",
        lambda *a, **k: FakeResp(payload={"prediction": 1, "raw_predictions": []}),
    )
    forecast = ks.KronosClient("http://x:8000").get_forecast(make_ohlcv(600))
    assert not forecast.success
    assert "fail-closed" in forecast.error
    assert forecast.last_close == pytest.approx(130.0)


def test_get_forecast_timeout(monkeypatch):
    def raise_timeout(*a, **k):
        raise ks.requests.Timeout("boom")

    monkeypatch.setattr(ks.requests, "post", raise_timeout)
    forecast = ks.KronosClient("http://x:8000").get_forecast(make_ohlcv(600))
    assert not forecast.success
    assert "Kronos unreachable/timeout" in forecast.error


def test_get_forecast_http_error(monkeypatch):
    monkeypatch.setattr(ks.requests, "post", lambda *a, **k: FakeResp(status_code=500, text="oops"))
    forecast = ks.KronosClient("http://x:8000").get_forecast(make_ohlcv(600))
    assert not forecast.success
    assert "HTTP 500" in forecast.error


# ---------------------------------------------------------------------------
# lifecycle gates
# ---------------------------------------------------------------------------
def test_start_exits_without_api_key(tmp_path):
    strategy = ks.KronosSizingStrategy(
        client=FakeClient(), env={"OPENALGO_API_KEY": "", "DATA_DIR": str(tmp_path)}
    )
    with pytest.raises(SystemExit) as exc:
        strategy.start()
    assert exc.value.code == 1


def test_start_exits_live_without_validation(tmp_path):
    strategy = ks.KronosSizingStrategy(
        client=FakeClient(),
        env={
            "OPENALGO_API_KEY": "k",
            "MODE": "live",
            "BASE_ENGINE_VALIDATED": "false",
            "DATA_DIR": str(tmp_path),
        },
    )
    with pytest.raises(SystemExit) as exc:
        strategy.start()
    assert exc.value.code == 1


def test_start_allows_live_when_validated(tmp_path):
    strategy = ks.KronosSizingStrategy(
        client=FakeClient(),
        env={
            "OPENALGO_API_KEY": "k",
            "MODE": "live",
            "BASE_ENGINE_VALIDATED": "true",
            "DATA_DIR": str(tmp_path),
        },
    )
    strategy.start()  # must not raise


# ---------------------------------------------------------------------------
# state machine (validate mode)
# ---------------------------------------------------------------------------
def make_strategy(tmp_path, **env_overrides) -> ks.KronosSizingStrategy:
    env = {"MODE": "validate", "DATA_DIR": str(tmp_path), "BASE_QTY": "75"}
    env.update(env_overrides)
    return ks.KronosSizingStrategy(client=FakeClient(), env=env)


def test_fail_closed_forecast_blocks_trade(tmp_path):
    strategy = make_strategy(tmp_path)
    strategy.kronos = StubKronos(ks.Forecast(success=False, error="kronos down"))
    strategy.symbol = "NIFTY26AUGFUT"
    strategy.lot_size = 25
    strategy.client.history_df = make_ohlcv(512)

    strategy.run_once()
    assert strategy.state == ks.State.SIGNAL_DETECTED

    strategy.run_once()
    assert strategy.state == ks.State.KRONOS_PENDING

    strategy.run_once()
    assert strategy.state == ks.State.IDLE
    assert strategy.position is None


def test_success_validate_lifecycle(tmp_path):
    strategy = make_strategy(tmp_path)
    strategy.kronos = StubKronos(
        ks.Forecast(
            success=True,
            last_close=130.0,
            forecast_price=130.078,  # 0.06% above last_close
            forecast_return_pct=0.06,
            strength_pct=0.06,
            signal=1,
        )
    )
    strategy.symbol = "NIFTY26AUGFUT"
    strategy.lot_size = 25
    strategy.client.history_df = make_ohlcv(512)

    strategy.run_once()
    assert strategy.state == ks.State.SIGNAL_DETECTED

    strategy.run_once()  # SIGNAL_DETECTED -> KRONOS_PENDING
    assert strategy.state == ks.State.KRONOS_PENDING

    strategy.run_once()  # KRONOS_PENDING -> position created, ORDER_PENDING
    assert strategy.state == ks.State.ORDER_PENDING
    pos = strategy.position
    assert pos is not None
    assert pos.direction == +1
    assert pos.weight == 0.75  # strength 0.06 -> mid tier
    assert pos.qty == 50  # floor(75 * 0.75 / 25) * 25
    assert pos.sl == pytest.approx(pos.entry_price - 1.0 * pos.atr)
    assert pos.tp == pytest.approx(pos.entry_price + 2.5 * pos.atr)

    strategy.run_once()  # ORDER_PENDING -> POSITION_OPEN (validate simulates fill)
    assert strategy.state == ks.State.POSITION_OPEN
    assert len(strategy.client.placed) == 0  # no real orders in validate

    strategy.client.quotes_price = 100.0  # below SL -> stop hit
    strategy.run_once()
    assert strategy.state == ks.State.EXIT_PENDING

    strategy.run_once()  # EXIT_PENDING -> FLAT
    assert strategy.state == ks.State.FLAT
    assert strategy.position is None

    # decisions log carries the causal forecast fields
    log_path = strategy.decision_log
    records = [json.loads(line) for line in open(log_path, encoding="utf-8")]
    stages = {r["stage"] for r in records}
    assert stages == {"signal", "forecast"}
    forecast_records = [r for r in records if r["stage"] == "forecast"]
    assert len(forecast_records) == 1
    rec = forecast_records[0]
    for key in (
        "signal_time",
        "forecast_request_time",
        "forecast_response_time",
        "last_close",
        "forecast_price",
        "forecast_return_pct",
        "strength_pct",
        "selected_weight",
        "success",
    ):
        assert key in rec
    assert rec["success"] is True
    assert rec["forecast_return_pct"] == pytest.approx(0.06)
    assert rec["selected_weight"] == 0.75


def test_no_signal_never_consults_kronos(tmp_path):
    strategy = make_strategy(tmp_path)
    stub = StubKronos(ks.Forecast(success=True, forecast_return_pct=0.1, strength_pct=0.1, signal=1))
    strategy.kronos = stub
    strategy.symbol = "NIFTY26AUGFUT"
    strategy.lot_size = 25
    df = make_ohlcv(512)
    df["close"] = np.full(len(df), 100.0)
    df["open"] = np.full(len(df), 100.0)
    df["high"] = np.full(len(df), 101.0)
    df["low"] = np.full(len(df), 99.0)
    strategy.client.history_df = df

    strategy.run_once()
    assert strategy.state == ks.State.IDLE
    assert stub.calls == 0


def test_reconcile_mismatch_halts_in_live(tmp_path):
    strategy = make_strategy(
        tmp_path, MODE="live", BASE_ENGINE_VALIDATED="true"
    )
    strategy.symbol = "NIFTY26AUGFUT"
    strategy.position = ks.Position(
        direction=+1,
        symbol="NIFTY26AUGFUT",
        exchange="NFO",
        qty=75,
        entry_price=130.0,
        atr=5.0,
        sl=125.0,
        tp=142.5,
        entry_ts=datetime.now().isoformat(),
        entry_epoch=0.0,
    )
    strategy.state = ks.State.POSITION_OPEN
    strategy.client.position_rows = []  # broker says flat -> mismatch

    strategy.run_once()
    assert strategy.state == ks.State.ERROR


# ---------------------------------------------------------------------------
# order helpers
# ---------------------------------------------------------------------------
def test_order_is_settled(tmp_path):
    strategy = make_strategy(tmp_path)
    strategy.client.order_status_map = {
        "a": "COMPLETE",
        "b": "OPEN",
        "c": "REJECTED",
        "d": "FILLED",
    }
    assert strategy._order_is_settled("a") == "complete"
    assert strategy._order_is_settled("b") == "open"
    assert strategy._order_is_settled("c") == "rejected"
    assert strategy._order_is_settled("d") == "complete"
    assert strategy._order_is_settled("unknown") is None
