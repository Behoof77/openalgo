"""Tests for mcp.upstox_company_data (company_snapshot fundamentals provider).

Covers the fetch-on-miss contract, per-kind freshness TTLs, the atomic cache
write, graceful degradation when no broker token is available, and the
envelope shapes consumed by mcpserver's ``company_snapshot`` tool. The
provider is loaded via importlib file-path loading (the same technique
mcpserver uses), so the installed PyPI ``mcp`` SDK package cannot shadow it.
"""

import importlib.util
import json
import os
import pathlib
from datetime import datetime, timedelta

import pytest

_MODULE_PATH = (
    pathlib.Path(__file__).resolve().parent.parent / "mcp" / "upstox_company_data.py"
)


def _load_provider():
    # The provider imports database.auth_db, which fails fast at import time
    # without a configured environment - same bootstrap as other auth tests.
    os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
    os.environ.setdefault("API_KEY_PEPPER", "t" * 64)
    spec = importlib.util.spec_from_file_location(
        "_upstox_company_data_under_test", _MODULE_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod = _load_provider()
get_section = mod.get_section


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Point every cache directory at a throwaway tmp_path tree."""
    monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(mod, "_HISTORY_DIR", tmp_path / "company_history")
    monkeypatch.setattr(mod, "_NEWS_DIR", tmp_path / "company_news")
    monkeypatch.setattr(mod, "_FUNDAMENTALS_DIR", tmp_path / "company_fundamentals")

    def _no_network(*_args, **_kwargs):
        raise RuntimeError("network disabled in tests")

    # History is fallback-eligible, so existing no-token/failure tests would
    # otherwise hit real Yahoo Finance. Fallback tests opt in with their own
    # _fetch_yfinance patch.
    monkeypatch.setattr(mod, "_fetch_yfinance", _no_network)
    return tmp_path


def _write_cache(kind, symbol, exchange, payload):
    path = mod._cache_path(kind, symbol, exchange)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def _now_iso():
    return mod._now_iso()


def _iso_seconds_ago(seconds):
    return (datetime.now(mod.IST) - timedelta(seconds=seconds)).isoformat(
        timespec="seconds"
    )


def _history_payload(fetched_at, candle_count=0):
    candles = []
    for idx in range(candle_count):
        candles.append(
            {
                "timestamp": _now_iso(),
                "open": 100.0 + idx,
                "high": 110.0 + idx,
                "low": 99.0 + idx,
                "close": 105.0 + idx,
                "volume": 100 + idx,
                "oi": 0,
            }
        )
    return {
        "fetched_at": fetched_at,
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "interval": "D",
        "start_date": "2025-08-01",
        "end_date": "2026-08-01",
        "candle_count": candle_count,
        "candles": candles,
    }


def test_unknown_kind_raises_value_error(isolated_cache):
    with pytest.raises(ValueError):
        get_section("market-cap", "RELIANCE", "NSE")


def test_fresh_cache_hit_never_touches_token_resolution(isolated_cache, monkeypatch):
    now = _now_iso()
    _write_cache("history", "RELIANCE", "NSE", _history_payload(now, candle_count=2))

    def _forbid_token(*_args, **_kwargs):
        raise AssertionError("token resolution must not run on a fresh cache hit")

    monkeypatch.setattr(mod, "_resolve_token", _forbid_token)

    envelope = get_section("history", "reliance", "nse")

    assert envelope["source"] == "upstox:historical-candle (file cache)"
    assert envelope["provider"] == "upstox"
    assert envelope["freshness"] == "cached"
    assert envelope["status"] == {
        "status": "cache_hit",
        "provider": "upstox",
        "cached": True,
        "can_refresh": True,
        "last_updated": now,
    }
    assert envelope["data"]["candle_count"] == 2
    assert envelope["data"]["data"][0]["close"] == 105.0
    assert envelope["data"]["data"][0]["timestamp"] == now


def test_stale_cache_refetches_and_updates_file(isolated_cache, monkeypatch):
    stale = _iso_seconds_ago(mod.SECTION_TTL_SECONDS["history"] * 2)
    _write_cache("history", "RELIANCE", "NSE", _history_payload(stale))

    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))
    fetched = _history_payload(_now_iso(), candle_count=1)

    def _fake_fetch(kind, symbol, exchange, auth_token, *, interval, days):
        assert kind == "history"
        assert symbol == "RELIANCE"
        assert exchange == "NSE"
        assert auth_token == "TOKEN"
        assert interval == "D"
        assert days == 365
        return fetched

    monkeypatch.setattr(mod, "_fetch_payload", _fake_fetch)

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY")

    assert envelope["freshness"] == "live"
    assert envelope["status"]["status"] == "fetched_live"
    assert envelope["status"]["cached"] is True
    assert envelope["status"]["can_refresh"] is True
    assert envelope["latency_ms"] >= 0.0

    stored = json.loads(mod._cache_path("history", "RELIANCE", "NSE").read_text())
    assert stored["fetched_at"] == fetched["fetched_at"]
    assert not list(isolated_cache.rglob("*.tmp"))


def test_stale_at_ttl_boundary_refetches(isolated_cache, monkeypatch):
    boundary = _iso_seconds_ago(mod.SECTION_TTL_SECONDS["history"] + 1)
    _write_cache("history", "RELIANCE", "NSE", _history_payload(boundary))

    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))
    monkeypatch.setattr(
        mod, "_fetch_payload", lambda *a, **k: _history_payload(_now_iso())
    )

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY")

    assert envelope["freshness"] == "live"
    assert envelope["status"]["status"] == "fetched_live"


def test_force_bypasses_fresh_cache(isolated_cache, monkeypatch):
    now = _now_iso()
    _write_cache("history", "RELIANCE", "NSE", _history_payload(now))

    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))
    monkeypatch.setattr(
        mod, "_fetch_payload", lambda *a, **k: _history_payload(_now_iso(), candle_count=1)
    )

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY", force=True)

    assert envelope["freshness"] == "live"
    assert envelope["status"]["status"] == "fetched_live"


def test_no_token_stale_cache_returns_stale_without_refresh(isolated_cache, monkeypatch):
    stale = _iso_seconds_ago(mod.SECTION_TTL_SECONDS["news"] * 2)
    payload = {
        "fetched_at": stale,
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "instrument_key": "NSE_EQ|INE002A01018",
        "news_count": 1,
        "news": [{"heading": "old"}],
    }
    _write_cache("news", "RELIANCE", "NSE", payload)

    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: (None, None))

    envelope = get_section("news", "RELIANCE", "NSE")

    assert envelope["freshness"] == "stale"
    assert envelope["status"]["status"] == "cache_hit"
    assert envelope["status"]["can_refresh"] is False
    assert "no active broker token" in envelope["data"]["note"]
    assert envelope["data"]["news"][0]["heading"] == "old"


def test_no_token_no_cache_returns_miss_without_refresh(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: (None, None))

    envelope = get_section("history", "RELIANCE", "NSE")

    assert envelope["freshness"] == "miss"
    assert envelope["status"]["status"] == "cache_miss"
    assert envelope["status"]["cached"] is False
    assert envelope["status"]["can_refresh"] is False
    assert envelope["status"]["provider"] == "upstox"
    assert envelope["data"] == {
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "note": (
            "No cached history for RELIANCE; refresh unavailable: "
            "no active broker token."
        ),
    }


def test_fetch_failure_keeps_stale_cache_with_note(isolated_cache, monkeypatch):
    stale = _iso_seconds_ago(mod.SECTION_TTL_SECONDS["history"] * 2)
    _write_cache("history", "RELIANCE", "NSE", _history_payload(stale, candle_count=2))

    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "_fetch_payload", _boom)

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY")

    assert envelope["freshness"] == "stale"
    assert envelope["status"]["status"] == "cache_hit"
    assert envelope["status"]["can_refresh"] is True
    assert "refresh failed" in envelope["data"]["note"]
    assert envelope["data"]["candle_count"] == 2


def test_fetch_failure_no_cache_returns_miss_with_refresh(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr(mod, "_fetch_payload", _boom)

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY")

    assert envelope["freshness"] == "miss"
    assert envelope["status"]["status"] == "cache_miss"
    assert envelope["status"]["can_refresh"] is True
    assert envelope["data"]["note"] == "Unable to refresh history for RELIANCE: network down"


def test_fundamentals_section_data_strips_fetched_at(isolated_cache):
    now = _now_iso()
    payload = {
        "fetched_at": now,
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "isin": "INE002A01018",
        "field": "key-ratios",
        "count": 2,
        "data": [{"label": "pe", "value": 22.5}, {"label": "roe", "value": 14.2}],
    }
    _write_cache("key-ratios", "RELIANCE", "NSE", payload)

    envelope = get_section("key-ratios", "RELIANCE", "NSE")

    assert envelope["source"] == "upstox:key-ratios (file cache)"
    assert envelope["freshness"] == "cached"
    assert envelope["status"]["status"] == "cache_hit"
    assert "fetched_at" not in envelope["data"]
    assert envelope["data"]["count"] == 2
    assert envelope["data"]["data"][0]["label"] == "pe"


def test_write_cache_is_atomic(isolated_cache):
    payload = {"fetched_at": _now_iso(), "count": 1, "data": []}
    path = mod._cache_path("company-profile", "RELIANCE", "NSE")

    mod._write_cache_file(path, payload)

    assert json.loads(path.read_text()) == payload
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob(".*.tmp"))


def _yfinance_history_payload(fetched_at, candle_count=1):
    candles = [
        {
            "timestamp": _now_iso(),
            "open": 100.0,
            "high": 110.0,
            "low": 99.0,
            "close": 105.0,
            "volume": 100,
            "oi": 0,
        }
        for _ in range(candle_count)
    ]
    return {
        "fetched_at": fetched_at,
        "symbol": "RELIANCE",
        "exchange": "NSE",
        "interval": "D",
        "start_date": "2025-08-01",
        "end_date": "2026-08-01",
        "candle_count": candle_count,
        "candles": candles,
        "provider_hint": "yfinance",
    }


def test_no_token_fallback_serves_live_yfinance_envelope(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: (None, None))
    fetched = _yfinance_history_payload(_now_iso(), candle_count=2)
    monkeypatch.setattr(mod, "_fetch_yfinance", lambda *a, **k: fetched)

    envelope = get_section("history", "RELIANCE", "NSE")

    assert envelope["provider"] == "yfinance"
    assert envelope["source"] == "yfinance:historical-candle (live)"
    assert envelope["freshness"] == "live"
    assert envelope["status"] == {
        "status": "fetched_live",
        "provider": "yfinance",
        "cached": True,
        "can_refresh": True,
        "last_updated": fetched["fetched_at"],
    }
    assert envelope["data"]["candle_count"] == 2
    assert envelope["data"]["data"][0]["close"] == 105.0
    stored = json.loads(mod._cache_path("history", "RELIANCE", "NSE").read_text())
    assert stored["provider_hint"] == "yfinance"


def test_no_token_fallback_failure_preserves_miss(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: (None, None))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("yfinance down")

    monkeypatch.setattr(mod, "_fetch_yfinance", _boom)

    envelope = get_section("history", "RELIANCE", "NSE")

    assert envelope["freshness"] == "miss"
    assert envelope["status"]["status"] == "cache_miss"
    assert envelope["status"]["can_refresh"] is False
    assert envelope["provider"] == "upstox"


def test_non_eligible_kind_skips_fallback(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: (None, None))

    def _must_not_call(*_args, **_kwargs):
        raise AssertionError("news must never reach the yfinance fallback")

    monkeypatch.setattr(mod, "_fetch_yfinance", _must_not_call)

    envelope = get_section("news", "RELIANCE", "NSE")

    assert envelope["freshness"] == "miss"
    assert envelope["status"]["can_refresh"] is False


def test_fetch_failure_falls_back_to_yfinance(isolated_cache, monkeypatch):
    monkeypatch.setattr(mod, "_resolve_token", lambda api_key: ("TOKEN", "upstox"))

    def _boom(*_args, **_kwargs):
        raise RuntimeError("upstox refresh failed")

    monkeypatch.setattr(mod, "_fetch_payload", _boom)
    fetched = _yfinance_history_payload(_now_iso(), candle_count=1)
    monkeypatch.setattr(mod, "_fetch_yfinance", lambda *a, **k: fetched)

    envelope = get_section("history", "RELIANCE", "NSE", api_key="KEY")

    assert envelope["freshness"] == "live"
    assert envelope["status"]["status"] == "fetched_live"
    assert envelope["provider"] == "yfinance"
    assert envelope["status"]["can_refresh"] is True
