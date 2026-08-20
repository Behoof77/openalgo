"""Regression tests for the arbitrage universe service exchange handling.

Covers the request matrix from the v1.0 stabilization pass:
default exchanges, None, empty list, invalid exchanges, NFO-only,
MCX-only, NFO+MCX, and case normalisation. The master-contract lookup
(``fno_search_symbols``) is monkeypatched so the tests are deterministic
and need no broker connection.
"""

import pytest

from services import arbitrage_service as svc


NFO_ROWS = [
    {"symbol": "NIFTY30JUN26FUT", "name": "NIFTY", "exchange": "NFO", "expiry": "30-JUN-26", "lotsize": 75, "tick_size": 0.05},
    {"symbol": "NIFTY31JUL26FUT", "name": "NIFTY", "exchange": "NFO", "expiry": "31-JUL-26", "lotsize": 75, "tick_size": 0.05},
    {"symbol": "NIFTY28AUG26FUT", "name": "NIFTY", "exchange": "NFO", "expiry": "28-AUG-26", "lotsize": 75, "tick_size": 0.05},
    {"symbol": "BANKNIFTY30JUN26FUT", "name": "BANKNIFTY", "exchange": "NFO", "expiry": "30-JUN-26", "lotsize": 30, "tick_size": 0.05},
    {"symbol": "BANKNIFTY31JUL26FUT", "name": "BANKNIFTY", "exchange": "NFO", "expiry": "31-JUL-26", "lotsize": 30, "tick_size": 0.05},
]

MCX_ROWS = [
    {"symbol": "CRUDEOIL29JUN26FUT", "name": "CRUDEOIL", "exchange": "MCX", "expiry": "29-JUN-26", "lotsize": 100, "tick_size": 1.0},
    {"symbol": "CRUDEOIL28JUL26FUT", "name": "CRUDEOIL", "exchange": "MCX", "expiry": "28-JUL-26", "lotsize": 100, "tick_size": 1.0},
    {"symbol": "CRUDEOIL27AUG26FUT", "name": "CRUDEOIL", "exchange": "MCX", "expiry": "27-AUG-26", "lotsize": 100, "tick_size": 1.0},
]

BY_EXCHANGE = {"NFO": NFO_ROWS, "MCX": MCX_ROWS}


def _fake_fno_search_symbols(exchange=None, instrumenttype=None, limit=10000, **kwargs):
    """Return the synthetic rows for the requested exchange."""
    assert instrumenttype == "FUT"
    return BY_EXCHANGE.get(exchange, [])


@pytest.fixture(autouse=True)
def _patch_master_contract(monkeypatch):
    monkeypatch.setattr(svc, "fno_search_symbols", _fake_fno_search_symbols)


def _call(exchanges=None):
    if exchanges is None:
        return svc.get_arbitrage_universe()
    return svc.get_arbitrage_universe(exchanges=exchanges)


def _assert_ok(result, expected_pairs, expected_underlyings, expected_symbols):
    success, response, status = result
    assert success is True
    assert status == 200
    assert response["status"] == "success"
    data = response["data"]
    assert data["counts"]["pairs"] == expected_pairs
    assert data["counts"]["underlyings"] == expected_underlyings
    assert data["counts"]["symbols"] == expected_symbols
    assert len(data["pairs"]) == expected_pairs
    assert len(data["symbols"]) == expected_symbols


def test_default_exchanges():
    """No argument -> NFO + MCX (documented default)."""
    _assert_ok(_call(), expected_pairs=5, expected_underlyings=3, expected_symbols=8)


def test_exchanges_none():
    """Explicit None must not raise TypeError (regression for the 500)."""
    _assert_ok(_call(exchanges=None), expected_pairs=5, expected_underlyings=3, expected_symbols=8)


def test_exchanges_empty_list():
    """Empty list means 'no preference' -> default exchanges."""
    _assert_ok(_call(exchanges=[]), expected_pairs=5, expected_underlyings=3, expected_symbols=8)


def test_nfo_only():
    _assert_ok(_call(exchanges=["NFO"]), expected_pairs=3, expected_underlyings=2, expected_symbols=5)


def test_mcx_only():
    _assert_ok(_call(exchanges=["MCX"]), expected_pairs=2, expected_underlyings=1, expected_symbols=3)


def test_nfo_plus_mcx():
    _assert_ok(_call(exchanges=["NFO", "MCX"]), expected_pairs=5, expected_underlyings=3, expected_symbols=8)


def test_lowercase_exchanges():
    """Exchange codes are case-normalised."""
    _assert_ok(_call(exchanges=["nfo"]), expected_pairs=3, expected_underlyings=2, expected_symbols=5)


def test_invalid_exchange_only():
    """No supported exchange -> structured 400, not a crash."""
    success, response, status = _call(exchanges=["NASDAQ"])
    assert success is False
    assert status == 400
    assert response["status"] == "error"
    assert "No supported exchanges" in response["message"]


def test_mixed_valid_and_invalid_exchanges():
    """Unsupported codes are ignored; supported ones still scanned."""
    _assert_ok(
        _call(exchanges=["NFO", "NASDAQ"]), expected_pairs=3, expected_underlyings=2, expected_symbols=5
    )


def test_leg_shape():
    """Each pair carries near/far legs with the compact contract shape."""
    _, response, _ = _call(exchanges=["NFO"])
    pair = response["data"]["pairs"][0]
    assert pair["id"].startswith("NFO:")
    assert pair["type"] in ("near-next", "near-third")
    for leg in (pair["near"], pair["far"]):
        assert set(leg) == {"symbol", "exchange", "expiry", "lotsize", "tick_size"}
