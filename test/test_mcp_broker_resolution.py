"""Regression tests for MCP broker resolution (P3/P7).

Verifies that _get_broker() reports the broker that actually executes
orders (the auth database) instead of a fabricated VALID_BROKERS[0]
value, and that explicit configuration (OPENALGO_BROKER env var or MCP
session context) still takes precedence.
"""

import pytest

from utils.mcp_tool_registry import _load_mcpserver_module

mcpserver = _load_mcpserver_module()
if mcpserver is None:
    pytest.fail("Failed to load mcp/mcpserver.py via _load_mcpserver_module()")


@pytest.fixture(autouse=True)
def _clear_broker_state():
    """Each test starts with an empty probe cache and session context."""
    mcpserver._PROBE_CACHE.clear()
    mcpserver._SESSION_CONTEXT["broker"] = None
    yield
    mcpserver._PROBE_CACHE.clear()
    mcpserver._SESSION_CONTEXT["broker"] = None


def test_db_broker_wins_over_valid_brokers_first_entry(monkeypatch):
    """Without env override, the auth DB broker is reported, not VALID_BROKERS[0]."""
    monkeypatch.delenv("OPENALGO_BROKER", raising=False)
    monkeypatch.setenv("VALID_BROKERS", "fivepaisa,fivepaisaxts,upstox")
    monkeypatch.setattr(
        "database.auth_db.get_active_broker_name", lambda: "upstox"
    )
    assert mcpserver._get_broker() == "upstox"


def test_env_override_wins_over_db(monkeypatch):
    """OPENALGO_BROKER is explicit configuration and beats the DB."""
    monkeypatch.setenv("OPENALGO_BROKER", "zerodha")
    monkeypatch.setenv("VALID_BROKERS", "fivepaisa,upstox")
    monkeypatch.setattr(
        "database.auth_db.get_active_broker_name", lambda: "upstox"
    )
    assert mcpserver._get_broker() == "zerodha"


def test_session_context_wins_over_everything():
    """An MCP client's set_session_context override always wins."""
    mcpserver._SESSION_CONTEXT["broker"] = "dhan"
    assert mcpserver._get_broker() == "dhan"


def test_no_db_broker_reports_unknown(monkeypatch):
    """No session, no env, no DB row -> 'unknown', never a fabricated broker."""
    monkeypatch.delenv("OPENALGO_BROKER", raising=False)
    monkeypatch.setenv("VALID_BROKERS", "fivepaisa,fivepaisaxts,upstox")
    monkeypatch.setattr("database.auth_db.get_active_broker_name", lambda: "")
    assert mcpserver._get_broker() == "unknown"


def test_db_failure_never_leaks_valid_brokers_first_entry(monkeypatch):
    """Even when the DB lookup errors, VALID_BROKERS[0] must not be reported."""
    monkeypatch.delenv("OPENALGO_BROKER", raising=False)
    monkeypatch.setenv("VALID_BROKERS", "fivepaisa,upstox")

    def boom():
        raise RuntimeError("db unavailable")

    monkeypatch.setattr("database.auth_db.get_active_broker_name", boom)
    assert mcpserver._get_broker() == "unknown"
