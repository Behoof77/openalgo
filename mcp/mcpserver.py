import functools
import json
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from mcp.server.fastmcp import FastMCP
from openalgo import api, ta

# Two boot paths share this module:
#
# 1. Stdio (legacy / local) — Claude Desktop, Cursor, Windsurf spawn this
#    file as `python -m mcp.mcpserver <api_key> <host>`. argv[1] and argv[2]
#    must be present. The original MCP integration lives here unchanged.
#
# 2. HTTP / SSE — blueprints/mcp_http.py imports this module to access the
#    `mcp` FastMCP instance (and every @mcp.tool decorated function), then
#    calls init_for_http(api_key, host) once per process to wire the SDK
#    client. The Flask app sets OPENALGO_MCP_HTTP_BOOT=1 *before* importing
#    so the argv check is bypassed.
#
# The branching is at module scope rather than inside a function so the
# FastMCP `mcp = FastMCP(...)` instance and every `@mcp.tool` decorator
# remain top-level (FastMCP relies on import-time registration).

if os.environ.get("OPENALGO_MCP_HTTP_BOOT") == "1":
    # HTTP transport — Flask sets the env var before import. The SDK
    # client is wired by init_for_http() right after import. Stdio
    # users never hit this branch.
    api_key: str | None = None
    host: str | None = None
    client = None
else:
    # === Original stdio behavior — preserved verbatim ===
    # Existing Claude Desktop / Cursor / Windsurf integrations launch
    # this file as `python -m mcp.mcpserver <api_key> <host>` and rely
    # on this exact error path on misconfiguration. Do NOT change the
    # check, the order, or the error message here.
    if len(sys.argv) < 3:
        raise ValueError("API key and host must be provided as command line arguments")

    api_key = sys.argv[1]
    host = sys.argv[2]

    # Initialize OpenAlgo client with provided arguments
    client = api(api_key=api_key, host=host)


def init_for_http(api_key_value: str, host_value: str) -> None:
    """Wire the SDK client when running under the HTTP transport.

    Called once from blueprints/mcp_http.py after the Flask app has
    determined the admin's API key and the local OpenAlgo loopback URL.
    Idempotent — safe to call repeatedly with the same values; later
    calls overwrite the global so a restarted broker session can rotate
    the underlying SDK client without restarting Gunicorn.
    """
    global api_key, host, client
    api_key = api_key_value
    host = host_value
    client = api(api_key=api_key_value, host=host_value)

# Default strategy name for all order-related calls originating from the MCP server.
# Surfaced in OpenAlgo logs and analyzer views so MCP-driven trades are identifiable.
MCP_STRATEGY = "python mcp"

# OpenAlgo standardized index symbols (NSE_INDEX / BSE_INDEX) — rolled out across all brokers.
# Source: https://docs.openalgo.in/symbol-format
NSE_INDEX_SYMBOLS = [
    "NIFTY", "NIFTYNXT50", "FINNIFTY", "BANKNIFTY", "MIDCPNIFTY", "INDIAVIX",
    "HANGSENGBEESNAV",
    "NIFTY100", "NIFTY200", "NIFTY500",
    "NIFTYALPHA50", "NIFTYAUTO", "NIFTYCOMMODITIES", "NIFTYCONSUMPTION",
    "NIFTYCPSE", "NIFTYDIVOPPS50", "NIFTYENERGY", "NIFTYFMCG",
    "NIFTYGROWSECT15",
    "NIFTYGS10YR", "NIFTYGS10YRCLN", "NIFTYGS1115YR", "NIFTYGS15YRPLUS",
    "NIFTYGS48YR", "NIFTYGS813YR", "NIFTYGSCOMPSITE",
    "NIFTYINFRA", "NIFTYIT", "NIFTYMEDIA", "NIFTYMETAL",
    "NIFTYMIDLIQ15", "NIFTYMIDCAP100", "NIFTYMIDCAP150", "NIFTYMIDCAP50",
    "NIFTYMIDSML400", "NIFTYMNC", "NIFTYPHARMA", "NIFTYPSE", "NIFTYPSUBANK",
    "NIFTYPVTBANK", "NIFTYREALTY", "NIFTYSERVSECTOR",
    "NIFTYSMLCAP100", "NIFTYSMLCAP250", "NIFTYSMLCAP50",
    "NIFTY100EQLWGT", "NIFTY100LIQ15", "NIFTY100LOWVOL30",
    "NIFTY100QUALTY30", "NIFTY200QUALTY30",
    "NIFTY50DIVPOINT", "NIFTY50EQLWGT",
    "NIFTY50PR1XINV", "NIFTY50PR2XLEV", "NIFTY50TR1XINV", "NIFTY50TR2XLEV",
    "NIFTY50VALUE20",
]
BSE_INDEX_SYMBOLS = [
    "SENSEX", "BANKEX", "SENSEX50",
    "BSE100", "BSE150MIDCAPINDEX", "BSE200", "BSE250LARGEMIDCAPINDEX",
    "BSE400MIDSMALLCAPINDEX", "BSE500",
    "BSEAUTO", "BSECAPITALGOODS", "BSECARBONEX", "BSECONSUMERDURABLES",
    "BSECPSE", "BSEDOLLEX100", "BSEDOLLEX200", "BSEDOLLEX30",
    "BSEENERGY", "BSEFASTMOVINGCONSUMERGOODS", "BSEFINANCIALSERVICES",
    "BSEGREENEX", "BSEHEALTHCARE", "BSEINDIAINFRASTRUCTUREINDEX",
    "BSEINDUSTRIALS", "BSEINFORMATIONTECHNOLOGY", "BSEIPO",
    "BSELARGECAP", "BSEMETAL", "BSEMIDCAP", "BSEMIDCAPSELECTINDEX",
    "BSEOIL&GAS", "BSEPOWER", "BSEPSU", "BSEREALTY", "BSESENSEXNEXT50",
    "BSESMALLCAP", "BSESMALLCAPSELECTINDEX", "BSESMEIPO",
    "BSETECK", "BSETELECOM",
]

# Create MCP server
mcp = FastMCP("openalgo")

# ---------------------------------------------------------------------------
# Response envelope foundation (Phase 1.5 hardening)
#
# Every @mcp.tool decorated function is wrapped below so ALL tools (existing
# and future) return the same standardized envelope:
#   success envelope:  {success, timestamp, backend_version, mcp_version,
#                       broker, exchange, latency_ms, market_status, data}
#   error envelope:    {success:false, error_code, message, retryable,
#                       timestamp, backend_version, mcp_version, broker,
#                       exchange, latency_ms, market_status}
# ---------------------------------------------------------------------------

MCP_VERSION = "1.5.0"  # MCP surface contract version (bump on interface change)

_START_TIME = time.monotonic()

# Lightweight session context shared across tool calls (single-worker process).
_SESSION_CONTEXT: dict[str, Any] = {
    "broker": None,
    "exchange": None,
    "preferred_expiry": None,
    "watchlist": None,
    "portfolio": None,
    "current_market": None,
}

# Lazy probe caches: key -> (timestamp, value). TTLs in seconds.
_PROBE_CACHE: dict[str, tuple[float, Any]] = {}
_PROBE_TTL = 600


def _ist_now_iso() -> str:
    """Return the current IST timestamp as an ISO string (seconds precision)."""
    return datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds")


def _normalize_symbol(symbol: str) -> str:
    """Normalize a symbol to the canonical OpenAlgo format.

    Accepts plain (NIFTY), option (NIFTY11AUG2624500CE) and futures
    (NIFTY11AUG26FUT) forms. Returns the uppercased, space-free symbol.
    Raises ValueError for empty or unrecognized symbols.
    """
    if not symbol or not isinstance(symbol, str):
        raise ValueError("Symbol is required and must be a non-empty string")
    cleaned = re.sub(r"\s+", "", symbol).upper()
    if not cleaned:
        raise ValueError("Symbol must be a non-empty string")
    if not re.match(r"^[A-Z0-9]+$", cleaned):
        raise ValueError(
            f"Unrecognized symbol format: {symbol}. Use plain (NIFTY), "
            "option (NIFTY11AUG2624500CE) or futures (NIFTY11AUG26FUT) form."
        )
    return cleaned


def _map_error_code(text: str) -> tuple[str, bool]:
    """Map an error message/status to a standardized (error_code, retryable)."""
    lowered = text.lower()
    if any(k in lowered for k in ("invalid api key", "unauthorized", "401")):
        return "UNAUTHORIZED", False
    if any(k in lowered for k in ("rate limit", "rate_limit", "429", "too many")):
        return "RATE_LIMITED", True
    if any(k in lowered for k in ("not found", "404")):
        return "NOT_FOUND", False
    if any(k in lowered for k in ("forbidden", "403")):
        return "FORBIDDEN", False
    if any(k in lowered for k in ("timeout", "timed out")):
        return "TIMEOUT", True
    if any(k in lowered for k in ("error calling", "connection", "network")):
        return "NETWORK_ERROR", True
    if any(k in lowered for k in ("internal server", "500", "backend", "traceback")):
        return "BACKEND_ERROR", True
    if any(k in lowered for k in ("validation", "invalid", "400", "required")):
        return "INVALID_REQUEST", False
    return "UNKNOWN_ERROR", False


def _get_market_status() -> str:
    """Derive market status from IST clock (no network call).

    OPEN between 09:15 and 15:30 IST on weekdays, CLOSED otherwise.
    """
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    if now.weekday() >= 5:
        return "CLOSED"
    seconds = now.hour * 3600 + now.minute * 60 + now.second
    if 9 * 3600 + 15 * 60 <= seconds <= 15 * 3600 + 30 * 60:
        return "OPEN"
    return "CLOSED"


def _get_backend_version() -> str:
    """Return the backend platform version (lazy, cached TTL)."""
    now = time.monotonic()
    cached = _PROBE_CACHE.get("backend_version")
    if cached and now - cached[0] < _PROBE_TTL:
        return cached[1]
    version = "unknown"
    try:
        from openalgo import __version__

        version = str(__version__)
    except Exception:
        pass
    _PROBE_CACHE["backend_version"] = (now, version)
    return version


def _get_broker() -> str:
    """Return the active broker (session context -> env -> lazy probe -> unknown)."""
    if _SESSION_CONTEXT.get("broker"):
        return _SESSION_CONTEXT["broker"]
    now = time.monotonic()
    cached = _PROBE_CACHE.get("broker")
    if cached and now - cached[0] < _PROBE_TTL:
        return cached[1]
    broker = os.getenv("OPENALGO_BROKER", "")
    if not broker:
        brokers = os.getenv("VALID_BROKERS", "")
        broker = brokers.split(",")[0].strip() if brokers else ""
    broker = broker or "unknown"
    _PROBE_CACHE["broker"] = (now, broker)
    return broker


def _build_envelope(
    data: Any, exchange: str | None = None, latency_ms: float | None = None
) -> str:
    """Wrap tool output in the standardized success envelope."""
    return json.dumps(
        {
            "success": True,
            "timestamp": _ist_now_iso(),
            "backend_version": _get_backend_version(),
            "mcp_version": MCP_VERSION,
            "broker": _get_broker(),
            "exchange": exchange,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            "market_status": _get_market_status(),
            "data": data,
        },
        indent=2,
        default=str,
    )


def _build_error(
    error_code: str,
    message: str,
    retryable: bool = False,
    exchange: str | None = None,
    latency_ms: float | None = None,
) -> str:
    """Build the standardized error envelope."""
    return json.dumps(
        {
            "success": False,
            "error_code": error_code,
            "message": message,
            "retryable": retryable,
            "timestamp": _ist_now_iso(),
            "backend_version": _get_backend_version(),
            "mcp_version": MCP_VERSION,
            "broker": _get_broker(),
            "exchange": exchange,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else None,
            "market_status": _get_market_status(),
        },
        indent=2,
        default=str,
    )


# Result cache for expensive read tools. Keys are (tool_name, args_json,
# kwargs_json); values are (expiry_ts, envelope_json). Production runs a
# single eventlet worker, so no locking is needed. TTLs mirror each tool's
# cache_ttl docstring; tools absent from the map never cache.
_TOOL_CACHE: dict[tuple, tuple[float, str]] = {}
_TOOL_CACHE_TTL: dict[str, float] = {
    "market_snapshot": 30.0,
    "option_snapshot": 30.0,
    "company_snapshot": 60.0,
    "portfolio_snapshot": 30.0,
    "position_snapshot": 30.0,
    "analyze": 30.0,
    "analyze_market": 30.0,
    "system_health": 30.0,
    "get_capabilities": 300.0,
    "get_gex_data": 30.0,
    "get_iv_smile_data": 30.0,
    "get_oi_data": 30.0,
    "calculate_max_pain": 30.0,
    "get_oi_profile_data": 30.0,
    "get_straddle_chart_data": 30.0,
    "get_vol_surface_data": 60.0,
    "get_iv_chart_data": 30.0,
    "get_gamma_density_data": 60.0,
    "get_multi_strike_oi_data": 60.0,
    "get_custom_straddle_simulation": 60.0,
    "get_default_symbols": 300.0,
    "get_support_resistance": 30.0,
    "get_trend_snapshot": 30.0,
    "get_momentum_snapshot": 30.0,
    "get_volatility_snapshot": 30.0,
    "screen_instruments": 60.0,
    "multi_timeframe_analysis": 60.0,
    "correlation_beta": 60.0,
    "calculate_indicator": 30.0,
    "get_historical_data": 60.0,
}


def _cache_key(fn_name: str, args, kwargs) -> tuple | None:
    """Canonical cache key for a tool call, or None when unhashable."""
    try:
        arg_part = json.dumps(args, default=str)
        kw_part = json.dumps(kwargs, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return None
    return (fn_name, arg_part, kw_part)


def _tool_wrapper(fn):
    """Wrap a tool function so its output becomes a standardized envelope."""

    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        exchange = kwargs.get("exchange") or _SESSION_CONTEXT.get("exchange")
        ttl = _TOOL_CACHE_TTL.get(fn.__name__, 0.0)
        key = None
        if ttl > 0:
            key = _cache_key(fn.__name__, args, kwargs)
            if key is not None:
                cached = _TOOL_CACHE.get(key)
                if cached is not None and cached[0] > time.monotonic():
                    return cached[1]
        try:
            result = fn(*args, **kwargs)
        except Exception as e:
            code, retryable = _map_error_code(str(e))
            return _build_error(
                code, str(e), retryable=retryable, exchange=exchange,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        latency_ms = (time.perf_counter() - start) * 1000
        if not isinstance(result, str):
            result = json.dumps(result, indent=2, default=str)
        stripped = result.lstrip()
        if stripped.startswith("Error"):
            code, retryable = _map_error_code(stripped)
            return _build_error(
                code, stripped, retryable=retryable, exchange=exchange,
                latency_ms=latency_ms,
            )
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return _build_envelope(
                {"message": stripped}, exchange=exchange, latency_ms=latency_ms
            )
        if isinstance(parsed, dict) and parsed.get("status") == "error":
            message = str(parsed.get("message", "Operation failed"))
            code, retryable = _map_error_code(message)
            return _build_error(
                code, message, retryable=retryable, exchange=exchange,
                latency_ms=latency_ms,
            )
        envelope = _build_envelope(
            parsed, exchange=exchange, latency_ms=latency_ms
        )
        if key is not None:
            if len(_TOOL_CACHE) >= 2000:
                _TOOL_CACHE.clear()
            _TOOL_CACHE[key] = (time.monotonic() + ttl, envelope)
        return envelope

    return wrapped


# Monkeypatch mcp.tool so every @mcp.tool() decorated function inherits the
# envelope wrapper. FastMCP's tool() only supports the @mcp.tool() call style;
# it returns a decorator, so we must invoke _original_tool(*args, **kwargs)
# first and then apply the returned decorator to the wrapped function.
_original_tool = mcp.tool


def _mcp_tool_wrapper(fn=None, *args, **kwargs):
    if fn is not None:
        raise TypeError(
            "The @tool decorator was used incorrectly. Did you forget to call it? Use @tool() instead of @tool"
        )

    def decorate(func):
        return _original_tool(*args, **kwargs)(_tool_wrapper(func))

    return decorate


mcp.tool = _mcp_tool_wrapper


def _to_json(payload: Any) -> str:
    """Serialize any SDK response (dict, list, or pandas DataFrame) to a JSON string."""
    if hasattr(payload, "to_dict") and hasattr(payload, "reset_index"):
        df = payload.reset_index()
        return json.dumps(
            {"count": len(df), "data": df.to_dict(orient="records")},
            indent=2,
            default=str,
        )
    return json.dumps(payload, indent=2, default=str)

# ORDER MANAGEMENT TOOLS


@mcp.tool()
def place_order(
    symbol: str,
    quantity: int,
    action: str,
    exchange: str = "NSE",
    price_type: str = "MARKET",
    product: str = "MIS",
    strategy: str = MCP_STRATEGY,
    price: float | None = None,
    trigger_price: float | None = None,
    disclosed_quantity: int | None = None,
) -> str:
    """
    Place a new order (market or limit).

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE')
        quantity: Number of shares
        action: 'BUY' or 'SELL'
        exchange: 'NSE', 'NFO', 'CDS', 'BSE', 'BFO', 'BCD', 'MCX', 'NCDEX'
        price_type: 'MARKET', 'LIMIT', 'SL', 'SL-M'
        product: 'CNC', 'NRML', 'MIS'
        strategy: Strategy name (defaults to 'python mcp')
        price: Limit price (required for LIMIT orders)
        trigger_price: Trigger price (required for SL and SL-M orders)
        disclosed_quantity: Disclosed quantity
    """
    try:
        params = {
            "strategy": strategy,
            "symbol": symbol.upper(),
            "action": action.upper(),
            "exchange": exchange.upper(),
            "price_type": price_type.upper(),
            "product": product.upper(),
            "quantity": quantity,
        }

        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if disclosed_quantity is not None:
            params["disclosed_quantity"] = disclosed_quantity

        response = client.placeorder(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing order: {str(e)}"


@mcp.tool()
def place_smart_order(
    symbol: str,
    quantity: int,
    action: str,
    position_size: int,
    exchange: str = "NSE",
    price_type: str = "MARKET",
    product: str = "MIS",
    strategy: str = MCP_STRATEGY,
    price: float | None = None,
    trigger_price: float | None = None,
    disclosed_quantity: int | None = None,
) -> str:
    """
    Place a smart order that considers the current position size (auto-calculates delta
    between requested and current size before sending to the broker).

    Args:
        symbol: Stock symbol
        quantity: Target quantity
        action: 'BUY' or 'SELL'
        position_size: Current position size
        exchange: Exchange name
        price_type: 'MARKET', 'LIMIT', 'SL', 'SL-M'
        product: 'CNC', 'NRML', 'MIS'
        strategy: Strategy name (defaults to 'python mcp')
        price: Limit price (required for LIMIT orders)
        trigger_price: Trigger price (required for SL / SL-M orders)
        disclosed_quantity: Disclosed quantity
    """
    try:
        params = {
            "strategy": strategy,
            "symbol": symbol.upper(),
            "action": action.upper(),
            "exchange": exchange.upper(),
            "price_type": price_type.upper(),
            "product": product.upper(),
            "quantity": quantity,
            "position_size": position_size,
        }

        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if disclosed_quantity is not None:
            params["disclosed_quantity"] = disclosed_quantity

        response = client.placesmartorder(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing smart order: {str(e)}"


@mcp.tool()
def place_basket_order(orders: list[dict[str, Any]], strategy: str = MCP_STRATEGY) -> str:
    """
    Place multiple orders in a basket.

    Args:
        orders: List of order dictionaries. Each order should contain:
            - symbol (str): Trading symbol. Required.
            - exchange (str): Exchange code. Required.
            - action (str): BUY or SELL. Required.
            - quantity (int/str): Quantity to trade. Required.
            - pricetype (str): MARKET, LIMIT, SL, SL-M. Optional, defaults to MARKET.
            - product (str): MIS, CNC, NRML. Optional, defaults to MIS.
            - price (str): Required for LIMIT orders.
            - trigger_price (str): Required for SL orders.
        strategy: Strategy name (default: Python)

        Example: [
            {"symbol": "BHEL", "exchange": "NSE", "action": "BUY", "quantity": 1, "pricetype": "MARKET", "product": "MIS"},
            {"symbol": "ZOMATO", "exchange": "NSE", "action": "SELL", "quantity": 1, "pricetype": "MARKET", "product": "MIS"}
        ]

    Returns:
        JSON with results for each order including orderid, status, and symbol
    """
    try:
        response = client.basketorder(strategy=strategy, orders=orders)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing basket order: {str(e)}"


@mcp.tool()
def place_split_order(
    symbol: str,
    quantity: int,
    split_size: int,
    action: str,
    exchange: str = "NSE",
    price_type: str = "MARKET",
    product: str = "MIS",
    strategy: str = MCP_STRATEGY,
    price: float | None = None,
    trigger_price: float | None = None,
    disclosed_quantity: int | None = None,
) -> str:
    """
    Place a large order split into smaller chunks.

    Args:
        symbol: Stock symbol (e.g., 'YESBANK')
        quantity: Total quantity to trade
        split_size: Size of each split order
        action: 'BUY' or 'SELL'
        exchange: Exchange name (default: NSE)
        price_type: 'MARKET', 'LIMIT', 'SL', 'SL-M' (default: MARKET)
        product: 'MIS', 'CNC', 'NRML' (default: MIS)
        strategy: Strategy name (default: Python)
        price: Limit price (required for LIMIT orders)
        trigger_price: Trigger price (required for SL orders)
        disclosed_quantity: Disclosed quantity (optional)

    Returns:
        JSON with results array containing each split order's orderid, quantity, and status

    Example:
        # Split 105 shares into orders of 20 each (5 orders of 20 + 1 order of 5)
        place_split_order("YESBANK", 105, 20, "SELL", "NSE")
    """
    try:
        params = {
            "strategy": strategy,
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "splitsize": split_size,
            "price_type": price_type.upper(),
            "product": product.upper(),
        }

        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if disclosed_quantity is not None:
            params["disclosed_quantity"] = disclosed_quantity

        response = client.splitorder(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing split order: {str(e)}"


@mcp.tool()
def place_options_order(
    underlying: str,
    exchange: str,
    offset: str,
    option_type: str,
    action: str,
    quantity: int,
    expiry_date: str | None = None,
    strategy: str = MCP_STRATEGY,
    price_type: str = "MARKET",
    product: str = "MIS",
    price: float | None = None,
    trigger_price: float | None = None,
    disclosed_quantity: int | None = None,
) -> str:
    """
    Place an options order with ATM/ITM/OTM offset.

    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY', 'NIFTY28OCT25FUT')
        exchange: Exchange for underlying ('NSE_INDEX', 'BSE_INDEX', 'NFO')
        offset: Strike offset - 'ATM', 'ITM1'-'ITM50', 'OTM1'-'OTM50'
        option_type: 'CE' for Call or 'PE' for Put
        action: 'BUY' or 'SELL'
        quantity: Absolute quantity — must be a multiple of the contract lot size.
                  Do NOT hardcode lot size — call get_option_symbol() or get_option_chain()
                  first to read the current 'lotsize' from the broker master contract,
                  then pass quantity = lots * lotsize.
        expiry_date: Expiry date in format 'DDMMMYY' (e.g., '28OCT25'). Optional if underlying includes expiry.
        strategy: Strategy name (default: Python)
        price_type: 'MARKET', 'LIMIT', 'SL', 'SL-M' (default: MARKET)
        product: 'MIS', 'NRML' (default: MIS). Note: CNC not supported for options.
        price: Limit price (required for LIMIT orders)
        trigger_price: Trigger price (required for SL and SL-M orders)
        disclosed_quantity: Disclosed quantity (optional)

    Returns:
        JSON with orderid, symbol, underlying_ltp, offset, option_type, mode

    Example:
        # Basic ATM call order
        place_options_order("NIFTY", "NSE_INDEX", "ATM", "CE", "BUY", 75, "28NOV25")

        # Using future as underlying (expiry auto-detected)
        place_options_order("NIFTY28OCT25FUT", "NFO", "ITM2", "CE", "BUY", 75)
    """
    try:
        params = {
            "strategy": strategy,
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "offset": offset.upper(),
            "option_type": option_type.upper(),
            "action": action.upper(),
            "quantity": quantity,
            "price_type": price_type.upper(),
            "product": product.upper(),
        }

        if expiry_date is not None:
            params["expiry_date"] = expiry_date
        if price is not None:
            params["price"] = price
        if trigger_price is not None:
            params["trigger_price"] = trigger_price
        if disclosed_quantity is not None:
            params["disclosed_quantity"] = disclosed_quantity

        response = client.optionsorder(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing options order: {str(e)}"


@mcp.tool()
def place_options_multi_order(
    underlying: str,
    exchange: str,
    legs: list[dict[str, Any]],
    expiry_date: str | None = None,
    strategy: str = MCP_STRATEGY,
) -> str:
    """
    Place a multi-leg options order (spreads, iron condor, straddles, etc.).
    BUY legs are executed first for margin efficiency, then SELL legs.

    Args:
        strategy: Strategy name (defaults to 'python mcp'). Give each multi-leg trade
                  a meaningful name (e.g., 'nifty iron condor') to make tracking easier.
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY', 'NIFTY28OCT25FUT')
        exchange: Exchange for underlying ('NSE_INDEX', 'BSE_INDEX', 'NFO')
        legs: List of leg dictionaries (1-20 legs). Each leg must contain:
            Required:
            - offset: Strike offset ('ATM', 'ITM1'-'ITM50', 'OTM1'-'OTM50')
            - option_type: 'CE' for Call or 'PE' for Put
            - action: 'BUY' or 'SELL'
            - quantity: Absolute quantity — must be a multiple of the contract lot size.
                        Do NOT hardcode lot size. Look up the current 'lotsize' per leg
                        using get_option_symbol() or get_option_chain() first, then pass
                        quantity = lots * lotsize. Lot sizes can change (e.g., NIFTY has
                        changed multiple times) and differ by underlying.
            Optional:
            - expiry_date: Per-leg expiry in DDMMMYY format for diagonal/calendar spreads
            - pricetype: 'MARKET', 'LIMIT', 'SL', 'SL-M' (default: MARKET)
            - product: 'MIS', 'NRML' (default: MIS)
            - price: Limit price for LIMIT orders
            - trigger_price: Trigger price for SL orders
            - disclosed_quantity: Disclosed quantity
        expiry_date: Default expiry date in format 'DDMMMYY' (e.g., '25NOV25') for all legs

    Returns:
        JSON with underlying, underlying_ltp, mode, and results array containing each leg's
        orderid, symbol, offset, option_type, action, and status

    Example - Iron Condor (same expiry):
        [
            {"offset": "OTM10", "option_type": "CE", "action": "BUY", "quantity": 75},
            {"offset": "OTM10", "option_type": "PE", "action": "BUY", "quantity": 75},
            {"offset": "OTM5", "option_type": "CE", "action": "SELL", "quantity": 75},
            {"offset": "OTM5", "option_type": "PE", "action": "SELL", "quantity": 75}
        ]

    Example - Bull Call Spread with NRML:
        [
            {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": 75, "product": "NRML"},
            {"offset": "OTM1", "option_type": "CE", "action": "SELL", "quantity": 75, "product": "NRML"}
        ]

    Example - Diagonal Spread (different expiry):
        [
            {"offset": "ITM2", "option_type": "CE", "action": "BUY", "quantity": 75, "expiry_date": "30DEC25"},
            {"offset": "OTM2", "option_type": "CE", "action": "SELL", "quantity": 75, "expiry_date": "25NOV25"}
        ]

    Example - Long Straddle with LIMIT orders:
        [
            {"offset": "ATM", "option_type": "CE", "action": "BUY", "quantity": 30, "pricetype": "LIMIT", "price": 250.0},
            {"offset": "ATM", "option_type": "PE", "action": "BUY", "quantity": 30, "pricetype": "LIMIT", "price": 250.0}
        ]
    """
    try:
        params = {
            "strategy": strategy,
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "legs": legs,
        }

        if expiry_date is not None:
            params["expiry_date"] = expiry_date

        response = client.optionsmultiorder(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error placing options multi order: {str(e)}"


@mcp.tool()
def modify_order(
    order_id: str,
    symbol: str,
    action: str,
    exchange: str,
    product: str,
    quantity: int,
    price: float,
    strategy: str = MCP_STRATEGY,
    price_type: str = "LIMIT",
    trigger_price: float = 0,
    disclosed_quantity: int = 0,
) -> str:
    """
    Modify an existing order.

    Args:
        order_id: Order ID to modify
        symbol: Stock symbol
        action: 'BUY' or 'SELL'
        exchange: Exchange name
        product: 'CNC', 'NRML', 'MIS'
        quantity: New quantity
        price: New price (required by the API — use current price if unchanged)
        strategy: Strategy name (defaults to 'python mcp')
        price_type: 'MARKET', 'LIMIT', 'SL', 'SL-M' (defaults to 'LIMIT')
        trigger_price: New trigger price for SL/SL-M orders (default 0)
        disclosed_quantity: New disclosed quantity (default 0)
    """
    try:
        response = client.modifyorder(
            order_id=order_id,
            strategy=strategy,
            symbol=symbol.upper(),
            action=action.upper(),
            exchange=exchange.upper(),
            price_type=price_type.upper(),
            product=product.upper(),
            quantity=quantity,
            price=price,
            trigger_price=trigger_price,
            disclosed_quantity=disclosed_quantity,
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error modifying order: {str(e)}"


@mcp.tool()
def cancel_order(order_id: str, strategy: str = MCP_STRATEGY) -> str:
    """
    Cancel a specific order.

    Args:
        order_id: Order ID to cancel
        strategy: Strategy name (defaults to 'python mcp')
    """
    try:
        response = client.cancelorder(order_id=order_id, strategy=strategy)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error canceling order: {str(e)}"


@mcp.tool()
def cancel_all_orders(strategy: str = MCP_STRATEGY) -> str:
    """
    Cancel all open orders for a strategy.

    Args:
        strategy: Strategy name (defaults to 'python mcp')
    """
    try:
        response = client.cancelallorder(strategy=strategy)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error canceling all orders: {str(e)}"


# POSITION MANAGEMENT TOOLS


@mcp.tool()
def close_all_positions(strategy: str = MCP_STRATEGY) -> str:
    """
    Close all open positions for a strategy.

    Args:
        strategy: Strategy name (defaults to 'python mcp')
    """
    try:
        response = client.closeposition(strategy=strategy)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error closing positions: {str(e)}"


@mcp.tool()
def get_open_position(
    symbol: str, exchange: str, product: str, strategy: str = MCP_STRATEGY
) -> str:
    """
    Get current open position for a specific instrument.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        product: Product type ('CNC', 'NRML', 'MIS')
        strategy: Strategy name (defaults to 'python mcp')
    """
    try:
        response = client.openposition(
            strategy=strategy,
            symbol=symbol.upper(),
            exchange=exchange.upper(),
            product=product.upper(),
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting open position: {str(e)}"


# ORDER STATUS AND TRACKING TOOLS


@mcp.tool()
def get_order_status(order_id: str, strategy: str = MCP_STRATEGY) -> str:
    """
    Get status of a specific order.

    Args:
        order_id: Order ID
        strategy: Strategy name (defaults to 'python mcp')
    """
    try:
        response = client.orderstatus(order_id=order_id, strategy=strategy)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting order status: {str(e)}"


@mcp.tool()
def get_order_book() -> str:
    """Get all orders from the order book."""
    try:
        response = client.orderbook()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting order book: {str(e)}"


@mcp.tool()
def get_trade_book() -> str:
    """Get all executed trades."""
    try:
        response = client.tradebook()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting trade book: {str(e)}"


@mcp.tool()
def get_position_book() -> str:
    """Get all current positions."""
    try:
        response = client.positionbook()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting position book: {str(e)}"


@mcp.tool()
def get_holdings() -> str:
    """Get all holdings (long-term investments)."""
    try:
        response = client.holdings()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting holdings: {str(e)}"


@mcp.tool()
def get_funds() -> str:
    """Get account funds and margin information."""
    try:
        response = client.funds()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting funds: {str(e)}"


@mcp.tool()
def calculate_margin(positions: list[dict[str, Any]]) -> str:
    """
    Calculate margin requirements for positions.

    Args:
        positions: List of position dictionaries
        Example: [{"symbol": "NIFTY25NOV2525000CE", "exchange": "NFO", "action": "BUY", "product": "NRML", "pricetype": "MARKET", "quantity": "75"}]

        For Futures: [{"symbol": "NIFTY25NOV25FUT", "exchange": "NFO", "action": "BUY", "product": "NRML", "pricetype": "MARKET", "quantity": "25"}]
        For Options: [{"symbol": "NIFTY25NOV2525500CE", "exchange": "NFO", "action": "BUY", "product": "NRML", "pricetype": "MARKET", "quantity": "75"}]

    Returns:
        JSON with total_margin_required, span_margin, and exposure_margin
    """
    try:
        response = client.margin(positions=positions)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error calculating margin: {str(e)}"


# MARKET DATA TOOLS


@mcp.tool()
def get_quote(symbol: str, exchange: str = "NSE") -> str:
    """
    Get current quote for a symbol.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
    """
    try:
        response = client.quotes(symbol=symbol.upper(), exchange=exchange.upper())
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting quote: {str(e)}"


@mcp.tool()
def get_multi_quotes(symbols: list[dict[str, str]]) -> str:
    """
    Get real-time quotes for multiple symbols in a single request.

    Args:
        symbols: List of symbol-exchange pairs
        Example: [{"symbol": "RELIANCE", "exchange": "NSE"}, {"symbol": "INFY", "exchange": "NSE"}]

    Returns:
        JSON with quotes for all requested symbols including ltp, bid, ask, open, high, low, volume, oi
    """
    try:
        # Normalize symbols to uppercase
        normalized_symbols = [
            {"symbol": s["symbol"].upper(), "exchange": s["exchange"].upper()} for s in symbols
        ]
        response = client.multiquotes(symbols=normalized_symbols)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting multi quotes: {str(e)}"


@mcp.tool()
def get_option_chain(
    underlying: str,
    exchange: str,
    expiry_date: str | None = None,
    strike_count: int | None = None,
) -> str:
    """
    Get option chain data with real-time quotes for all strikes.

    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY', 'RELIANCE',
                    or a future like 'NIFTY30DEC25FUT')
        exchange: Exchange for underlying ('NSE_INDEX', 'BSE_INDEX', 'NSE', 'BSE', 'NFO', 'BFO')
        expiry_date: Expiry date in DDMMMYY format (e.g., '30DEC25'). Optional when the
                     underlying already includes an expiry (e.g., 'NIFTY30DEC25FUT').
        strike_count: Number of strikes above and below ATM (1-100). If not provided, returns entire chain.

    Returns:
        JSON with:
        - underlying: Base symbol
        - underlying_ltp: Current price of underlying
        - expiry_date: Expiry date
        - atm_strike: At-The-Money strike price
        - chain: Array of strikes with CE and PE data including:
            - symbol, label (ATM/ITM1/OTM1 etc.), ltp, bid, ask, open, high, low, volume, oi, lotsize

    Note: CE and PE have different labels at the same strike:
        - Strikes below ATM: CE is ITM, PE is OTM
        - Strikes above ATM: CE is OTM, PE is ITM

    Example for 10 strikes around ATM:
        get_option_chain("NIFTY", "NSE_INDEX", "30DEC25", 10)

    Example for full chain:
        get_option_chain("NIFTY", "NSE_INDEX", "30DEC25")
    """
    try:
        params: dict[str, Any] = {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
        }
        if expiry_date is not None:
            params["expiry_date"] = expiry_date.upper()
        if strike_count is not None:
            params["strike_count"] = strike_count

        response = client.optionchain(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting option chain: {str(e)}"


@mcp.tool()
def get_market_depth(symbol: str, exchange: str = "NSE") -> str:
    """
    Get market depth (order book) for a symbol.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
    """
    try:
        response = client.depth(symbol=symbol.upper(), exchange=exchange.upper())
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting market depth: {str(e)}"


@mcp.tool()
def get_historical_data(
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
    source: str = "api",
    bars: int = 20,
    lookback_days: int | None = None,
) -> str:
    """
    Get historical OHLCV data for a symbol.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Time interval. With source='api': '1m', '3m', '5m', '10m', '15m', '30m', '1h', 'D'.
                  With source='db': also supports custom intervals (2m, 4m, 6m, 7m, 2h, 3h, 4h) and
                  daily-based (W, M, Q, Y plus multiples like 2W, 3M).
        start_date: Start date (YYYY-MM-DD). Optional — when omitted, the last `bars`
                    (default 20) most-recent bars are returned (or `lookback_days` if given).
        end_date: End date (YYYY-MM-DD). Optional — defaults to today.
        source: 'api' (default) fetches from broker API. 'db' fetches from the local
                OpenAlgo Historify DuckDB store (1m/D stored, other intervals computed via SQL).
        bars: Number of most-recent bars to return (default 20). The window is fetched
              server-side; only the last `bars` rows are sent back to keep the payload small.
              Increase only if you explicitly need more rows.
        lookback_days: When dates are omitted, fetch the last N calendar days instead of a
                       bar-count window (e.g., 30 for "last 30 days").

    Returns:
        JSON with total count, returned count, a truncated flag, and data (list of
        {timestamp, open, high, low, close, volume}) — the last `bars` rows.
    """
    try:
        # Fetch enough to satisfy `bars` (min 252) unless an explicit range/lookback is given.
        response = _load_history(
            symbol, exchange, interval, start_date, end_date, max(252, bars), lookback_days, source
        )
        total = len(response)
        return json.dumps(
            {
                "count": total,
                "returned": min(bars, total),
                "truncated": total > bars,
                "bars": bars,
                "data": _df_records(response, bars),
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error getting historical data: {str(e)}"


# INSTRUMENT SEARCH AND INFO TOOLS


@mcp.tool()
def search_instruments(
    query: str, exchange: str | None = None, instrument_type: str | None = None
) -> str:
    """
    Search for instruments by name or symbol.

    Args:
        query: Search query (e.g., 'NIFTY 26000 DEC CE', 'RELIANCE')
        exchange: Exchange to restrict the search to (NSE, BSE, NFO, BFO, MCX, NSE_INDEX, etc.).
                  Optional — when omitted, searches across all exchanges.
        instrument_type: Optional convenience filter — pass 'INDEX' to auto-rewrite
                         exchange=NSE → NSE_INDEX and BSE → BSE_INDEX.
    """
    try:
        resolved_exchange = exchange
        if instrument_type and instrument_type.upper() == "INDEX" and exchange:
            if exchange.upper() == "NSE":
                resolved_exchange = "NSE_INDEX"
            elif exchange.upper() == "BSE":
                resolved_exchange = "BSE_INDEX"
        if resolved_exchange is not None:
            response = client.search(query=query, exchange=resolved_exchange.upper())
        else:
            response = client.search(query=query)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error searching instruments: {str(e)}"


@mcp.tool()
def get_symbol_info(symbol: str, exchange: str = "NSE", instrument_type: str = None) -> str:
    """
    Get detailed information about a symbol.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        instrument_type: Optional - 'INDEX' for index symbols
    """
    try:
        # Handle index symbols
        if instrument_type and instrument_type.upper() == "INDEX":
            if exchange.upper() == "NSE":
                exchange = "NSE_INDEX"
            elif exchange.upper() == "BSE":
                exchange = "BSE_INDEX"

        # Or auto-route to the _INDEX exchange if the symbol is a known index.
        if symbol.upper() in NSE_INDEX_SYMBOLS and exchange.upper() == "NSE":
            exchange = "NSE_INDEX"
        elif symbol.upper() in BSE_INDEX_SYMBOLS and exchange.upper() == "BSE":
            exchange = "BSE_INDEX"

        response = client.symbol(symbol=symbol.upper(), exchange=exchange.upper())
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting symbol info: {str(e)}"


@mcp.tool()
def get_index_symbols(exchange: str = "NSE") -> str:
    """
    Get the OpenAlgo-standardized index symbols for NSE or BSE.

    These are the common index names rolled out across all supported brokers via the
    OpenAlgo symbol standardization. Use exchange code 'NSE_INDEX' / 'BSE_INDEX' when
    placing orders or fetching quotes for these symbols.

    Args:
        exchange: NSE or BSE

    Returns:
        JSON with exchange, exchange_code, and the full list of standardized index
        symbols (57+ NSE, 40+ BSE).
    """
    indices = {
        "NSE": {"exchange_code": "NSE_INDEX", "symbols": NSE_INDEX_SYMBOLS},
        "BSE": {"exchange_code": "BSE_INDEX", "symbols": BSE_INDEX_SYMBOLS},
    }

    exchange_upper = exchange.upper()
    if exchange_upper in indices:
        return json.dumps(
            {
                "exchange": exchange_upper,
                "exchange_code": indices[exchange_upper]["exchange_code"],
                "indices": indices[exchange_upper]["symbols"],
            },
            indent=2,
        )
    else:
        return json.dumps({"error": f"Unknown exchange: {exchange}. Use NSE or BSE."}, indent=2)


@mcp.tool()
def get_expiry_dates(symbol: str, exchange: str = "NFO", instrument_type: str = "options") -> str:
    """
    Get expiry dates for derivatives.

    Args:
        symbol: Underlying symbol
        exchange: Exchange name (typically NFO for F&O)
        instrument_type: 'options' or 'futures'
    """
    try:
        response = client.expiry(
            symbol=symbol.upper(), exchange=exchange.upper(), instrumenttype=instrument_type.lower()
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting expiry dates: {str(e)}"


@mcp.tool()
def get_available_intervals() -> str:
    """Get all available time intervals for historical data."""
    try:
        response = client.intervals()
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting intervals: {str(e)}"


@mcp.tool()
def get_option_symbol(
    underlying: str,
    exchange: str,
    offset: str,
    option_type: str,
    expiry_date: str | None = None,
) -> str:
    """
    Get option symbol for specific strike and expiry.

    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY', 'NIFTY28OCT25FUT')
        exchange: Exchange for underlying ('NSE_INDEX', 'BSE_INDEX', 'NFO', 'BFO')
        offset: Strike offset - 'ATM', 'ITM1'-'ITM50', 'OTM1'-'OTM50'
        option_type: 'CE' for Call or 'PE' for Put
        expiry_date: Expiry date in 'DDMMMYY' format (e.g., '28OCT25'). Optional when
                     the underlying already includes an expiry.

    Returns:
        JSON with symbol, exchange, lotsize, tick_size, underlying_ltp
    """
    try:
        params: dict[str, Any] = {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "offset": offset.upper(),
            "option_type": option_type.upper(),
        }
        if expiry_date is not None:
            params["expiry_date"] = expiry_date
        response = client.optionsymbol(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error getting option symbol: {str(e)}"


@mcp.tool()
def get_synthetic_future(underlying: str, exchange: str, expiry_date: str) -> str:
    """
    Calculate synthetic future price using put-call parity.

    Args:
        underlying: Underlying symbol (e.g., 'NIFTY', 'BANKNIFTY')
        exchange: Exchange for underlying ('NSE_INDEX', 'BSE_INDEX')
        expiry_date: Expiry date in format 'DDMMMYY' (e.g., '25NOV25')

    Returns:
        JSON with atm_strike, expiry, status, synthetic_future_price, underlying, underlying_ltp
    """
    try:
        response = client.syntheticfuture(
            underlying=underlying.upper(), exchange=exchange.upper(), expiry_date=expiry_date
        )
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error calculating synthetic future: {str(e)}"


@mcp.tool()
def get_option_greeks(
    symbol: str,
    exchange: str,
    interest_rate: float | None = None,
    forward_price: float | None = None,
    underlying_symbol: str | None = None,
    underlying_exchange: str | None = None,
    expiry_time: str | None = None,
) -> str:
    """
    Calculate option Greeks (Delta, Gamma, Theta, Vega, Rho) and Implied Volatility using Black-76.

    Args:
        symbol: Option symbol (e.g., 'NIFTY25NOV2526000CE'). Required.
        exchange: Exchange code ('NFO', 'BFO', 'CDS', 'MCX'). Required.
        interest_rate: Risk-free interest rate in annualized % (e.g., 6.5 for RBI repo).
                       Optional — defaults to 0.
        forward_price: Custom forward / synthetic futures price. If provided, skips the
                       underlying price fetch. Useful for illiquid underlyings (FINNIFTY,
                       MIDCPNIFTY) or custom scenario analysis.
        underlying_symbol: Custom underlying symbol (e.g., 'NIFTY', 'NIFTY30DEC25FUT').
                           Optional — auto-detected from the option symbol when omitted.
        underlying_exchange: Custom underlying exchange ('NSE_INDEX', 'NFO', etc.).
                             Optional — auto-detected when omitted.
        expiry_time: Custom expiry time in HH:MM format (e.g., '19:00'). Required for
                     MCX contracts with non-standard expiry times. Exchange defaults:
                     NFO/BFO=15:30, CDS=12:30, MCX=23:30.

    Returns:
        JSON with greeks, implied_volatility, spot_price, strike, days_to_expiry.
    """
    try:
        params: dict[str, Any] = {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
        }
        if interest_rate is not None:
            params["interest_rate"] = interest_rate
        if forward_price is not None:
            params["forward_price"] = forward_price
        if underlying_symbol is not None:
            params["underlying_symbol"] = underlying_symbol.upper()
        if underlying_exchange is not None:
            params["underlying_exchange"] = underlying_exchange.upper()
        if expiry_time is not None:
            params["expiry_time"] = expiry_time
        response = client.optiongreeks(**params)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error calculating option greeks: {str(e)}"


# UTILITY TOOLS


@mcp.tool()
def get_openalgo_version() -> str:
    """Get the OpenAlgo library version."""
    try:
        import openalgo

        return f"OpenAlgo version: {openalgo.__version__}"
    except Exception as e:
        return f"Error getting version: {str(e)}"


@mcp.tool()
def validate_order_constants() -> str:
    """Display all valid order constants for reference."""
    constants = {
        "exchanges": {
            "NSE": "NSE Equity",
            "NFO": "NSE Futures & Options",
            "CDS": "NSE Currency",
            "BSE": "BSE Equity",
            "BFO": "BSE Futures & Options",
            "BCD": "BSE Currency",
            "MCX": "MCX Commodity",
            "NCDEX": "NCDEX Commodity",
        },
        "product_types": {
            "CNC": "Cash & Carry for equity",
            "NRML": "Normal for futures and options",
            "MIS": "Intraday Square off",
        },
        "price_types": {
            "MARKET": "Market Order",
            "LIMIT": "Limit Order",
            "SL": "Stop Loss Limit Order",
            "SL-M": "Stop Loss Market Order",
        },
        "actions": {"BUY": "Buy", "SELL": "Sell"},
        "intervals": ["1m", "3m", "5m", "10m", "15m", "30m", "1h", "D"],
    }
    return json.dumps(constants, indent=2)


@mcp.tool()
def send_telegram_alert(username: str, message: str, priority: int = 5) -> str:
    """
    Send a Telegram alert notification.

    Args:
        username: OpenAlgo login ID/username
        message: Alert message to send
        priority: Notification priority (1-10, default 5). Higher values may be used
                  by the bot for emphasis/sorting depending on configuration.

    Returns:
        JSON with status and message
    """
    try:
        response = client.telegram(username=username, message=message, priority=priority)
        return json.dumps(response, indent=2)
    except Exception as e:
        return f"Error sending telegram alert: {str(e)}"


@mcp.tool()
def get_holidays(year: int | None = None) -> str:
    """
    Get trading holidays for a specific year.

    Args:
        year: Year to get holidays for (e.g., 2026). Optional — defaults to current year.

    Returns:
        JSON with list of trading holidays including:
        - date: Holiday date (YYYY-MM-DD)
        - description: Holiday name/reason
        - holiday_type: TRADING_HOLIDAY, SETTLEMENT_HOLIDAY, or SPECIAL_SESSION
        - closed_exchanges: List of closed exchanges
        - open_exchanges: List of exchanges with special timings

    Example:
        get_holidays(2026)
        get_holidays()          # current year
    """
    try:
        response = client.holidays(year=year) if year is not None else client.holidays()
        return json.dumps(response, indent=2, default=str)
    except Exception as e:
        return f"Error getting holidays: {str(e)}"


@mcp.tool()
def get_timings(date: str | None = None) -> str:
    """
    Get exchange trading timings for a specific date.

    Args:
        date: Date in YYYY-MM-DD format (e.g., '2026-04-23'). Optional — defaults to today.

    Returns:
        JSON with exchange timings including:
        - exchange: Exchange name (NSE, BSE, NFO, BFO, MCX, CDS, BCD)
        - start_time: Market open time in epoch milliseconds
        - end_time: Market close time in epoch milliseconds

    Example:
        get_timings("2026-04-23")
        get_timings()           # today
    """
    try:
        response = client.timings(date=date) if date is not None else client.timings()
        return json.dumps(response, indent=2, default=str)
    except Exception as e:
        return f"Error getting timings: {str(e)}"


@mcp.tool()
def check_holiday(date: str, exchange: str | None = None) -> str:
    """
    Check if a specific date is a market holiday for an exchange.

    This calls the /api/v1/checkholiday endpoint directly (not yet in the openalgo SDK).
    Use this for fast pre-trade "is the market open?" checks.

    Args:
        date: Date in YYYY-MM-DD format (between 2020-01-01 and 2050-12-31). Required.
        exchange: Exchange code (NSE, BSE, NFO, BFO, MCX, CDS, BCD). Optional.
                  When omitted, returns true if the date is a holiday for any major exchange.

    Returns:
        JSON with:
        - status: 'success' or 'error'
        - data.date, data.exchange (if specified), data.is_holiday (bool)

    Notes:
        - Weekends and national holidays both return is_holiday=true.
        - For a full calendar, use get_holidays(year).

    Examples:
        check_holiday("2026-01-26", "NSE")
        check_holiday("2026-01-27")
    """
    try:
        url = f"{host.rstrip('/')}/api/v1/checkholiday"
        payload: dict[str, Any] = {"apikey": api_key, "date": date}
        if exchange:
            payload["exchange"] = exchange.upper()
        with httpx.Client(timeout=30.0) as http:
            r = http.post(url, json=payload, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error checking holiday: {str(e)}"


@mcp.tool()
def get_instruments(exchange: str | None = None, limit: int = 500) -> str:
    """
    Download the full instrument master.

    Args:
        exchange: Exchange name (NSE, BSE, NFO, BFO, MCX, CDS, BCD, NSE_INDEX, BSE_INDEX).
                  Optional — when omitted, downloads instruments for ALL exchanges.
        limit: Maximum number of rows to return in the response (default: 500).
               The full dataset can exceed 100k rows for derivatives exchanges, which
               overwhelms the MCP tool output. Use search_instruments for targeted lookups.

    Returns:
        JSON with count, returned, truncated flag, and data (list of instrument records).
        Each record includes: symbol, brsymbol, name, exchange, lotsize,
        instrumenttype, expiry, strike, token, tick_size.
    """
    try:
        response = (
            client.instruments(exchange=exchange.upper())
            if exchange is not None
            else client.instruments()
        )
        # SDK returns a DataFrame on success, dict on error
        if hasattr(response, "reset_index"):
            total = len(response)
            df_head = response.head(limit).reset_index(drop=True)
            return json.dumps(
                {
                    "exchange": exchange.upper() if exchange else "ALL",
                    "count": total,
                    "returned": len(df_head),
                    "truncated": total > limit,
                    "limit": limit,
                    "data": df_head.to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        return json.dumps(response, indent=2, default=str)
    except Exception as e:
        return f"Error getting instruments: {str(e)}"


# Tool to get analyzer status
@mcp.tool()
def analyzer_status() -> str:
    """
    Get the current analyzer status including mode and total logs.

    Returns:
        JSON with analyzer status information:
        - data.analyze_mode: Boolean indicating if analyzer is active
        - data.mode: Current mode ('analyze' or 'live')
        - data.total_logs: Number of logs in analyzer
        - status: 'success' or 'error'
    """
    try:
        response = client.analyzerstatus()
        return json.dumps(response, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# Tool to toggle analyzer mode
@mcp.tool()
def analyzer_toggle(mode: bool) -> str:
    """
    Toggle the analyzer mode between analyze (simulated) and live trading.

    Args:
        mode: True for analyze mode (simulated), False for live mode

    Returns:
        JSON with updated analyzer status:
        - data.analyze_mode, data.message, data.mode, data.total_logs
        - status: 'success' or 'error'

    Example:
        analyzer_toggle(True)  # Switch to analyze mode (simulated responses)
        analyzer_toggle(False) # Switch to live trading mode
    """
    try:
        response = client.analyzertoggle(mode=mode)
        return json.dumps(response, indent=2, default=str)
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, indent=2)


# ============================================================
# RESEARCH TOOLS — TECHNICAL INDICATORS (openalgo.ta)
# ============================================================
# These tools fetch OHLCV history via the SDK (client.history) and
# compute indicators with `from openalgo import ta`. They are SDK-only
# and work under BOTH the stdio and HTTP transports.

# Indicators whose first inputs are High/Low/Close (and optionally
# Volume) rather than a single Close series. Used to auto-pick inputs
# in calculate_indicator() when the caller does not pass `inputs`.
_HLC_INDICATORS = {
    "atr", "natr", "true_range", "adx", "adxr", "dmi", "dx", "supertrend",
    "stochastic", "stochf", "cci", "williams_r", "keltner", "donchian",
    "aroon", "aroon_oscillator", "psar", "ichimoku", "pivot_points",
    "ultimate_oscillator", "uo_oscillator", "chandelier_exit", "starc",
    "elderray", "ckstop", "fractals", "rwi", "alligator", "gator_oscillator",
    "bop", "rvi", "fisher", "avgprice", "medprice", "midprice", "typprice",
    "wclprice",
}
_HLCV_INDICATORS = {"mfi", "cmf", "adl", "emv", "klingervolumeoscillator"}


def _history_df(
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str,
    end_date: str,
    source: str = "api",
):
    """Fetch OHLCV history as a timestamp-indexed DataFrame. Raises on failure.

    source: 'api' (default) fetches from the broker API; 'db' fetches from the local
    OpenAlgo Historify DuckDB store (1m/D stored, other intervals computed via SQL,
    enabling custom intervals like 2m/4m/W/M/Q for research).
    """
    df = client.history(
        symbol=symbol.upper(),
        exchange=exchange.upper(),
        interval=interval,
        start_date=start_date,
        end_date=end_date,
        source=source,
    )
    # SDK returns a DataFrame on success, a dict on error.
    if not hasattr(df, "reset_index"):
        raise ValueError(f"history error: {df}")
    if len(df) == 0:
        raise ValueError("no historical data returned for the given range")
    return df


# Approx bars per trading day per interval (NSE ~6h15m session). Used only to size
# the calendar window when fetching by bar-count, so it can be a rough estimate.
_BARS_PER_DAY = {
    "1m": 375, "3m": 125, "5m": 75, "10m": 38, "15m": 25, "30m": 13,
    "1h": 7, "60m": 7, "2h": 4, "3h": 3, "4h": 2,
    "d": 1, "day": 1, "w": 0.2, "week": 0.2, "m": 0.05, "month": 0.05,
}


def _load_history(
    symbol: str,
    exchange: str,
    interval: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
):
    """Fetch OHLCV history with a flexible lookback window.

    Resolution priority:
      1. Explicit start_date (with optional end_date) -> use that range verbatim.
      2. lookback_days given -> last N calendar days ending today (e.g., "last 30 days").
      3. else -> last `lookback_bars` bars (default 252 ≈ one trading year of daily data):
         fetch a wide-enough calendar window, then tail to exactly `lookback_bars` rows.
    """
    end = end_date or date.today().isoformat()
    if start_date:
        return _history_df(symbol, exchange, interval, start_date, end, source)
    if lookback_days:
        start = (date.fromisoformat(end) - timedelta(days=int(lookback_days))).isoformat()
        return _history_df(symbol, exchange, interval, start, end, source)
    bpd_per_day = _BARS_PER_DAY.get(interval.lower(), 75)
    cal_days = int((lookback_bars / bpd_per_day) * 1.6) + 5
    start = (date.fromisoformat(end) - timedelta(days=cal_days)).isoformat()
    df = _history_df(symbol, exchange, interval, start, end, source)
    return df.tail(int(lookback_bars))


def _idx_iso(df, pos: int) -> str:
    """ISO timestamp of the row at positional index `pos` (e.g., 0 first, -1 last)."""
    ts = df.index[pos]
    return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)


def _last(series) -> float | None:
    """Last non-null value of a Series-like as a rounded float, or None."""
    try:
        s = series.dropna()
        return round(float(s.iloc[-1]), 4) if len(s) else None
    except Exception:
        return None


def _df_records(df, limit: int | None = None):
    """Convert a DataFrame to JSON-safe records with an ISO 'timestamp' column."""
    out = df.tail(limit) if limit else df
    out = out.reset_index()
    out = out.rename(columns={out.columns[0]: "timestamp"})
    return json.loads(out.to_json(orient="records", date_format="iso"))


def _resolve_inputs(df, name: str, inputs: list[str] | None):
    """Pick the ordered input Series for an indicator (caller override or heuristic)."""
    if inputs:
        cols = [c.lower() for c in inputs]
    elif name in _HLCV_INDICATORS:
        cols = ["high", "low", "close", "volume"]
    elif name in _HLC_INDICATORS:
        cols = ["high", "low", "close"]
    else:
        cols = ["close"]
    return cols, [df[c] for c in cols]


def _bundle(df, specs: list[tuple]):
    """Compute latest values for a set of indicators, capturing per-item errors.

    specs: list of (key, callable) where callable returns a Series or tuple of Series.
    Tuple results become a list of latest values (see each tool's 'legend').
    """
    result: dict[str, Any] = {}
    for key, fn in specs:
        try:
            val = fn()
            result[key] = [_last(s) for s in val] if isinstance(val, tuple) else _last(val)
        except Exception as e:
            result[key] = {"error": str(e)}
    return result


def _as_bool(x, index) -> pd.Series:
    """Coerce an indicator boolean result (Series or array) to a clean bool Series."""
    s = pd.Series(list(x), index=index)
    return s.fillna(False).astype(bool)


@mcp.tool()
def calculate_indicator(
    symbol: str,
    exchange: str,
    indicator: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    params: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    bars: int = 20,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    Run ANY of the 80+ openalgo.ta indicators over a symbol's historical OHLCV.

    History is fetched (db/api) and the indicator is computed entirely on the
    OpenAlgo server; only compact results are returned — never the raw OHLCV.

    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'NIFTY')
        exchange: Exchange name (NSE, NFO, NSE_INDEX, etc.)
        indicator: ta function name, case-insensitive (e.g., 'rsi','macd','supertrend',
                   'atr','bbands','adx','ema','vwap').
        interval: '1m','3m','5m','10m','15m','30m','1h','D' (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — when omitted, a lookback window
                   ending today is used.
        params: Extra keyword args for the indicator (e.g., {"period": 14} for rsi;
                {"period": 10, "multiplier": 3} for supertrend;
                {"fast_period": 12, "slow_period": 26, "signal_period": 9} for macd).
        inputs: Ordered list of OHLCV columns to feed the indicator, e.g. ["close"] or
                ["high","low","close"]. Optional — auto-detected for common indicators;
                pass it explicitly if a result errors on inputs.
        bars: Number of most-recent computed rows to return (default 20). The indicator
              is ALWAYS computed server-side over the FULL fetched history; only the last
              `bars` rows (plus latest value and summary stats) are sent back, so the
              payload stays small. Increase only if you explicitly need more rows.
        lookback_bars: Bars of history to load/compute over when dates are omitted
                       (default 252 ≈ one trading year of daily data).
        lookback_days: Alternative calendar-day lookback (e.g., 30 for "last 30 days").
                       Overrides lookback_bars when set.
        source: 'api' (default, broker API) or 'db' (local Historify DuckDB store, which
                supports custom research intervals like 2m/4m/W/M/Q).

    Returns:
        JSON with the latest value(s), summary stats (last/min/max/mean), and a 'data'
        series of the last `bars` rows — all computed server-side. Multi-output indicators
        (macd, bbands, supertrend, stochastic, adx, ichimoku, keltner, donchian) report
        out0, out1, ...
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        name = indicator.lower()
        fn = getattr(ta, name, None)
        if fn is None:
            return json.dumps(
                {"status": "error", "message": f"unknown indicator '{indicator}'"}, indent=2
            )
        cols, args = _resolve_inputs(df, name, inputs)
        result = fn(*args, **(params or {}))
        out = pd.DataFrame(index=df.index)
        if isinstance(result, tuple):
            for i, s in enumerate(result):
                out[f"out{i}"] = pd.Series(list(s), index=df.index)
        else:
            out["value"] = pd.Series(list(result), index=df.index)

        def _stats(s):
            s = s.dropna()
            if not len(s):
                return None
            return {
                "last": round(float(s.iloc[-1]), 4),
                "min": round(float(s.min()), 4),
                "max": round(float(s.max()), 4),
                "mean": round(float(s.mean()), 4),
            }

        cols_out = list(out.columns)
        ts = out.index[-1]
        payload: dict[str, Any] = {
            "symbol": symbol.upper(),
            "exchange": exchange.upper(),
            "indicator": name,
            "inputs": cols,
            "params": params or {},
            "interval": interval,
            "source": source,
            "bars": len(out),
            "last_close": _last(df["close"]),
            "latest_timestamp": ts.isoformat() if hasattr(ts, "isoformat") else str(ts),
            "latest": {c: _last(out[c]) for c in cols_out},
            "summary": {c: _stats(out[c]) for c in cols_out},
            "returned_bars": min(bars, len(out)),
            "data": _df_records(out, bars),
        }
        return json.dumps(payload, indent=2, default=str)
    except Exception as e:
        return f"Error calculating indicator: {str(e)}"


@mcp.tool()
def get_trend_snapshot(
    symbol: str,
    exchange: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    One-call trend read: SMA(20/50/200), EMA(20/50), Supertrend, ADX/DMI, Ichimoku.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to a lookback window ending today.
        lookback_bars: Bars of history loaded when dates are omitted (default 252, enough for SMA200).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.

    Returns:
        JSON with latest indicator values and a 'legend' explaining multi-value entries.
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        snap = _bundle(
            df,
            [
                ("sma_20", lambda: ta.sma(df["close"], 20)),
                ("sma_50", lambda: ta.sma(df["close"], 50)),
                ("sma_200", lambda: ta.sma(df["close"], 200)),
                ("ema_20", lambda: ta.ema(df["close"], 20)),
                ("ema_50", lambda: ta.ema(df["close"], 50)),
                ("supertrend", lambda: ta.supertrend(df["high"], df["low"], df["close"])),
                ("adx_di", lambda: ta.adx(df["high"], df["low"], df["close"], period=14)),
                ("ichimoku", lambda: ta.ichimoku(df["high"], df["low"], df["close"])),
            ],
        )
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "interval": interval,
                "from": _idx_iso(df, 0),
                "to": _idx_iso(df, -1),
                "bars_loaded": len(df),
                "last_close": _last(df["close"]),
                "indicators": snap,
                "legend": {
                    "supertrend": "[supertrend_value, direction(+1 up / -1 down)]",
                    "adx_di": "[+DI, -DI, ADX]",
                    "ichimoku": "[tenkan, kijun, senkou_a, senkou_b, chikou]",
                },
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error getting trend snapshot: {str(e)}"


@mcp.tool()
def get_momentum_snapshot(
    symbol: str,
    exchange: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    One-call momentum read: RSI(14), MACD, Stochastic, CCI(20), Williams %R(14).

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to a lookback window ending today.
        lookback_bars: Bars of history loaded when dates are omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.

    Returns:
        JSON with latest values and a 'legend' for multi-value entries.
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        snap = _bundle(
            df,
            [
                ("rsi_14", lambda: ta.rsi(df["close"], 14)),
                ("macd", lambda: ta.macd(df["close"])),
                ("stochastic", lambda: ta.stochastic(df["high"], df["low"], df["close"])),
                ("cci_20", lambda: ta.cci(df["high"], df["low"], df["close"], 20)),
                ("williams_r_14", lambda: ta.williams_r(df["high"], df["low"], df["close"], 14)),
            ],
        )
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "interval": interval,
                "from": _idx_iso(df, 0),
                "to": _idx_iso(df, -1),
                "bars_loaded": len(df),
                "last_close": _last(df["close"]),
                "indicators": snap,
                "legend": {
                    "macd": "[macd_line, signal_line, histogram]",
                    "stochastic": "[%K, %D]",
                },
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error getting momentum snapshot: {str(e)}"


@mcp.tool()
def get_volatility_snapshot(
    symbol: str,
    exchange: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    One-call volatility read: ATR, NATR, Bollinger Bands (+%B, width), Keltner,
    Donchian, Historical Volatility.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to a lookback window ending today.
        lookback_bars: Bars of history loaded when dates are omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.

    Returns:
        JSON with latest values and a 'legend' for multi-value band entries.
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        snap = _bundle(
            df,
            [
                ("atr_14", lambda: ta.atr(df["high"], df["low"], df["close"], period=14)),
                ("natr_14", lambda: ta.natr(df["high"], df["low"], df["close"], period=14)),
                ("bbands", lambda: ta.bbands(df["close"], period=20, std_dev=2.0)),
                ("bb_percent_b", lambda: ta.bbpercent(df["close"], period=20, std_dev=2.0)),
                ("bb_width", lambda: ta.bbwidth(df["close"], period=20, std_dev=2.0)),
                ("keltner", lambda: ta.keltner(df["high"], df["low"], df["close"])),
                ("donchian", lambda: ta.donchian(df["high"], df["low"], period=20)),
                ("historical_volatility", lambda: ta.hv(df["close"])),
            ],
        )
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "interval": interval,
                "from": _idx_iso(df, 0),
                "to": _idx_iso(df, -1),
                "bars_loaded": len(df),
                "last_close": _last(df["close"]),
                "indicators": snap,
                "legend": {
                    "bbands": "[upper, middle, lower]",
                    "keltner": "[upper, middle, lower]",
                    "donchian": "[upper, middle, lower]",
                },
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error getting volatility snapshot: {str(e)}"


@mcp.tool()
def get_support_resistance(
    symbol: str,
    exchange: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    period: int = 20,
    source: str = "api",
) -> str:
    """
    Support/resistance levels: Pivot Points, Donchian channel, and rolling
    highest-high / lowest-low over `period`.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to a lookback window ending today.
        lookback_bars: Bars of history loaded when dates are omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.
        period: Lookback window for Donchian / highest / lowest (default 20).

    Returns:
        JSON with latest levels. 'pivot_points' is returned as ta.pivot_points emits it
        (typically [pivot, r1, s1, r2, s2, r3, s3]).
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        snap = _bundle(
            df,
            [
                ("donchian", lambda: ta.donchian(df["high"], df["low"], period=period)),
                ("highest_high", lambda: ta.highest(df["high"], period)),
                ("lowest_low", lambda: ta.lowest(df["low"], period)),
                ("pivot_points", lambda: ta.pivot_points(df["high"], df["low"], df["close"])),
            ],
        )
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "interval": interval,
                "from": _idx_iso(df, 0),
                "to": _idx_iso(df, -1),
                "bars_loaded": len(df),
                "period": period,
                "last_close": _last(df["close"]),
                "levels": snap,
                "legend": {"donchian": "[upper, middle, lower]"},
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error getting support/resistance: {str(e)}"


@mcp.tool()
def detect_signals(
    symbol: str,
    exchange: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    signal_type: str = "ema_cross",
    fast: int = 20,
    slow: int = 50,
    period: int = 14,
    upper: float = 70.0,
    lower: float = 30.0,
    limit: int = 20,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    Detect technical signals over a symbol's history using ta crossover/threshold logic.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to the lookback window.
        lookback_bars: Bars loaded when dates omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.
        signal_type: One of:
            'ema_cross'      - EMA(fast) crossing EMA(slow)
            'sma_cross'      - SMA(fast) crossing SMA(slow)
            'macd_cross'     - MACD line crossing its signal line
            'supertrend_flip'- Supertrend direction flip
            'rsi_threshold'  - RSI crossing out of oversold(lower) / overbought(upper)
        fast / slow: MA periods for ema_cross / sma_cross
        period: Lookback for rsi_threshold (default 14)
        upper / lower: RSI overbought / oversold levels (default 70 / 30)
        limit: Max number of most-recent signal events to return (default 20)

    Returns:
        JSON with recent events [{timestamp, signal: 'bullish'|'bearish'}] plus current values.
    """
    try:
        df = _load_history(
            symbol, exchange, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        close = df["close"]
        extra: dict[str, Any] = {}

        if signal_type in ("ema_cross", "sma_cross"):
            ma = ta.ema if signal_type == "ema_cross" else ta.sma
            f, s = ma(close, fast), ma(close, slow)
            bull = _as_bool(ta.crossover(f, s), df.index)
            bear = _as_bool(ta.crossunder(f, s), df.index)
            extra = {"fast": _last(f), "slow": _last(s)}
        elif signal_type == "macd_cross":
            line, sig, _hist = ta.macd(close)
            bull = _as_bool(ta.crossover(line, sig), df.index)
            bear = _as_bool(ta.crossunder(line, sig), df.index)
            extra = {"macd_line": _last(line), "signal_line": _last(sig)}
        elif signal_type == "supertrend_flip":
            st, d = ta.supertrend(df["high"], df["low"], close)
            d = pd.Series(list(d), index=df.index)
            bull = (d > 0) & (d.shift(1) <= 0)
            bear = (d < 0) & (d.shift(1) >= 0)
            bull, bear = bull.fillna(False), bear.fillna(False)
            extra = {"supertrend": _last(st), "direction": _last(d)}
        elif signal_type == "rsi_threshold":
            r = pd.Series(list(ta.rsi(close, period)), index=df.index)
            bull = (r > lower) & (r.shift(1) <= lower)
            bear = (r < upper) & (r.shift(1) >= upper)
            bull, bear = bull.fillna(False), bear.fillna(False)
            extra = {"rsi": _last(r)}
        else:
            return json.dumps(
                {"status": "error", "message": f"unknown signal_type '{signal_type}'"}, indent=2
            )

        def _stamp(ts):
            return ts.isoformat() if hasattr(ts, "isoformat") else str(ts)

        events = [{"timestamp": _stamp(ts), "signal": "bullish"} for ts, v in bull.items() if v]
        events += [{"timestamp": _stamp(ts), "signal": "bearish"} for ts, v in bear.items() if v]
        events.sort(key=lambda r: r["timestamp"])

        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "interval": interval,
                "signal_type": signal_type,
                "last_close": _last(close),
                "current": extra,
                "event_count": len(events),
                "events": events[-limit:],
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error detecting signals: {str(e)}"


@mcp.tool()
def screen_instruments(
    symbols: list[dict[str, str]],
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    condition: str = "rsi_below",
    value: float = 30.0,
    period: int = 14,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    Scan a watchlist of symbols for a technical condition.

    Note: this fetches history per symbol sequentially — keep the list modest (≤ ~25)
    or use a coarse interval to bound runtime and broker API calls.

    Args:
        symbols: List of {"symbol","exchange"} pairs.
            Example: [{"symbol":"RELIANCE","exchange":"NSE"},{"symbol":"INFY","exchange":"NSE"}]
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to the lookback window.
        lookback_bars: Bars loaded per symbol when dates omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.
        condition: One of:
            'rsi_below' / 'rsi_above'        - RSI(period) vs `value`
            'price_above_sma'/'price_below_sma' - last close vs SMA(period)
            'supertrend_bullish'/'supertrend_bearish' - current Supertrend direction
        value: Threshold for rsi conditions (default 30)
        period: Lookback for rsi / sma (default 14)

    Returns:
        JSON with per-symbol {passed, metric} and a count of matches.
    """
    try:
        results = []
        for item in symbols:
            sym, exch = item.get("symbol", ""), item.get("exchange", "")
            try:
                df = _load_history(
                    sym, exch, interval, start_date, end_date, lookback_bars, lookback_days, source
                )
                close = df["close"]
                metric: Any = None
                passed = False
                if condition in ("rsi_below", "rsi_above"):
                    metric = _last(ta.rsi(close, period))
                    if metric is not None:
                        passed = metric < value if condition == "rsi_below" else metric > value
                elif condition in ("price_above_sma", "price_below_sma"):
                    sma, c = _last(ta.sma(close, period)), _last(close)
                    metric = c
                    if sma is not None and c is not None:
                        passed = c > sma if condition == "price_above_sma" else c < sma
                elif condition in ("supertrend_bullish", "supertrend_bearish"):
                    _st, d = ta.supertrend(df["high"], df["low"], close)
                    metric = _last(pd.Series(list(d), index=df.index))
                    if metric is not None:
                        passed = metric > 0 if condition == "supertrend_bullish" else metric < 0
                else:
                    return json.dumps(
                        {"status": "error", "message": f"unknown condition '{condition}'"}, indent=2
                    )
                results.append(
                    {"symbol": sym.upper(), "exchange": exch.upper(), "passed": passed, "metric": metric}
                )
            except Exception as e:
                results.append({"symbol": sym, "exchange": exch, "error": str(e)})

        matched = [r for r in results if r.get("passed")]
        return json.dumps(
            {
                "condition": condition,
                "value": value,
                "period": period,
                "scanned": len(symbols),
                "matched": len(matched),
                "results": results,
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error screening instruments: {str(e)}"


@mcp.tool()
def multi_timeframe_analysis(
    symbol: str,
    exchange: str,
    start_date: str | None = None,
    end_date: str | None = None,
    intervals: list[str] | None = None,
    indicator: str = "rsi",
    params: dict[str, Any] | None = None,
    inputs: list[str] | None = None,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    Compute the same indicator across multiple timeframes for confluence analysis.

    Args:
        symbol: Stock symbol
        exchange: Exchange name
        start_date / end_date: YYYY-MM-DD. Optional — default to the lookback window per interval.
        intervals: List of intervals (default ['5m','15m','1h','D'])
        indicator: ta function name (default 'rsi')
        params: Extra keyword args for the indicator (e.g., {"period": 14})
        inputs: Ordered input columns; auto-detected if omitted.
        lookback_bars: Bars loaded per interval when dates omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.

    Returns:
        JSON with the latest indicator value (and last_close) per timeframe.
    """
    try:
        intervals = intervals or ["5m", "15m", "1h", "D"]
        name = indicator.lower()
        fn = getattr(ta, name, None)
        if fn is None:
            return json.dumps(
                {"status": "error", "message": f"unknown indicator '{indicator}'"}, indent=2
            )
        out: dict[str, Any] = {}
        for itv in intervals:
            try:
                df = _load_history(
                    symbol, exchange, itv, start_date, end_date, lookback_bars, lookback_days, source
                )
                _cols, args = _resolve_inputs(df, name, inputs)
                res = fn(*args, **(params or {}))
                value = [_last(s) for s in res] if isinstance(res, tuple) else _last(res)
                out[itv] = {"value": value, "last_close": _last(df["close"]), "bars": len(df)}
            except Exception as e:
                out[itv] = {"error": str(e)}
        return json.dumps(
            {
                "symbol": symbol.upper(),
                "exchange": exchange.upper(),
                "indicator": name,
                "params": params or {},
                "timeframes": out,
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error in multi-timeframe analysis: {str(e)}"


@mcp.tool()
def correlation_beta(
    symbol1: str,
    exchange1: str,
    symbol2: str,
    exchange2: str,
    interval: str = "D",
    start_date: str | None = None,
    end_date: str | None = None,
    period: int = 20,
    lookback_bars: int = 252,
    lookback_days: int | None = None,
    source: str = "api",
) -> str:
    """
    Correlation / Beta / Linear-regression slope between two symbols (pairs & hedge research).

    Both symbols' closes are aligned on common timestamps before computing.

    Args:
        symbol1 / exchange1: First instrument (the 'asset')
        symbol2 / exchange2: Second instrument (the 'market'/benchmark)
        interval: Candle interval (default 'D')
        start_date / end_date: YYYY-MM-DD. Optional — default to the lookback window.
        period: Rolling window for correlation/beta/slope (default 20)
        lookback_bars: Bars loaded per symbol when dates omitted (default 252).
        lookback_days: Alternative calendar-day lookback (e.g., 30). Overrides lookback_bars.

    Returns:
        JSON with rolling correlation, rolling beta, LR slope of symbol1, the full-sample
        Pearson correlation, and the number of overlapping bars.
    """
    try:
        df1 = _load_history(
            symbol1, exchange1, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        df2 = _load_history(
            symbol2, exchange2, interval, start_date, end_date, lookback_bars, lookback_days, source
        )
        j = pd.DataFrame({"a": df1["close"], "b": df2["close"]}).dropna()
        if len(j) < 2:
            return json.dumps(
                {"status": "error", "message": "insufficient overlapping bars between symbols"},
                indent=2,
            )
        p = min(period, len(j))
        metrics = _bundle(
            j,
            [
                ("correlation_rolling", lambda: ta.correlation(j["a"], j["b"], p)),
                ("beta_rolling", lambda: ta.beta(j["a"], j["b"], p)),
                ("lrslope_symbol1", lambda: ta.lrslope(j["a"], p)),
            ],
        )
        metrics["pearson_full_sample"] = round(float(j["a"].corr(j["b"])), 4)
        return json.dumps(
            {
                "symbol1": symbol1.upper(),
                "symbol2": symbol2.upper(),
                "interval": interval,
                "period": p,
                "overlapping_bars": len(j),
                "metrics": metrics,
            },
            indent=2,
            default=str,
        )
    except Exception as e:
        return f"Error calculating correlation/beta: {str(e)}"


def _post_api_v1(path: str, payload: dict[str, Any]) -> str:
    """POST an API-key-authenticated request to an /api/v1 endpoint.

    Used for analytics endpoints that have no SDK method (GEX, IV Smile,
    OI Tracker, OI Profile, Straddle Chart, Vol Surface, IV Chart,
    Custom Straddle). The platform REST layer resolves the API key from
    the JSON body and calls the same services the /tools pages use.

    Args:
        path: API endpoint path, e.g. "/gex" or "/oitracker/maxpain".
        payload: Request body fields (apikey is injected automatically).

    Returns:
        JSON string of the endpoint response, or an error message.
    """
    url = f"{host.rstrip('/')}/api/v1/{path.lstrip('/')}"
    body = {"apikey": api_key, **payload}
    try:
        with httpx.Client(timeout=30.0) as http:
            r = http.post(url, json=body, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error calling /api/v1{path}: {str(e)}"


def _put_api_v1(path: str, payload: dict[str, Any]) -> str:
    """PUT an API-key-authenticated request to an /api/v1 endpoint.

    Mirrors _post_api_v1 for update endpoints that use PUT.

    Args:
        path: API endpoint path, e.g. "/strategyportfolio/3".
        payload: Request body fields (apikey is injected automatically).

    Returns:
        JSON string of the endpoint response, or an error message.
    """
    url = f"{host.rstrip('/')}/api/v1/{path.lstrip('/')}"
    body = {"apikey": api_key, **payload}
    try:
        with httpx.Client(timeout=30.0) as http:
            r = http.put(url, json=body, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error calling /api/v1{path}: {str(e)}"


def _delete_api_v1(path: str) -> str:
    """DELETE an API-key-authenticated request to an /api/v1 endpoint.

    Mirrors _post_api_v1 for delete endpoints. The apikey travels in the
    JSON body so the platform REST layer can authenticate the request.

    Args:
        path: API endpoint path, e.g. "/strategy/3".

    Returns:
        JSON string of the endpoint response, or an error message.
    """
    url = f"{host.rstrip('/')}/api/v1/{path.lstrip('/')}"
    body = {"apikey": api_key}
    try:
        with httpx.Client(timeout=30.0) as http:
            r = http.delete(url, json=body, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error calling /api/v1{path}: {str(e)}"


def _post_webhook(webhook_id: str, payload: dict[str, Any]) -> str:
    """POST a webhook trigger for a strategy.

    The webhook route is keyed by the strategy webhook_id (not the API
    key), so no apikey is injected here.

    Args:
        webhook_id: Strategy webhook ID (UUID).
        payload: Webhook body as sent by the trading platform.

    Returns:
        JSON string of the endpoint response, or an error message.
    """
    url = f"{host.rstrip('/')}/strategy/webhook/{webhook_id}"
    try:
        with httpx.Client(timeout=30.0) as http:
            r = http.post(url, json=payload, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error calling webhook {webhook_id}: {str(e)}"


def _parse_json_response(response: str) -> dict[str, Any]:
    """Parse a JSON string response into a dict, or return an empty dict."""
    try:
        return json.loads(response)
    except Exception:
        return {}


def _normalize_expiry_display(expiry_display: str) -> str:
    """Convert an expiry display value (e.g., 11-AUG-26) to DDMMMYY (11AUG26)."""
    return expiry_display.upper().replace("-", "")


@mcp.tool()
def get_gex_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
) -> str:
    """
    Get Gamma Exposure (GEX) data for an underlying/expiry.

    Computes per-strike GEX from option chain OI and Black-76 greeks.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).

    Returns:
        JSON with spot/futures price, lot size, ATM strike, PCR OI, total
        CE/PE OI, total CE/PE GEX, net GEX, and the per-strike chain.
    """
    return _post_api_v1(
        "/gex",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
        },
    )


@mcp.tool()
def get_iv_smile_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
) -> str:
    """
    Get Implied Volatility (IV) Smile data for an underlying/expiry.

    Returns IV for all strikes so the smile curve can be plotted.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).

    Returns:
        JSON with the per-strike IV smile for calls and puts.
    """
    return _post_api_v1(
        "/ivsmile",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
        },
    )


@mcp.tool()
def get_oi_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
) -> str:
    """
    Get Open Interest (OI) data for all strikes of an expiry.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).

    Returns:
        JSON with per-strike CE/PE open interest, change in OI, and volume.
    """
    return _post_api_v1(
        "/oitracker",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
        },
    )


@mcp.tool()
def calculate_max_pain(
    underlying: str,
    exchange: str,
    expiry_date: str,
) -> str:
    """
    Calculate Max Pain for an underlying/expiry.

    Max Pain is the strike where option buyers lose the most money
    (total payout to option writers is minimized).

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).

    Returns:
        JSON with the max pain strike and per-strike total payout.
    """
    return _post_api_v1(
        "/oitracker/maxpain",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
        },
    )


@mcp.tool()
def get_oi_profile_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
    interval: str = "5m",
    days: int = 5,
) -> str:
    """
    Get OI Profile data with an intraday futures panel.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).
        interval: Candle interval for the futures panel (1m, 5m, 15m; default 5m).
        days: Number of days of history to load (default 5, max 30).

    Returns:
        JSON with per-strike OI profile and the underlying futures series.
    """
    return _post_api_v1(
        "/oiprofile",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
            "interval": interval,
            "days": days,
        },
    )


@mcp.tool()
def get_straddle_chart_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
    interval: str = "1m",
    days: int = 5,
) -> str:
    """
    Get Dynamic ATM Straddle chart data.

    Computes the per-candle ATM strike and straddle value (CE + PE) with
    the synthetic future price over the requested window.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).
        interval: Candle interval (default 1m).
        days: Number of days of history to load (default 5).

    Returns:
        JSON with the straddle time series and synthetic future values.
    """
    return _post_api_v1(
        "/straddle",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
            "interval": interval,
            "days": days,
        },
    )


@mcp.tool()
def get_vol_surface_data(
    underlying: str,
    exchange: str,
    expiry_dates: list[str],
    strike_count: int = 15,
) -> str:
    """
    Get 3D Volatility Surface data across expiries.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_dates: List of expiry dates in DDMMMYY format (e.g., 28NOV25),
                      max 8.
        strike_count: Number of strikes above and below ATM (default 15,
                      clamped between 5 and 40).

    Returns:
        JSON with per-expiry IV values at each strike.
    """
    return _post_api_v1(
        "/volsurface",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_dates": [expiry.upper() for expiry in expiry_dates],
            "strike_count": strike_count,
        },
    )


@mcp.tool()
def get_iv_chart_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
    interval: str = "5m",
    days: int = 1,
) -> str:
    """
    Get intraday Implied Volatility (IV) chart data.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).
        interval: Candle interval (default 5m).
        days: Number of days of history to load (default 1).

    Returns:
        JSON with the intraday IV time series for the ATM strike.
    """
    return _post_api_v1(
        "/ivchart",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
            "interval": interval,
            "days": days,
        },
    )


@mcp.tool()
def get_default_symbols(
    underlying: str,
    exchange: str,
    expiry_date: str,
) -> str:
    """
    Get default ATM option symbols for an underlying/expiry.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).

    Returns:
        JSON with the ATM call and put symbols.
    """
    return _post_api_v1(
        "/ivchart/default-symbols",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
        },
    )


@mcp.tool()
def get_custom_straddle_simulation(
    underlying: str,
    exchange: str,
    expiry_date: str,
    interval: str = "1m",
    days: int = 1,
    adjustment_points: int = 50,
    lot_size: int = 65,
    lots: int = 1,
) -> str:
    """
    Simulate a custom straddle with adjustment points and lot sizing.

    Models rolling the ATM straddle when the underlying moves by
    adjustment_points, and scales PnL by lot_size and lots.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).
        interval: Candle interval (default 1m).
        days: Number of days of history to load (default 1).
        adjustment_points: Straddle adjustment threshold in points (default 50).
        lot_size: Contract lot size for PnL scaling (default 65).
        lots: Number of lots to simulate (default 1).

    Returns:
        JSON with the simulated straddle PnL over the window.
    """
    return _post_api_v1(
        "/straddlepnl/simulate",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": expiry_date.upper(),
            "interval": interval,
            "days": days,
            "adjustment_points": adjustment_points,
            "lot_size": lot_size,
            "lots": lots,
        },
    )


@mcp.tool()
def get_gamma_density_data(
    underlying: str,
    exchange: str,
    expiry_date: str,
    interest_rate: float | None = None,
) -> str:
    """
    Get Gamma Density data for an underlying/expiry.

    Computes the per-strike gamma density curve, ATM/one/two sigma bands,
    and peak gamma strikes from the option chain and Black-76 greeks.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Expiry date in DDMMMYY format (e.g., 28NOV25).
        interest_rate: Optional risk-free interest rate override as a fraction (e.g., 0.065).

    Returns:
        JSON with spot/forward price, ATM IV, sigma bands and the gamma density chain.
    """
    payload: dict[str, Any] = {
        "underlying": underlying.upper(),
        "exchange": exchange.upper(),
        "expiry_date": expiry_date.upper(),
    }
    if interest_rate is not None:
        payload["interest_rate"] = interest_rate
    return _post_api_v1("/gammadensity", payload)


@mcp.tool()
def get_arbitrage_universe(
    exchanges: list[str] | None = None,
) -> str:
    """
    Get the arbitrage universe of near/far futures pairs.

    Lists futures pairs with the same underlying across nearby and far
    expiries, built from the master contract database.

    Args:
        exchanges: Optional list of exchanges to scan (NFO, MCX, BFO, CDS). Defaults to NFO and MCX.

    Returns:
        JSON with arbitrage pairs, symbols and pair counts per exchange.
    """
    payload: dict[str, Any] = {}
    if exchanges:
        payload["exchanges"] = [e.upper() for e in exchanges]
    return _post_api_v1("/arbitrage", payload)


@mcp.tool()
def get_multi_strike_oi_data(
    underlying: str,
    exchange: str,
    legs: list[dict],
    interval: str = "1m",
    days: int = 5,
) -> str:
    """
    Get Open Interest data for multiple option strikes of an underlying.

    Tracks OI history for each provided option leg over an intraday or
    multi-day window so option activity can be compared across strikes.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        legs: List of option legs, each with symbol, exchange, side, strike, optionType and expiry.
        interval: Candle interval (default 1m).
        days: Number of days of history to load (default 5).

    Returns:
        JSON with the underlying LTP, OI series and per-leg OI series.
    """
    return _post_api_v1(
        "/multistrikeoi",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "legs": legs,
            "interval": interval,
            "days": days,
        },
    )


@mcp.tool()
def create_strategy(
    platform: str,
    name: str,
    strategy_type: str = "intraday",
    trading_mode: str = "LONG",
    start_time: str | None = None,
    end_time: str | None = None,
    squareoff_time: str | None = None,
) -> str:
    """
    Create a new trading strategy.

    Registers a strategy that external platforms (TradingView, Chartink,
    etc.) can trigger via its webhook. Intraday strategies get a
    scheduled square-off at squareoff_time.

    Args:
        platform: Platform type (tradingview, chartink, etc.).
        name: Strategy name.
        strategy_type: intraday or positional (default intraday).
        trading_mode: LONG, SHORT or BOTH (default LONG).
        start_time: Entry window start in HH:MM 24h format.
        end_time: Entry window end in HH:MM 24h format.
        squareoff_time: Square-off time in HH:MM 24h format.

    Returns:
        JSON with the new strategy id and webhook id.
    """
    payload: dict[str, Any] = {
        "platform": platform,
        "name": name,
        "strategy_type": strategy_type,
        "trading_mode": trading_mode,
    }
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
    if squareoff_time:
        payload["squareoff_time"] = squareoff_time
    return _post_api_v1("/strategy", payload)


@mcp.tool()
def list_strategies() -> str:
    """
    List all strategies for the authenticated user.

    Returns:
        JSON array of strategies with webhook ids and trading windows.
    """
    return _post_api_v1("/strategy/list", {})


@mcp.tool()
def get_strategy(strategy_id: int) -> str:
    """
    Get a single strategy with its symbol mappings.

    Args:
        strategy_id: Numeric strategy id.

    Returns:
        JSON with the strategy details and its symbol mappings.
    """
    return _post_api_v1(f"/strategy/{strategy_id}", {})


@mcp.tool()
def toggle_strategy(strategy_id: int) -> str:
    """
    Toggle a strategy between active and inactive.

    Activating an intraday strategy re-schedules its square-off;
    deactivating removes the scheduled job.

    Args:
        strategy_id: Numeric strategy id.

    Returns:
        JSON with the new is_active state.
    """
    return _post_api_v1(f"/strategy/{strategy_id}/toggle", {})


@mcp.tool()
def delete_strategy(strategy_id: int) -> str:
    """
    Delete a strategy and its symbol mappings.

    Args:
        strategy_id: Numeric strategy id.

    Returns:
        JSON confirming deletion.
    """
    return _delete_api_v1(f"/strategy/{strategy_id}")


@mcp.tool()
def add_strategy_symbols(strategy_id: int, symbols: list[dict]) -> str:
    """
    Add symbol mappings to a strategy.

    Each mapping must carry symbol, exchange, quantity and product_type.

    Args:
        strategy_id: Numeric strategy id.
        symbols: List of dicts with symbol, exchange, quantity and product_type (MIS/CNC).

    Returns:
        JSON confirming the mappings were added.
    """
    return _post_api_v1(f"/strategy/{strategy_id}/symbols", {"symbols": symbols})


@mcp.tool()
def remove_strategy_symbol(strategy_id: int, mapping_id: int) -> str:
    """
    Remove a symbol mapping from a strategy.

    Args:
        strategy_id: Numeric strategy id.
        mapping_id: Numeric symbol mapping id.

    Returns:
        JSON confirming the mapping was removed.
    """
    return _delete_api_v1(f"/strategy/{strategy_id}/symbol/{mapping_id}")


@mcp.tool()
def update_strategy_times(
    strategy_id: int,
    start_time: str | None = None,
    end_time: str | None = None,
    squareoff_time: str | None = None,
) -> str:
    """
    Update the trading windows of a strategy.

    Args:
        strategy_id: Numeric strategy id.
        start_time: Entry window start in HH:MM 24h format.
        end_time: Entry window end in HH:MM 24h format.
        squareoff_time: Square-off time in HH:MM 24h format.

    Returns:
        JSON confirming the times were updated.
    """
    payload: dict[str, Any] = {}
    if start_time:
        payload["start_time"] = start_time
    if end_time:
        payload["end_time"] = end_time
    if squareoff_time:
        payload["squareoff_time"] = squareoff_time
    return _post_api_v1(f"/strategy/{strategy_id}/times", payload)


@mcp.tool()
def trigger_strategy_webhook(webhook_id: str, payload: dict) -> str:
    """
    Trigger a strategy webhook with a signal payload.

    Simulates a TradingView/Chartink alert: the platform validates the
    symbol/action against the strategy and queues the order.

    Args:
        webhook_id: Strategy webhook id (UUID).
        payload: Webhook body with symbol, action and optional position_size.

    Returns:
        JSON confirming the order was queued, or the platform error.
    """
    return _post_webhook(webhook_id, payload)


@mcp.tool()
def list_strategy_portfolio(watchlist: str | None = None) -> str:
    """
    List strategy portfolio entries, optionally filtered by watchlist.

    Args:
        watchlist: Optional watchlist filter (mytrades or simulation).

    Returns:
        JSON array of portfolio entries with legs and notes.
    """
    payload: dict[str, Any] = {}
    if watchlist:
        payload["watchlist"] = watchlist
    return _post_api_v1("/strategyportfolio/list", payload)


@mcp.tool()
def get_strategy_portfolio(entry_id: int) -> str:
    """
    Get a single strategy portfolio entry.

    Args:
        entry_id: Numeric portfolio entry id.

    Returns:
        JSON with the portfolio entry and its legs.
    """
    return _post_api_v1(f"/strategyportfolio/{entry_id}", {})


@mcp.tool()
def save_strategy_portfolio(
    name: str,
    watchlist: str,
    underlying: str,
    exchange: str,
    legs: list[dict],
    expiry: str | None = None,
    notes: str | None = None,
) -> str:
    """
    Create a new strategy portfolio entry.

    Args:
        name: Portfolio entry name.
        watchlist: mytrades or simulation.
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        legs: List of strategy leg definitions (option or future legs).
        expiry: Optional expiry date in DDMMMYY format (e.g., 28NOV25).
        notes: Optional free-text notes.

    Returns:
        JSON with the created portfolio entry.
    """
    payload: dict[str, Any] = {
        "name": name,
        "watchlist": watchlist,
        "underlying": underlying.upper(),
        "exchange": exchange.upper(),
        "legs": legs,
    }
    if expiry:
        payload["expiry"] = expiry.upper()
    if notes:
        payload["notes"] = notes
    return _post_api_v1("/strategyportfolio", payload)


@mcp.tool()
def update_strategy_portfolio(
    entry_id: int,
    name: str,
    watchlist: str,
    underlying: str,
    exchange: str,
    legs: list[dict],
    expiry: str | None = None,
    notes: str | None = None,
) -> str:
    """
    Update an existing strategy portfolio entry.

    Args:
        entry_id: Numeric portfolio entry id.
        name: Portfolio entry name.
        watchlist: mytrades or simulation.
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        legs: List of strategy leg definitions (option or future legs).
        expiry: Optional expiry date in DDMMMYY format (e.g., 28NOV25).
        notes: Optional free-text notes.

    Returns:
        JSON with the updated portfolio entry.
    """
    payload: dict[str, Any] = {
        "name": name,
        "watchlist": watchlist,
        "underlying": underlying.upper(),
        "exchange": exchange.upper(),
        "legs": legs,
    }
    if expiry:
        payload["expiry"] = expiry.upper()
    if notes:
        payload["notes"] = notes
    return _put_api_v1(f"/strategyportfolio/{entry_id}", payload)


@mcp.tool()
def delete_strategy_portfolio(entry_id: int) -> str:
    """
    Delete a strategy portfolio entry.

    Args:
        entry_id: Numeric portfolio entry id.

    Returns:
        JSON confirming deletion.
    """
    return _delete_api_v1(f"/strategyportfolio/{entry_id}")


@mcp.tool()
def analyze_market(
    underlying: str,
    exchange: str = "NSE_INDEX",
    expiry_date: str | None = None,
) -> str:
    """
    Get a consolidated snapshot of an underlying for market analysis.

    Aggregates quotes, option chain, GEX, max pain and IV smile into a
    single JSON payload for a high-level view of an expiry.

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE).
        exchange: Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS).
        expiry_date: Optional expiry in DDMMMYY format (e.g., 28NOV25). If omitted, resolved from the exchange.

    Returns:
        JSON with quotes, option chain, GEX, max pain and IV smile for the underlying.
    """
    resolved_expiry = expiry_date.upper() if expiry_date else None

    if not resolved_expiry:
        expiry_response = _post_api_v1(
            "/expiry",
            {
                "symbol": underlying.upper(),
                "exchange": exchange.upper(),
                "instrumenttype": "options",
            },
        )
        expiry_data = _parse_json_response(expiry_response)
        expiry_dates = expiry_data.get("expiry_dates") or expiry_data.get("data") or []
        if not expiry_dates:
            return json.dumps(
                {
                    "status": "error",
                    "message": "Could not resolve expiry date from exchange. Provide expiry_date in DDMMMYY format (e.g., 28NOV25).",
                    "expiry_response": expiry_response,
                },
                indent=2,
            )
        # Display format is DD-MMM-YY (e.g., 11-AUG-26); convert to DDMMMYY.
        resolved_expiry = _normalize_expiry_display(str(expiry_dates[0]))

    snapshot: dict[str, Any] = {
        "status": "success",
        "underlying": underlying.upper(),
        "exchange": exchange.upper(),
        "expiry_date": resolved_expiry,
    }

    quotes_response = _post_api_v1(
        "/quotes",
        {"symbol": underlying.upper(), "exchange": exchange.upper()},
    )
    quotes_data = _parse_json_response(quotes_response)
    if quotes_data.get("status") == "success":
        snapshot["quotes"] = quotes_data
    else:
        snapshot["quotes_error"] = quotes_response

    option_chain_response = _post_api_v1(
        "/optionchain",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": resolved_expiry,
            "strike_count": 10,
        },
    )
    chain_data = _parse_json_response(option_chain_response)
    if chain_data.get("status") == "success":
        snapshot["option_chain"] = chain_data
    else:
        snapshot["option_chain_error"] = option_chain_response

    gex_response = _post_api_v1(
        "/gex",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": resolved_expiry,
        },
    )
    gex_data = _parse_json_response(gex_response)
    if gex_data.get("status") == "success":
        snapshot["gex"] = gex_data
    else:
        snapshot["gex_error"] = gex_response

    max_pain_response = _post_api_v1(
        "/oitracker/maxpain",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": resolved_expiry,
        },
    )
    max_pain_data = _parse_json_response(max_pain_response)
    if max_pain_data.get("status") == "success":
        snapshot["max_pain"] = max_pain_data
    else:
        snapshot["max_pain_error"] = max_pain_response

    iv_smile_response = _post_api_v1(
        "/ivsmile",
        {
            "underlying": underlying.upper(),
            "exchange": exchange.upper(),
            "expiry_date": resolved_expiry,
        },
    )
    iv_smile_data = _parse_json_response(iv_smile_response)
    if iv_smile_data.get("status") == "success":
        snapshot["iv_smile"] = iv_smile_data
    else:
        snapshot["iv_smile_error"] = iv_smile_response

    return json.dumps(snapshot, indent=2, default=str)


# ---------------------------------------------------------------------------
# Phase 1.5 foundation tools: session context, snapshots, capabilities,
# health, and risk guardrails. These aggregate lower-level tools and become
# the default entry points for AI agents.
# ---------------------------------------------------------------------------


def _quick_post(path: str, payload: dict[str, Any], timeout: float = 5.0) -> str:
    """POST to the backend with a short timeout (used for health probes)."""
    if not host:
        return "Error calling {path}: host is not configured"
    url = f"{host.rstrip('/')}/api/v1/{path.lstrip('/')}"
    body = {"apikey": api_key, **payload}
    try:
        with httpx.Client(timeout=timeout) as http:
            r = http.post(url, json=body, headers={"Content-Type": "application/json"})
            return json.dumps(r.json(), indent=2, default=str)
    except Exception as e:
        return f"Error calling /api/v1{path}: {str(e)}"


def _snapshot_get(path: str, **payload: Any) -> dict[str, Any]:
    """Call a backend endpoint and return its parsed data dict (or error)."""
    try:
        response = _post_api_v1(path, payload)
        parsed = _parse_json_response(response)
        if isinstance(parsed, dict) and parsed.get("status") == "success":
            data = parsed.get("data")
            if isinstance(data, dict):
                return data
            if data is not None:
                return {"data": data}
            # Analytics endpoints have no top-level 'data' key; pass their
            # full payload (chain, pain_data, atm_strike, ...) through.
            return parsed
        return parsed if isinstance(parsed, dict) else {"response": response}
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Phase 2: market intelligence data layer.
#
# Snapshot tools below are the preferred interface for AI agents. Every
# section is normalized to a stable MCP schema (no raw provider field names)
# and carries provenance metadata: source (backend endpoint), provider
# (broker/vendor), fetched_at (IST ISO), latency_ms and freshness
# (live | cached | derived). Provider-specific payloads (e.g. broker funds
# keys) are never used by summaries; they are passed through explicitly
# tagged as provider_specific.
# ---------------------------------------------------------------------------


def _num(value: Any, default: float = 0.0) -> float:
    """Defensively convert a provider field to float (may be str/None)."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _snapshot_section(
    source: str, payload: dict[str, Any], freshness: str
) -> dict[str, Any]:
    """Fetch a backend section and wrap it with provenance metadata."""
    start = time.perf_counter()
    data = _snapshot_get(source, **payload)
    latency_ms = (time.perf_counter() - start) * 1000
    return {
        "source": source,
        "provider": _get_broker(),
        "fetched_at": _ist_now_iso(),
        "latency_ms": round(latency_ms, 1),
        "freshness": freshness,
        "data": data,
    }


def _section_data(section: dict[str, Any]) -> dict[str, Any]:
    """Return the data payload of a snapshot section (or {} on error)."""
    if not isinstance(section, dict):
        return {}
    data = section.get("data")
    return data if isinstance(data, dict) else {}


# /expiry validates exchange against derivative venues only; index and equity
# exchanges map to their derivative venue for option-expiry resolution.
_DERIVATIVE_EXCHANGE_MAP = {
    "NSE_INDEX": "NFO",
    "NSE": "NFO",
    "BSE_INDEX": "BFO",
    "BSE": "BFO",
}


def _resolve_expiry(
    underlying: str, exchange: str, expiry_date: str | None
) -> str | None:
    """Resolve the nearest expiry in DDMMMYY format (backend when omitted)."""
    if expiry_date:
        return _normalize_expiry_display(expiry_date)
    derivative_exchange = _DERIVATIVE_EXCHANGE_MAP.get(exchange, exchange)
    response = _post_api_v1(
        "/expiry",
        {
            "symbol": underlying,
            "exchange": derivative_exchange,
            "instrumenttype": "options",
        },
    )
    expiry_data = _parse_json_response(response)
    dates = (
        expiry_data.get("expiry_dates")
        if isinstance(expiry_data, dict)
        else None
    ) or (
        expiry_data.get("data") if isinstance(expiry_data, dict) else None
    ) or []
    if dates:
        return _normalize_expiry_display(str(dates[0]))
    return None


_FUNDS_KEY_MAP = {
    "availablecash": "cash",
    "available_cash": "cash",
    "cash": "cash",
    "availablemargin": "available_margin",
    "available_margin": "available_margin",
    "usedmargin": "used_margin",
    "used_margin": "used_margin",
    "marginused": "used_margin",
    "intradaypayin": "intraday_payin",
    "adhocmargin": "adhoc_margin",
    "adhoc_margin": "adhoc_margin",
    "collateral": "collateral",
    "collateralvalue": "collateral_value",
    "payin": "payin",
    "payout": "payout",
    "openingbalance": "opening_balance",
    "opening_balance": "opening_balance",
    "closingbalance": "closing_balance",
    "closing_balance": "closing_balance",
}


def _normalize_funds(funds: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Map broker-specific funds keys to stable names; remainder is provider-specific."""
    normalized: dict[str, Any] = {}
    provider_specific: dict[str, Any] = {}
    for key, value in funds.items():
        stable = _FUNDS_KEY_MAP.get(str(key).lower())
        if stable:
            normalized.setdefault(stable, value)
        else:
            provider_specific[key] = value
    return normalized, provider_specific


def _bias_hint(pcr_oi: Any, net_gex: Any) -> str:
    """Derive a plain-language positioning note from PCR and net GEX (best effort)."""
    notes: list[str] = []
    if isinstance(net_gex, (int, float)):
        if net_gex < 0:
            notes.append(
                "negative net GEX (dealers short gamma, volatility expansion likely)"
            )
        else:
            notes.append("positive net GEX (dealers long gamma, mean-reverting tape)")
    if isinstance(pcr_oi, (int, float)):
        if pcr_oi >= 1.2:
            notes.append("elevated PCR (defensive put positioning)")
        elif pcr_oi <= 0.8:
            notes.append("low PCR (risk-on call positioning)")
    return "; ".join(notes) if notes else "neutral"


def _top_oi_rows(chain: Any, side: str, limit: int = 5) -> list[dict[str, Any]]:
    """Return the top-OI strikes for one side of an OI chain (best effort)."""
    if not isinstance(chain, list):
        return []
    rows: list[tuple[float, Any]] = []
    for row in chain:
        if not isinstance(row, dict):
            continue
        strike = row.get("strike")
        oi = _num(row.get(f"{side}_oi"))
        if oi > 0 and strike is not None:
            rows.append((oi, strike))
    rows.sort(reverse=True)
    return [{"strike": strike, "oi": int(oi)} for oi, strike in rows[:limit]]


def _quote_summary(quote: dict[str, Any]) -> dict[str, Any]:
    """Derive a compact quote summary (best effort, missing fields -> 0)."""
    ltp = _num(quote.get("ltp"))
    prev_close = _num(quote.get("prev_close"))
    change = ltp - prev_close if prev_close else 0.0
    return {
        "ltp": ltp,
        "prev_close": prev_close,
        "change": round(change, 2),
        "change_pct": round(change / prev_close * 100, 2) if prev_close else 0.0,
        "day_range": {"low": _num(quote.get("low")), "high": _num(quote.get("high"))},
        "volume": _num(quote.get("volume")),
        "oi": _num(quote.get("oi")),
    }


def _summarize_market(
    quote_section: dict[str, Any],
    chain_section: dict[str, Any],
    gex_section: dict[str, Any],
    max_pain_section: dict[str, Any],
    iv_smile_section: dict[str, Any],
) -> dict[str, Any]:
    """Derive a compact market summary from fetched sections (best effort)."""
    quote = _section_data(quote_section)
    chain = _section_data(chain_section)
    gex = _section_data(gex_section)
    max_pain = _section_data(max_pain_section)
    iv_smile = _section_data(iv_smile_section)
    summary = _quote_summary(quote)
    summary["underlying"] = chain.get("underlying") or gex.get("underlying")
    summary["spot"] = _num(chain.get("underlying_ltp")) or summary["ltp"]
    summary["atm_strike"] = chain.get("atm_strike") or gex.get("atm_strike")
    summary["max_pain_strike"] = max_pain.get("max_pain_strike")
    summary["pcr_oi"] = gex.get("pcr_oi") or max_pain.get("pcr_oi")
    summary["net_gex"] = gex.get("total_net_gex")
    summary["atm_iv"] = iv_smile.get("atm_iv")
    summary["iv_skew"] = iv_smile.get("skew")
    summary["bias_hint"] = _bias_hint(summary["pcr_oi"], summary["net_gex"])
    summary["note"] = "Derived by MCP from the sections above; not a trading signal."
    return summary


def _summarize_positioning(
    oi_section: dict[str, Any],
    max_pain_section: dict[str, Any],
    gex_section: dict[str, Any],
    iv_smile_section: dict[str, Any],
) -> dict[str, Any]:
    """Derive a compact options positioning summary (best effort)."""
    oi = _section_data(oi_section)
    max_pain = _section_data(max_pain_section)
    gex = _section_data(gex_section)
    iv_smile = _section_data(iv_smile_section)
    return {
        "underlying": oi.get("underlying") or gex.get("underlying"),
        "spot": _num(oi.get("spot_price")) or _num(gex.get("spot_price")),
        "atm_strike": oi.get("atm_strike") or gex.get("atm_strike"),
        "max_pain_strike": max_pain.get("max_pain_strike"),
        "pcr_oi": oi.get("pcr_oi"),
        "pcr_volume": oi.get("pcr_volume"),
        "total_ce_oi": oi.get("total_ce_oi"),
        "total_pe_oi": oi.get("total_pe_oi"),
        "net_gex": gex.get("total_net_gex"),
        "atm_iv": iv_smile.get("atm_iv"),
        "iv_skew": iv_smile.get("skew"),
        "top_oi_strikes": {
            "ce": _top_oi_rows(oi.get("chain"), "ce"),
            "pe": _top_oi_rows(oi.get("chain"), "pe"),
        },
        "bias_hint": _bias_hint(oi.get("pcr_oi"), gex.get("total_net_gex")),
        "note": (
            "Derived by MCP from OI, max pain, GEX and IV smile sections; "
            "not a trading signal."
        ),
    }


def _history_summary(history_section: dict[str, Any]) -> dict[str, Any]:
    """Derive trend/range/volume stats from a history section (best effort)."""
    data = _section_data(history_section)
    candles = data.get("data") if isinstance(data, dict) else None
    if not isinstance(candles, list) or not candles:
        return {"candles": 0, "note": "History unavailable"}
    closes: list[float] = []
    volumes: list[float] = []
    for candle in candles:
        if not isinstance(candle, dict):
            continue
        try:
            closes.append(float(candle["close"]))
            volumes.append(float(candle.get("volume") or 0))
        except (TypeError, ValueError, KeyError):
            continue
    if not closes:
        return {"candles": len(candles), "note": "Could not derive prices from history"}
    first, last = closes[0], closes[-1]
    return {
        "candles": len(candles),
        "first_close": round(first, 2),
        "last_close": round(last, 2),
        "period_change_pct": round((last - first) / first * 100, 2) if first else 0.0,
        "period_high": round(max(closes), 2),
        "period_low": round(min(closes), 2),
        "avg_volume": int(sum(volumes) / len(volumes)) if volumes else 0,
    }


_HISTORY_CACHE_TTL_SECONDS = 3600


def _company_history_section(symbol: str, exchange: str) -> dict[str, Any]:
    """Read the Upstox-direct history cache written by scripts/fetch_upstox_data.py."""
    start = time.perf_counter()
    cache_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp",
        "cache",
        "company_history",
        f"{symbol}_{exchange}.json",
    )
    candles: list[Any] = []
    fetched_at: str | None = None
    note: str | None = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as fh:
                cache = json.load(fh)
            fetched_at = cache.get("fetched_at")
            raw_candles = cache.get("candles")
            if isinstance(raw_candles, list):
                candles = raw_candles
        except (OSError, ValueError) as exc:
            note = f"Could not read history cache: {exc}"
    else:
        note = (
            f"No cached history for {symbol}. Run: "
            f"scripts/fetch_upstox_data.py --symbol {symbol} --exchange {exchange}"
        )

    freshness = "stale"
    if fetched_at:
        try:
            parsed = datetime.fromisoformat(fetched_at)
            if time.time() - parsed.timestamp() <= _HISTORY_CACHE_TTL_SECONDS:
                freshness = "cached"
        except ValueError:
            freshness = "stale"

    latency_ms = (time.perf_counter() - start) * 1000
    data: dict[str, Any] = {"data": candles, "candle_count": len(candles)}
    if note:
        data["note"] = note
    return {
        "source": "upstox:historical-candle (file cache)",
        "provider": _get_broker(),
        "fetched_at": fetched_at or _ist_now_iso(),
        "latency_ms": round(latency_ms, 1),
        "freshness": freshness,
        "data": data,
    }


def _read_upstox_cache(
    cache_kind: str, file_name: str, source: str, fetch_hint: str
) -> dict[str, Any]:
    """Read one Upstox-direct cache file and return a provenance envelope.

    Mirrors _company_history_section for the news/fundamentals caches written by
    scripts/fetch_upstox_data.py. Freshness is "cached" while the file is
    younger than _HISTORY_CACHE_TTL_SECONDS, "stale" otherwise (or when the file
    is missing or unreadable, in which case data.note explains how to populate
    it). The envelope data carries the payload minus fetched_at.

    Args:
        cache_kind: Cache subdirectory under mcp/cache (e.g. "company_news").
        file_name: Cache file name (e.g. "RELIANCE_NSE.json").
        source: Provenance source label for the envelope.
        fetch_hint: data.note text used when the cache file is missing.
    """
    start = time.perf_counter()
    cache_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "mcp",
        "cache",
        cache_kind,
        file_name,
    )
    payload: dict[str, Any] = {}
    fetched_at: str | None = None
    note: str | None = None
    if os.path.exists(cache_file):
        try:
            with open(cache_file, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                payload = loaded
            fetched_at = payload.get("fetched_at")
        except (OSError, ValueError) as exc:
            note = f"Could not read cache: {exc}"
    else:
        note = fetch_hint

    freshness = "stale"
    if fetched_at:
        try:
            parsed = datetime.fromisoformat(fetched_at)
            if time.time() - parsed.timestamp() <= _HISTORY_CACHE_TTL_SECONDS:
                freshness = "cached"
        except ValueError:
            freshness = "stale"

    latency_ms = (time.perf_counter() - start) * 1000
    data: dict[str, Any] = {
        key: value for key, value in payload.items() if key != "fetched_at"
    }
    if note:
        data["note"] = note
    return {
        "source": source,
        "provider": _get_broker(),
        "fetched_at": fetched_at or _ist_now_iso(),
        "latency_ms": round(latency_ms, 1),
        "freshness": freshness,
        "data": data,
    }


def _company_news_section(symbol: str, exchange: str) -> dict[str, Any]:
    """Read the Upstox-direct news cache (mcp/cache/company_news/)."""
    return _read_upstox_cache(
        "company_news",
        f"{symbol}_{exchange}.json",
        "upstox:news (file cache)",
        (
            f"No cached news for {symbol}. Run: scripts/fetch_upstox_data.py "
            f"--symbol {symbol} --exchange {exchange} --mode news"
        ),
    )


def _company_fundamentals_section(
    field: str, symbol: str, exchange: str
) -> dict[str, Any]:
    """Read one Upstox-direct fundamentals cache (mcp/cache/company_fundamentals/).

    Args:
        field: Fundamentals field: corporate-actions, share-holdings or
            income-statement.
        symbol: Company symbol.
        exchange: Exchange (default NSE).
    """
    return _read_upstox_cache(
        "company_fundamentals",
        f"{field}_{symbol}_{exchange}.json",
        f"upstox:{field} (file cache)",
        (
            f"No cached {field} for {symbol}. Run: scripts/fetch_upstox_data.py "
            f"--symbol {symbol} --exchange {exchange} --mode {field}"
        ),
    )


def _summarize_news(news_section: dict[str, Any] | None) -> dict[str, Any]:
    """Condense the news cache into the top headlines (best effort)."""
    data = _section_data(news_section) if news_section else {}
    articles = data.get("news") if isinstance(data, dict) else None
    if not isinstance(articles, list) or not articles:
        return {
            "available": False,
            "note": data.get("note") if isinstance(data, dict) else None
            or "No news cached. Run: scripts/fetch_upstox_data.py --mode news",
        }
    headlines = [
        {
            "heading": article.get("heading"),
            "published_time": article.get("published_time"),
        }
        for article in articles[:3]
        if isinstance(article, dict)
    ]
    return {"available": True, "count": len(articles), "headlines": headlines}


def _shareholding_summary(raw: Any) -> dict[str, Any] | None:
    """Latest-period holding percentage per category (promoters, fii, ...).

    Upstox returns share-holding history newest-first, so entry [0] is the
    most recent quarter.
    """
    if not isinstance(raw, list) or not raw:
        return None
    holding: dict[str, Any] = {}
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        category = entry.get("category")
        history = entry.get("history")
        if not category or not isinstance(history, list) or not history:
            continue
        latest = history[0]
        if isinstance(latest, dict) and latest.get("value") is not None:
            holding[category] = {
                "period": latest.get("period"),
                "percent": latest.get("value"),
            }
    return holding or None


def _corporate_actions_summary(raw: Any) -> list[dict[str, Any]] | None:
    """Ex-date corporate actions (dividend, bonus, split, rights) as a list."""
    if not isinstance(raw, list) or not raw:
        return None
    events: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        event = {
            key: entry[key]
            for key in ("name", "expiry_date", "amount", "ratio")
            if entry.get(key) is not None
        }
        if event:
            events.append(event)
    return events or None


def _earnings_summary(raw: Any) -> dict[str, Any] | None:
    """Latest annual revenue / operating profit / net profit figures.

    Upstox returns income-statement history newest-first (the newest entry
    carries the change field), so entry [0] is the latest reported period.
    """
    if not isinstance(raw, dict) or not raw.get("income_statement"):
        return None
    figures: dict[str, Any] = {"units_in": raw.get("units_in")}
    for category in ("revenue", "operating_profit", "net_profit"):
        series: list[Any] = []
        for entry in raw.get("income_statement") or []:
            if isinstance(entry, dict) and entry.get("category") == category:
                series = entry.get("history") or []
                break
        if series:
            latest = series[0]
            if isinstance(latest, dict):
                figures[category] = {
                    "value": latest.get("value"),
                    "period": latest.get("period"),
                    "change": latest.get("change"),
                }
    return figures


def _summarize_fundamentals(
    corporate_actions_section: dict[str, Any] | None,
    shareholdings_section: dict[str, Any] | None,
    income_statement_section: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build the fundamentals block from the Upstox-direct caches (best effort)."""
    actions_data = (
        _section_data(corporate_actions_section) if corporate_actions_section else {}
    )
    shareholding_data = (
        _section_data(shareholdings_section) if shareholdings_section else {}
    )
    income_data = (
        _section_data(income_statement_section) if income_statement_section else {}
    )
    corporate_actions = _corporate_actions_summary(
        actions_data.get("data") if isinstance(actions_data, dict) else None
    )
    shareholding = _shareholding_summary(
        shareholding_data.get("data") if isinstance(shareholding_data, dict) else None
    )
    earnings = _earnings_summary(
        income_data.get("data") if isinstance(income_data, dict) else None
    )
    if not any((corporate_actions, shareholding, earnings)):
        return {
            "available": False,
            "reason": (
                "No fundamentals cached. Run: scripts/fetch_upstox_data.py "
                "--symbol <SYMBOL> --exchange <EXCHANGE> --mode "
                "corporate-actions | share-holdings | income-statement"
            ),
        }
    block: dict[str, Any] = {"available": True}
    if corporate_actions:
        block["corporate_actions"] = corporate_actions
    if shareholding:
        block["shareholding"] = shareholding
    if earnings:
        block["earnings"] = earnings
    return block


def _summarize_company(
    quote_section: dict[str, Any],
    symbol_section: dict[str, Any],
    history_section: dict[str, Any],
    news_section: dict[str, Any] | None = None,
    corporate_actions_section: dict[str, Any] | None = None,
    shareholdings_section: dict[str, Any] | None = None,
    income_statement_section: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive a compact company summary (best effort)."""
    quote = _section_data(quote_section)
    symbol = _section_data(symbol_section)
    summary = _quote_summary(quote)
    summary.update(_history_summary(history_section))
    summary["instrument"] = {
        key: symbol[key]
        for key in (
            "symbol",
            "token",
            "exchange",
            "instrumenttype",
            "lotsize",
            "tick_size",
            "expiry",
            "strike",
        )
        if isinstance(symbol, dict) and symbol.get(key) is not None
    }
    summary["fundamentals"] = _summarize_fundamentals(
        corporate_actions_section,
        shareholdings_section,
        income_statement_section,
    )
    summary["news"] = _summarize_news(news_section)
    return summary


def _summarize_portfolio(
    funds_section: dict[str, Any],
    holdings_section: dict[str, Any],
    positions_section: dict[str, Any],
    orders_section: dict[str, Any],
) -> dict[str, Any]:
    """Derive a compact portfolio summary (best effort)."""
    funds = _section_data(funds_section)
    holdings = _section_data(holdings_section)
    positions = _section_data(positions_section)
    orders = _section_data(orders_section)

    h_list = holdings.get("holdings") if isinstance(holdings, dict) else None
    h_stats = holdings.get("statistics") if isinstance(holdings, dict) else None
    pos_list = positions.get("data") if isinstance(positions, dict) else None
    if not isinstance(pos_list, list):
        pos_list = None
    ord_list = orders.get("data") if isinstance(orders, dict) else None
    if not isinstance(ord_list, list):
        ord_list = None

    holdings_pnl = _num(h_stats.get("pnl")) if isinstance(h_stats, dict) else 0.0
    if isinstance(h_stats, dict):
        holdings_value = _num(h_stats.get("currentvalue") or h_stats.get("current_value"))
    else:
        holdings_value = 0.0

    positions_pnl = 0.0
    positions_count = 0
    gross_exposure = 0.0
    for position in pos_list or []:
        if not isinstance(position, dict):
            continue
        positions_count += 1
        positions_pnl += _num(position.get("pnl"))
        gross_exposure += abs(_num(position.get("quantity")) * _num(position.get("ltp")))

    orders_count = len(ord_list or [])
    open_orders = sum(
        1
        for order in ord_list or []
        if isinstance(order, dict)
        and str(order.get("status", "")).lower() in ("open", "pending", "trigger pending")
    )

    return {
        "cash": funds.get("cash"),
        "holdings_count": len(h_list) if isinstance(h_list, list) else None,
        "holdings_value": round(holdings_value, 2) if holdings_value else None,
        "unrealized_pnl": round(holdings_pnl + positions_pnl, 2),
        "positions_count": positions_count,
        "gross_exposure": round(gross_exposure, 2),
        "orders_count": orders_count,
        "open_orders": open_orders,
        "note": (
            "Derived by MCP from funds/holdings/positions/orders sections. "
            "Funds keys vary by broker; see the funds section for the raw "
            "provider payload."
        ),
    }


def _summarize_positions(
    positions_section: dict[str, Any],
    open_position_section: dict[str, Any],
    trades_section: dict[str, Any],
) -> dict[str, Any]:
    """Derive a compact position summary (best effort)."""
    positions = _section_data(positions_section)
    open_position = _section_data(open_position_section)
    trades = _section_data(trades_section)

    pos_list = positions.get("data") if isinstance(positions, dict) else None
    if not isinstance(pos_list, list):
        pos_list = None
    tr_list = trades.get("data") if isinstance(trades, dict) else None
    if not isinstance(tr_list, list):
        tr_list = None

    count = 0
    total_pnl = 0.0
    day_pnl = 0.0
    gross_exposure = 0.0
    for position in pos_list or []:
        if not isinstance(position, dict):
            continue
        count += 1
        total_pnl += _num(position.get("pnl"))
        day_pnl += _num(
            position.get("day_pnl")
            or position.get("daypnl")
            or position.get("today_pnl")
        )
        gross_exposure += abs(_num(position.get("quantity")) * _num(position.get("ltp")))

    buys = 0
    sells = 0
    trade_value = 0.0
    for trade in tr_list or []:
        if not isinstance(trade, dict):
            continue
        trade_value += _num(trade.get("price") or trade.get("average_price")) * abs(
            _num(trade.get("quantity"))
        )
        if str(trade.get("side") or trade.get("action") or "").upper() == "BUY":
            buys += 1
        else:
            sells += 1

    return {
        "position_count": count,
        "total_pnl": round(total_pnl, 2),
        "day_pnl": round(day_pnl, 2),
        "gross_exposure": round(gross_exposure, 2),
        "trades_count": len(tr_list or []),
        "trade_value": round(trade_value, 2),
        "buys": buys,
        "sells": sells,
        "open_position_symbol": (
            open_position.get("symbol")
            if isinstance(open_position, dict)
            else None
        ),
        "note": (
            "Derived by MCP from position book, open position and trade book "
            "sections."
        ),
    }


@mcp.tool()
def set_session_context(
    broker: str | None = None,
    exchange: str | None = None,
    preferred_expiry: str | None = None,
    watchlist: str | None = None,
    portfolio: str | None = None,
    current_market: str | None = None,
) -> str:
    """Set lightweight session context used as defaults across tool calls.

    Reduces repetitive parameters: default broker, exchange, preferred expiry,
    watchlist, portfolio and current market are injected into snapshots and
    order validation when the caller does not pass them explicitly.

    Args:
        broker: Default broker name (e.g., upstox).
        exchange: Default exchange (e.g., NFO, NSE_INDEX).
        preferred_expiry: Default expiry in DDMMMYY format (e.g., 28AUG26).
        watchlist: Default watchlist name.
        portfolio: Default portfolio reference.
        current_market: Default market context label.

    Returns:
        JSON with the updated session context.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: none (immediate)
        expected_latency: <5 ms
    """
    for key, value in (
        ("broker", broker),
        ("exchange", exchange),
        ("preferred_expiry", preferred_expiry),
        ("watchlist", watchlist),
        ("portfolio", portfolio),
        ("current_market", current_market),
    ):
        if value is not None:
            _SESSION_CONTEXT[key] = value
    return json.dumps(_SESSION_CONTEXT, indent=2, default=str)


@mcp.tool()
def get_session_context() -> str:
    """Return the current session context (defaults for other tools).

    Returns:
        JSON with the session context fields.

    Tool metadata:
        rate_limit: 60 per minute
        cache_ttl: none (immediate)
        expected_latency: <5 ms
    """
    return json.dumps(_SESSION_CONTEXT, indent=2, default=str)


@mcp.tool()
def market_snapshot(
    underlying: str | None = None,
    exchange: str | None = None,
    expiry_date: str | None = None,
) -> str:
    """Consolidated market snapshot for an underlying.

    Aggregates quotes, option chain, GEX, max pain and IV smile into a single
    response, with per-section provenance and a derived summary. Defaults come
    from session context when not provided. Answers "analyze NIFTY/BANKNIFTY".

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (defaults to session context or NSE_INDEX).
        expiry_date: Expiry in DDMMMYY format (resolved from the backend when omitted).

    Returns:
        JSON with provenance-tagged sections (quote, option_chain, gex,
        max_pain, iv_smile) and a derived summary block.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 2-20 s (fans out to 5 backend analytics calls)
    """
    underlying = _normalize_symbol(
        underlying or _SESSION_CONTEXT.get("current_market") or "NIFTY"
    )
    exchange = (exchange or _SESSION_CONTEXT.get("exchange") or "NSE_INDEX").upper()
    expiry_date = _resolve_expiry(
        underlying, exchange, expiry_date or _SESSION_CONTEXT.get("preferred_expiry")
    )
    snapshot = {"underlying": underlying, "exchange": exchange}
    if not expiry_date:
        snapshot["expiry_error"] = (
            "Could not resolve expiry date. Pass expiry_date in DDMMMYY format."
        )
        return json.dumps(snapshot, indent=2, default=str)
    snapshot["expiry_date"] = expiry_date
    quote_section = _snapshot_section(
        "/quotes", {"symbol": underlying, "exchange": exchange}, "live"
    )
    chain_section = _snapshot_section(
        "/optionchain",
        {
            "underlying": underlying,
            "exchange": exchange,
            "expiry_date": expiry_date,
            "strike_count": 10,
        },
        "live",
    )
    gex_section = _snapshot_section(
        "/gex",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    max_pain_section = _snapshot_section(
        "/oitracker/maxpain",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    iv_smile_section = _snapshot_section(
        "/ivsmile",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    snapshot["quote"] = quote_section
    snapshot["option_chain"] = chain_section
    snapshot["gex"] = gex_section
    snapshot["max_pain"] = max_pain_section
    snapshot["iv_smile"] = iv_smile_section
    snapshot["summary"] = _summarize_market(
        quote_section, chain_section, gex_section, max_pain_section, iv_smile_section
    )
    return json.dumps(snapshot, indent=2, default=str)


@mcp.tool()
def option_snapshot(
    underlying: str,
    exchange: str = "NFO",
    expiry_date: str | None = None,
) -> str:
    """Consolidated options analytics snapshot for an underlying/expiry.

    Aggregates option chain, OI tracker, max pain, GEX and IV smile with
    per-section provenance and a derived positioning summary. Answers
    "analyze today's option positioning".

    Args:
        underlying: Underlying symbol (e.g., NIFTY, BANKNIFTY).
        exchange: Exchange (default NFO).
        expiry_date: Expiry in DDMMMYY format (resolved when omitted).

    Returns:
        JSON with provenance-tagged sections (option_chain, oi_tracker,
        max_pain, gex, iv_smile) and a derived summary block.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 2-20 s (fans out to 5 backend analytics calls)
    """
    underlying = _normalize_symbol(underlying)
    exchange = exchange.upper()
    expiry_date = _resolve_expiry(
        underlying, exchange, expiry_date or _SESSION_CONTEXT.get("preferred_expiry")
    )
    snapshot = {"underlying": underlying, "exchange": exchange}
    if not expiry_date:
        snapshot["expiry_error"] = (
            "Could not resolve expiry date. Pass expiry_date in DDMMMYY format."
        )
        return json.dumps(snapshot, indent=2, default=str)
    snapshot["expiry_date"] = expiry_date
    chain_section = _snapshot_section(
        "/optionchain",
        {
            "underlying": underlying,
            "exchange": exchange,
            "expiry_date": expiry_date,
            "strike_count": 10,
        },
        "live",
    )
    oi_section = _snapshot_section(
        "/oitracker",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    max_pain_section = _snapshot_section(
        "/oitracker/maxpain",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    gex_section = _snapshot_section(
        "/gex",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    iv_smile_section = _snapshot_section(
        "/ivsmile",
        {"underlying": underlying, "exchange": exchange, "expiry_date": expiry_date},
        "derived",
    )
    snapshot["option_chain"] = chain_section
    snapshot["oi_tracker"] = oi_section
    snapshot["max_pain"] = max_pain_section
    snapshot["gex"] = gex_section
    snapshot["iv_smile"] = iv_smile_section
    snapshot["summary"] = _summarize_positioning(
        oi_section, max_pain_section, gex_section, iv_smile_section
    )
    return json.dumps(snapshot, indent=2, default=str)


@mcp.tool()
def portfolio_snapshot() -> str:
    """Consolidated account snapshot: funds, holdings, positions and orders.

    Funds are normalized to stable keys (provider-specific remainder passed
    through tagged as such). Answers "analyze my portfolio".

    Returns:
        JSON with provenance-tagged sections (funds, holdings, positions,
        orders) and a derived summary block.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 1-5 s (4 backend account calls)
    """
    funds_section = _snapshot_section("/funds", {}, "live")
    funds_data = funds_section.get("data")
    if isinstance(funds_data, dict) and "error" not in funds_data:
        normalized, provider_specific = _normalize_funds(funds_data)
        funds_section["data"] = {
            "normalized": normalized,
            "provider_specific": provider_specific,
        }
    holdings_section = _snapshot_section("/holdings", {}, "live")
    positions_section = _snapshot_section("/positionbook", {}, "live")
    orders_section = _snapshot_section("/orderbook", {}, "live")
    snapshot = {
        "funds": funds_section,
        "holdings": holdings_section,
        "positions": positions_section,
        "orders": orders_section,
        "summary": _summarize_portfolio(
            funds_section, holdings_section, positions_section, orders_section
        ),
    }
    return json.dumps(snapshot, indent=2, default=str)


@mcp.tool()
def company_snapshot(company: str, exchange: str = "NSE") -> str:
    """Consolidated company snapshot: search, symbol info, history, quote, news
    and fundamentals (corporate actions, shareholdings, earnings).

    News and fundamentals come from Upstox-direct caches populated by
    scripts/fetch_upstox_data.py. Answers "analyze RELIANCE".

    Args:
        company: Company symbol or search term (e.g., INFY, RELIANCE).
        exchange: Exchange (default NSE).

    Returns:
        JSON with provenance-tagged sections (quote, symbol, search,
        history, news, corporate_actions, shareholdings, income_statement)
        and a derived summary block.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 60 s
        expected_latency: 1-5 s (3 backend data calls + 5 cache reads)
    """
    company = company.strip().upper()
    exchange = exchange.upper()
    quote_section = _snapshot_section(
        "/quotes", {"symbol": company, "exchange": exchange}, "live"
    )
    symbol_section = _snapshot_section(
        "/symbol", {"symbol": company, "exchange": exchange}, "live"
    )
    search_section = _snapshot_section(
        "/search", {"searchtext": company, "exchange": exchange}, "live"
    )
    history_section = _company_history_section(company, exchange)
    news_section = _company_news_section(company, exchange)
    corporate_actions_section = _company_fundamentals_section(
        "corporate-actions", company, exchange
    )
    shareholdings_section = _company_fundamentals_section(
        "share-holdings", company, exchange
    )
    income_statement_section = _company_fundamentals_section(
        "income-statement", company, exchange
    )
    snapshot = {
        "quote": quote_section,
        "symbol": symbol_section,
        "search": search_section,
        "history": history_section,
        "news": news_section,
        "corporate_actions": corporate_actions_section,
        "shareholdings": shareholdings_section,
        "income_statement": income_statement_section,
        "summary": _summarize_company(
            quote_section,
            symbol_section,
            history_section,
            news_section,
            corporate_actions_section,
            shareholdings_section,
            income_statement_section,
        ),
    }
    return json.dumps(snapshot, indent=2, default=str)


@mcp.tool()
def position_snapshot() -> str:
    """Consolidated position snapshot: position book, open position, trade book.

    Returns:
        JSON with provenance-tagged sections (positions, open_position,
        trades) and a derived summary block.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 1-5 s (3 backend account calls)
    """
    positions_section = _snapshot_section("/positionbook", {}, "live")
    open_position_section = _snapshot_section("/openposition", {}, "live")
    trades_section = _snapshot_section("/tradebook", {}, "live")
    snapshot = {
        "positions": positions_section,
        "open_position": open_position_section,
        "trades": trades_section,
        "summary": _summarize_positions(
            positions_section, open_position_section, trades_section
        ),
    }
    return json.dumps(snapshot, indent=2, default=str)


def _route_subject(lowered: str) -> str:
    """Pick an analysis route from the subject wording."""
    if any(k in lowered for k in ("portfolio", "holdings", "account")):
        return "portfolio"
    if "position" in lowered:
        return "position"
    if "option" in lowered:
        return "options"
    if any(k in lowered for k in ("company", "fundamental", "news")):
        return "company"
    if "capabilit" in lowered:
        return "capabilities"
    if "health" in lowered or "system" in lowered:
        return "health"
    return "market"


def _subject_symbol(subject: str) -> str:
    """Extract a symbol from a subject phrase (strip qualifiers)."""
    symbol = subject.upper().strip()
    for suffix in (" OPTIONS", " OPTION", " INDEX"):
        if symbol.endswith(suffix):
            symbol = symbol[: -len(suffix)].strip()
    return symbol or "NIFTY"


def _infer_exchange(symbol: str) -> str:
    """Default exchange for a symbol: index memberships, else NSE."""
    if symbol in NSE_INDEX_SYMBOLS:
        return "NSE_INDEX"
    if symbol in BSE_INDEX_SYMBOLS:
        return "BSE_INDEX"
    return "NSE"


def _snapshot_payload(fn, *args, **kwargs) -> dict:
    """Call a snapshot tool and return its inner data, tolerating failures."""
    try:
        parsed = json.loads(fn(*args, **kwargs))
    except (TypeError, ValueError) as exc:
        return {"error": f"tool call failed: {exc}"}
    if not isinstance(parsed, dict):
        return {"error": f"unexpected tool output: {str(parsed)[:200]}"}
    if parsed.get("success") is False:
        return {
            "error": parsed.get("message", "tool failed"),
            "error_code": parsed.get("error_code"),
        }
    if "data" in parsed:
        return parsed["data"]
    return parsed


@mcp.tool()
def analyze(subject: str, analysis_type: str = "auto") -> str:
    """High-level analysis entry point that routes a subject to a snapshot.

    One call to start most sessions: pass a symbol or a phrase like
    "my portfolio" or "BANKNIFTY options" and get a structured context
    back without choosing among the low-level snapshot tools. The
    specialized tools remain available for deeper work.

    Args:
        subject: What to analyze, e.g. "NIFTY", "RELIANCE",
            "BANKNIFTY options", "my portfolio", "my positions",
            "company RELIANCE", "capabilities", "system health".
        analysis_type: "auto" (route by subject wording) or an explicit
            "market", "options", "company", "portfolio", "position",
            "capabilities" or "health".

    Returns:
        JSON with the resolved route and the underlying snapshot payload.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 1-20 s (delegates to one snapshot tool)
    """
    subject_clean = (subject or "").strip()
    if not subject_clean:
        return json.dumps(
            {"status": "error", "message": "subject is required."}, indent=2
        )
    route = (analysis_type or "auto").strip().lower()
    if route not in (
        "auto", "market", "options", "company", "portfolio", "position",
        "capabilities", "health",
    ):
        return json.dumps(
            {
                "status": "error",
                "message": f"Unknown analysis_type {analysis_type!r}. Use auto, "
                "market, options, company, portfolio, position, capabilities "
                "or health.",
            },
            indent=2,
        )
    if route == "auto":
        route = _route_subject(subject_clean.lower())
    if route == "portfolio":
        payload = _snapshot_payload(portfolio_snapshot)
        resolved = {}
    elif route == "position":
        payload = _snapshot_payload(position_snapshot)
        resolved = {}
    elif route == "capabilities":
        payload = _snapshot_payload(get_capabilities)
        resolved = {}
    elif route == "health":
        payload = _snapshot_payload(system_health)
        resolved = {}
    elif route == "company":
        symbol = _subject_symbol(subject_clean)
        payload = _snapshot_payload(company_snapshot, symbol, "NSE")
        resolved = {"symbol": symbol, "exchange": "NSE"}
    elif route == "options":
        symbol = _subject_symbol(subject_clean)
        payload = _snapshot_payload(option_snapshot, symbol, "NFO")
        resolved = {"symbol": symbol, "exchange": "NFO"}
    else:
        symbol = _subject_symbol(subject_clean)
        exchange = _infer_exchange(symbol)
        payload = _snapshot_payload(market_snapshot, symbol, exchange)
        resolved = {"symbol": symbol, "exchange": exchange}
    return json.dumps(
        {
            "status": "success",
            "subject": subject_clean,
            "analysis_type": route,
            "resolved": resolved,
            "analysis": payload,
        },
        indent=2,
        default=str,
    )


def _registered_tool_count() -> int:
    """Number of tools registered with FastMCP (drift-safe against the map)."""
    manager = getattr(mcp, "_tool_manager", None)
    tools = getattr(manager, "_tools", None) if manager is not None else None
    if isinstance(tools, dict):
        return len(tools)
    return 0


@mcp.tool()
def get_capabilities() -> str:
    """Report MCP server capabilities so AI agents can adapt dynamically.

    Returns:
        JSON with available brokers, enabled modules, supported exchanges,
        supported analytics, versions and available snapshot/risk tools.

    Tool metadata:
        rate_limit: 30 per minute
        cache_ttl: 300 s
        expected_latency: <10 ms
    """
    return json.dumps(
        {
            "backend_version": _get_backend_version(),
            "mcp_version": MCP_VERSION,
            "broker": _get_broker(),
            "available_brokers": [
                b.strip()
                for b in os.getenv("VALID_BROKERS", "").split(",")
                if b.strip()
            ]
            or [_get_broker()],
            "enabled_modules": [
                "orders",
                "market_data",
                "options_analytics",
                "strategies",
                "strategy_portfolio",
                "risk_validation",
                "snapshots",
            ],
            "supported_exchanges": [
                "NSE", "BSE", "NFO", "BFO", "CDS", "BCD", "MCX", "NCDEX",
                "NSE_INDEX", "BSE_INDEX", "GLOBAL_INDEX",
            ],
            "supported_analytics": [
                "gex", "iv_smile", "oi_tracker", "max_pain", "oi_profile",
                "straddle", "vol_surface", "iv_chart", "gamma_density",
                "arbitrage", "multi_strike_oi", "indicators",
            ],
            "snapshot_tools": [
                "market_snapshot", "option_snapshot", "portfolio_snapshot",
                "company_snapshot", "position_snapshot",
            ],
            "risk_tools": [
                "validate_order", "estimate_order_risk", "dry_run_order",
            ],
            "tool_count": _registered_tool_count(),
            "feature_flags": {
                "http_transport": (
                    os.getenv("MCP_HTTP_ENABLED", "False").lower() == "true"
                ),
                "api_key_auth": (
                    os.getenv("MCP_API_KEY_AUTH", "True").lower() == "true"
                ),
                "write_scope_enabled": (
                    os.getenv("MCP_OAUTH_WRITE_SCOPE_ENABLED", "True").lower()
                    == "true"
                ),
                "tool_cache_enabled": True,
                "streaming": False,
            },
            "rate_limits": {
                "dispatch_per_minute": 120,
                "scope_read_per_minute": int(
                    os.getenv("MCP_RATE_LIMIT_READ", "60").split()[0]
                ),
                "scope_write_per_minute": int(
                    os.getenv("MCP_RATE_LIMIT_WRITE", "50").split()[0]
                ),
                "expensive_per_minute": int(
                    os.getenv("MCP_RATE_LIMIT_EXPENSIVE", "30").split()[0]
                ),
            },
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def system_health() -> str:
    """Report system health: broker, database, market data, versions, uptime.

    Returns:
        JSON with broker connectivity, database status, market data status,
        market status, versions and uptime.

    Tool metadata:
        rate_limit: 10 per minute
        cache_ttl: 30 s
        expected_latency: 50-500 ms
    """
    uptime_seconds = int(time.monotonic() - _START_TIME)
    ping_response = _quick_post("/ping", {})
    ping_ok = _parse_json_response(ping_response).get("status") == "success"
    quotes_probe = _quick_post(
        "/quotes", {"symbol": "NIFTY", "exchange": "NSE_INDEX"}, timeout=5.0
    )
    market_data_ok = _parse_json_response(quotes_probe).get("status") == "success"
    return json.dumps(
        {
            "status": "healthy" if ping_ok else "degraded",
            "uptime_seconds": uptime_seconds,
            "backend_version": _get_backend_version(),
            "mcp_version": MCP_VERSION,
            "broker": _get_broker(),
            "market_status": _get_market_status(),
            "broker_connectivity": "ok" if ping_ok else "down",
            "database_status": "ok" if ping_ok else "down",
            "market_data_status": "ok" if market_data_ok else "down",
            "cache_status": "ok",
            "scheduler_status": "not exposed",
            "websocket_status": "not exposed",
        },
        indent=2,
        default=str,
    )


_ORDER_PRICE_TYPES = ("MARKET", "LIMIT", "SL", "SL-M")
_ORDER_PRODUCTS = ("CNC", "NRML", "MIS")


def _validate_order_fields(
    symbol: str,
    quantity: int,
    action: str,
    exchange: str,
    price_type: str,
    product: str,
    price: float | None,
    trigger_price: float | None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Validate order fields without placing any order. Returns (valid, errors, normalized)."""
    errors: list[str] = []
    normalized = {
        "symbol": _normalize_symbol(symbol),
        "quantity": quantity,
        "action": action.upper(),
        "exchange": exchange.upper(),
        "price_type": price_type.upper(),
        "product": product.upper(),
        "price": price,
        "trigger_price": trigger_price,
    }
    if normalized["action"] not in ("BUY", "SELL"):
        errors.append("action must be BUY or SELL")
    if normalized["price_type"] not in _ORDER_PRICE_TYPES:
        errors.append(f"price_type must be one of {', '.join(_ORDER_PRICE_TYPES)}")
    if normalized["product"] not in _ORDER_PRODUCTS:
        errors.append(f"product must be one of {', '.join(_ORDER_PRODUCTS)}")
    if not isinstance(quantity, int) or quantity <= 0:
        errors.append("quantity must be a positive integer")
    if normalized["price_type"] in ("LIMIT", "SL", "SL-M") and (
        price is None or price <= 0
    ):
        errors.append("price is required and must be positive for LIMIT/SL/SL-M")
    if normalized["price_type"] == "SL" and (trigger_price is None or trigger_price <= 0):
        errors.append("trigger_price is required and must be positive for SL")
    return (not errors, errors, normalized)


@mcp.tool()
def validate_order(
    symbol: str,
    quantity: int,
    action: str,
    exchange: str = "NSE",
    price_type: str = "MARKET",
    product: str = "MIS",
    price: float | None = None,
    trigger_price: float | None = None,
) -> str:
    """Validate an order before execution. Never places an order.

    Checks action, price type, product, quantity and required price fields.

    Args:
        symbol: Symbol in canonical format (e.g., NIFTY, NIFTY11AUG2624500CE).
        quantity: Quantity as a positive integer.
        action: BUY or SELL.
        exchange: Exchange (default NSE).
        price_type: MARKET, LIMIT, SL or SL-M (default MARKET).
        product: CNC, NRML or MIS (default MIS).
        price: Required for LIMIT/SL/SL-M.
        trigger_price: Required for SL.

    Returns:
        JSON with valid flag, errors and normalized order fields.

    Tool metadata:
        rate_limit: 30 per minute
        cache_ttl: none
        expected_latency: <5 ms
    """
    try:
        valid, errors, normalized = _validate_order_fields(
            symbol, quantity, action, exchange, price_type, product, price, trigger_price
        )
    except ValueError as e:
        return json.dumps({"valid": False, "errors": [str(e)], "normalized_order": None})
    return json.dumps(
        {"valid": valid, "errors": errors, "normalized_order": normalized},
        indent=2,
        default=str,
    )


@mcp.tool()
def estimate_order_risk(
    symbol: str,
    quantity: int,
    action: str,
    exchange: str = "NSE",
    price: float | None = None,
) -> str:
    """Estimate order risk (notional and max loss). Never places an order.

    Fetches the current price when not provided.

    Args:
        symbol: Symbol in canonical format.
        quantity: Quantity as a positive integer.
        action: BUY or SELL.
        exchange: Exchange (default NSE).
        price: Reference price; fetched from quotes when omitted.

    Returns:
        JSON with notional value, estimated max loss and disclaimer.

    Tool metadata:
        rate_limit: 20 per minute
        cache_ttl: 15 s
        expected_latency: 50-500 ms
    """
    try:
        symbol = _normalize_symbol(symbol)
        exchange = exchange.upper()
        if not isinstance(quantity, int) or quantity <= 0:
            return json.dumps(
                {
                    "valid": False,
                    "errors": ["quantity must be a positive integer"],
                }
            )
        if price is None or price <= 0:
            quote = _snapshot_get("/quotes", symbol=symbol, exchange=exchange)
            price = quote.get("ltp") or quote.get("last_price")
        if price is None:
            return json.dumps(
                {
                    "valid": False,
                    "errors": [
                        "Could not determine reference price; pass price explicitly"
                    ],
                }
            )
        notional = float(price) * quantity
        max_loss = notional if action.upper() == "BUY" else notional
        return json.dumps(
            {
                "valid": True,
                "symbol": symbol,
                "exchange": exchange,
                "quantity": quantity,
                "reference_price": float(price),
                "notional_value": round(notional, 2),
                "estimated_max_loss": round(max_loss, 2),
                "note": (
                    "Margin requirement depends on broker leverage and is not "
                    "estimated here. This is a risk estimate only; no order was placed."
                ),
            },
            indent=2,
            default=str,
        )
    except ValueError as e:
        return json.dumps({"valid": False, "errors": [str(e)]})


@mcp.tool()
def dry_run_order(
    symbol: str,
    quantity: int,
    action: str,
    exchange: str = "NSE",
    price_type: str = "MARKET",
    product: str = "MIS",
    price: float | None = None,
    trigger_price: float | None = None,
) -> str:
    """Simulate an order: build the exact payload, validate it, never execute.

    Returns the payload a real place_order call would send so callers can
    inspect it before committing to execution.

    Args:
        symbol: Symbol in canonical format.
        quantity: Quantity as a positive integer.
        action: BUY or SELL.
        exchange: Exchange (default NSE).
        price_type: MARKET, LIMIT, SL or SL-M (default MARKET).
        product: CNC, NRML or MIS (default MIS).
        price: Required for LIMIT/SL/SL-M.
        trigger_price: Required for SL.

    Returns:
        JSON with dry_run flag, would_execute, payload, validation and a note.

    Tool metadata:
        rate_limit: 30 per minute
        cache_ttl: none
        expected_latency: <10 ms
    """
    try:
        valid, errors, normalized = _validate_order_fields(
            symbol, quantity, action, exchange, price_type, product, price, trigger_price
        )
    except ValueError as e:
        return json.dumps(
            {"dry_run": True, "valid": False, "errors": [str(e)], "payload": None}
        )
    if not valid:
        return json.dumps(
            {
                "dry_run": True,
                "valid": False,
                "errors": errors,
                "payload": None,
                "note": "No real order was placed.",
            },
            indent=2,
            default=str,
        )
    payload = {
        "symbol": normalized["symbol"],
        "exchange": normalized["exchange"],
        "action": normalized["action"],
        "quantity": str(normalized["quantity"]),
        "product": normalized["product"],
        "pricetype": normalized["price_type"],
        "price": "0" if normalized["price"] is None else str(normalized["price"]),
        "trigger_price": (
            "0" if normalized["trigger_price"] is None else str(normalized["trigger_price"])
        ),
        "strategy": MCP_STRATEGY,
    }
    return json.dumps(
        {
            "dry_run": True,
            "valid": True,
            "would_execute": "place_order",
            "payload": payload,
            "validation": normalized,
            "note": "No real order was placed. Use place_order to execute.",
        },
        indent=2,
        default=str,
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
