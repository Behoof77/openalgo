"""
lot_size.py — Dynamic lot-size fetch from broker instrument master.

Safety rule: NEVER hardcode lot size. Always fetch from the instrument master
and validate before every order.

Uses the OpenAlgo search_instruments API to look up the current lot size for
a given symbol + exchange. Falls back to a cached value if the API is unavailable.
"""

import os
import time
import requests

# Cache: {symbol_exchange: (lot_size, timestamp)}
_lot_cache: dict[str, tuple[int, float]] = {}
CACHE_TTL_SECONDS = 3600  # 1 hour


def get_api_host() -> str:
    """Return the OpenAlgo API host URL."""
    return os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000")


def get_api_key() -> str:
    """Return the OpenAlgo API key."""
    return os.getenv("OPENALGO_API_KEY", "")


def get_lot_size(symbol: str, exchange: str) -> int:
    """Fetch the current lot size for a symbol from the broker instrument master.

    Args:
        symbol: Trading symbol (e.g. "NIFTY", "BANKNIFTY").
        exchange: Exchange code (e.g. "NFO", "NSE").

    Returns:
        Lot size as integer.

    Raises:
        ValueError: If lot size cannot be determined.
    """
    cache_key = f"{symbol}_{exchange}"

    # Check cache
    if cache_key in _lot_cache:
        lot_size, ts = _lot_cache[cache_key]
        if time.time() - ts < CACHE_TTL_SECONDS:
            return lot_size

    # Fetch from API
    api_host = get_api_host()
    api_key = get_api_key()

    try:
        resp = requests.post(
            f"{api_host}/api/v1/searchinstruments",
            headers={"X-API-KEY": api_key},
            json={"symbol": symbol, "exchange": exchange},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if data.get("status") == "success" and data.get("data"):
            instruments = data["data"]
            # Find exact match
            for inst in instruments:
                if inst.get("symbol", "").upper() == symbol.upper():
                    lot_size = int(inst.get("lotsize", 0))
                    if lot_size > 0:
                        _lot_cache[cache_key] = (lot_size, time.time())
                        return lot_size

            # If no exact match but we got results, use the first one
            if instruments:
                lot_size = int(instruments[0].get("lotsize", 0))
                if lot_size > 0:
                    _lot_cache[cache_key] = (lot_size, time.time())
                    return lot_size

    except (requests.RequestException, KeyError, ValueError) as e:
        # API failure — check stale cache
        if cache_key in _lot_cache:
            lot_size, _ = _lot_cache[cache_key]
            print(f"[lot_size] API fetch failed ({e}), using cached lot_size={lot_size}")
            return lot_size

    raise ValueError(
        f"Cannot determine lot size for {symbol} on {exchange}. "
        f"Instrument master returned no lotsize. Symbol may not exist on this exchange."
    )


def validate_lot_size(symbol: str, exchange: str, expected: int) -> int:
    """Validate that the current lot size matches the expected value.

    Fetches the current lot size from the instrument master and compares.
    Logs a warning if they differ but returns the CURRENT lot size (the
    authoritative value from the broker).

    Args:
        symbol: Trading symbol.
        exchange: Exchange code.
        expected: Lot size the strategy was configured with.

    Returns:
        Current lot size (may differ from expected).
    """
    current = get_lot_size(symbol, exchange)
    if current != expected:
        print(
            f"[lot_size] WARNING: Lot size changed for {symbol} on {exchange}: "
            f"configured={expected}, current={current}. Using current={current}."
        )
    return current


def clear_cache() -> None:
    """Clear the lot-size cache. Called on strategy restart."""
    _lot_cache.clear()
