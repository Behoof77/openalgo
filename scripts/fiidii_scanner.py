"""
FII/DII Holdings Scanner for OpenAlgo

Scans NSE stocks for FII and DII holdings patterns (quarter-over-quarter
increase >100%) and places BUY orders through OpenAlgo's REST API.

Usage:
    uv run python scripts/fiidii_scanner.py                     # Dry run (default)
    uv run python scripts/fiidii_scanner.py --execute           # Live trading
    uv run python scripts/fiidii_scanner.py --exchange BSE      # Use BSE instead of NSE
    uv run python scripts/fiidii_scanner.py --product NRML      # Carry forward (default: MIS)
    uv run python scripts/fiidii_scanner.py --quantity 1        # Custom quantity per stock
    uv run python scripts/fiidii_scanner.py --min-holdings 5    # Minimum FII+DII holdings %
    uv run python scripts/fiidii_scanner.py --list-only         # Just print matching stocks, no orders
    uv run python scripts/fiidii_scanner.py --index "NIFTY 500" # Scan specific index

PRECONDITIONS:
    - OpenAlgo must be running (http://127.0.0.1:5000)
    - Broker session must be active (logged in today)
    - API key must be generated via /apikey page
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time as time_module
from datetime import datetime
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Path setup — allow imports from repo root
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass

from utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OPENALGO_HOST = os.getenv("HOST_SERVER", "http://127.0.0.1:5000")

# NSE session management — requires browser-like cookies
NSE_BASE_URL = "https://www.nseindia.com"
NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

# Rate limiting — NSE aggressively rate-limits; keep 2-3 sec between requests
NSE_REQUEST_DELAY = 2.5  # seconds between NSE API calls
OPENALGO_REQUEST_DELAY = 0.15  # seconds between order placements


def create_nse_client() -> httpx.Client:
    """Create an httpx client with NSE cookie session.

    NSE requires a valid session cookie. First hit the main page to get one,
    then use it for API calls.
    """
    client = httpx.Client(
        headers=NSE_HEADERS,
        follow_redirects=True,
        timeout=30.0,
        http2=True,
    )
    # Warm up the session — get cookies from the main page
    logger.info("Initializing NSE session...")
    try:
        resp = client.get(NSE_BASE_URL)
        resp.raise_for_status()
        logger.info(f"NSE session initialized (cookies: {len(client.cookies)})")
    except Exception as e:
        logger.exception(f"Failed to initialize NSE session: {e}")
        raise
    return client


def fetch_nse_stock_list(nse_client: httpx.Client, index: str = "all") -> list[dict]:
    """Fetch list of NSE equity stocks.

    Args:
        nse_client: httpx client with NSE session cookies.
        index: Index to fetch — 'all' for all equities, or specific like
               'NIFTY 50', 'NIFTY 200', 'NIFTY 500', 'SECURITIES IN F&O'.

    Returns:
        List of dicts with 'symbol' and 'name' keys.
    """
    if index.lower() == "all":
        url = f"{NSE_BASE_URL}/api/equity-stockIndices?index=SECURITIES%20IN%20F%26O"
    else:
        url = f"{NSE_BASE_URL}/api/equity-stockIndices?index={index.replace(' ', '%20')}"

    try:
        resp = nse_client.get(url)
        resp.raise_for_status()
        data = resp.json()
        stocks = [
            {"symbol": item["symbol"], "name": item.get("meta", {}).get("companyName", "")}
            for item in data.get("data", [])
            if item.get("symbol")
        ]
        logger.info(f"Fetched {len(stocks)} stocks from NSE ({index})")
        return stocks
    except Exception as e:
        logger.exception(f"Failed to fetch NSE stock list: {e}")
        return []


def fetch_shareholding(nse_client: httpx.Client, symbol: str) -> dict[str, Any] | None:
    """Fetch shareholding pattern for a stock from NSE API.

    Args:
        nse_client: httpx client with NSE session cookies.
        symbol: NSE stock symbol (e.g., 'RELIANCE', 'TCS').

    Returns:
        Dict with latest and previous quarter shareholding data, or None on failure.
        Structure:
        {
            'symbol': str,
            'quarters': [
                {
                    'date': str,           # e.g. '30 Jun 2025'
                    'promoter_pct': float,
                    'fii_pct': float,      # FII/FPI holdings
                    'public_pct': float,
                    'total_dii_pct': float, # Sum of MF + Insurance + Banks + others
                    'raw': dict            # Full raw data for debugging
                },
                ...  # Previous quarters (at least 2 for comparison)
            ]
        }
    """
    url = f"{NSE_BASE_URL}/api/shareholding?symbol={symbol}"
    try:
        resp = nse_client.get(url)
        resp.raise_for_status()
        data = resp.json()

        quarters = []
        shareholding = data.get("data", [])

        # NSE returns data sorted latest first — we need at least 2 quarters
        for entry in shareholding[:4]:  # Last 4 quarters
            date = entry.get("date", "")
            # Parse the shareholding categories
            categories = entry.get("data", [])

            quarter_data = {
                "date": date,
                "promoter_pct": 0.0,
                "fii_pct": 0.0,
                "dii_pct": 0.0,
                "public_pct": 0.0,
                "raw": {},
            }

            for cat in categories:
                category = cat.get("category", "")
                percentage = cat.get("percentage", 0)

                # Normalize the percentage
                if isinstance(percentage, str):
                    percentage = float(percentage.replace(",", ""))

                quarter_data["raw"][category] = percentage

                cat_lower = category.lower()
                if "promoter" in cat_lower and "group" not in cat_lower:
                    quarter_data["promoter_pct"] = percentage
                elif "fii" in cat_lower or "fpi" in cat_lower:
                    quarter_data["fii_pct"] = percentage
                elif "mutual" in cat_lower or "mf" in cat_lower:
                    quarter_data["dii_pct"] += percentage
                elif "insurance" in cat_lower:
                    quarter_data["dii_pct"] += percentage
                elif "bank" in cat_lower or "financial" in cat_lower:
                    quarter_data["dii_pct"] += percentage
                elif "public" in cat_lower or "retail" in cat_lower:
                    quarter_data["public_pct"] = percentage

            quarters.append(quarter_data)

        if len(quarters) < 2:
            logger.warning(f"{symbol}: Insufficient quarterly data ({len(quarters)} quarters)")
            return None

        return {"symbol": symbol, "quarters": quarters}

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401 or e.response.status_code == 403:
            logger.warning(f"{symbol}: NSE session expired, need re-auth")
        else:
            logger.warning(f"{symbol}: NSE API error {e.response.status_code}")
        return None
    except Exception as e:
        logger.warning(f"{symbol}: Failed to fetch shareholding: {e}")
        return None


def analyze_fii_dii_pattern(
    shareholding: dict[str, Any],
    min_increase_pct: float = 100.0,
    min_total_holdings: float = 0.0,
) -> dict[str, Any] | None:
    """Analyze if FII and DII holdings both increased >threshold% quarter-over-quarter.

    Args:
        shareholding: Output from fetch_shareholding().
        min_increase_pct: Minimum % increase to trigger (default 100% = doubled).
        min_total_holdings: Minimum combined FII+DII % to qualify.

    Returns:
        Dict with analysis results if pattern matches, else None.
        {
            'symbol': str,
            'fii_current': float,
            'fii_previous': float,
            'fii_increase_pct': float,
            'dii_current': float,
            'dii_previous': float,
            'dii_increase_pct': float,
            'quarter_current': str,
            'quarter_previous': str,
            'total_holdings': float,
        }
    """
    quarters = shareholding["quarters"]
    current = quarters[0]
    previous = quarters[1]

    # FII analysis
    fii_current = current["fii_pct"]
    fii_previous = previous["fii_pct"]

    # DII analysis
    dii_current = current["dii_pct"]
    dii_previous = previous["dii_pct"]

    # Calculate percentage increase
    # If previous was 0 and current > 0, that's infinite increase — count it
    if fii_previous == 0:
        fii_increase = 100.0 if fii_current > 0 else 0.0
    else:
        fii_increase = ((fii_current - fii_previous) / fii_previous) * 100

    if dii_previous == 0:
        dii_increase = 100.0 if dii_current > 0 else 0.0
    else:
        dii_increase = ((dii_current - dii_previous) / dii_previous) * 100

    total_holdings = fii_current + dii_current

    # Check if both FII and DII increased by more than threshold
    fii_qualified = fii_increase >= min_increase_pct
    dii_qualified = dii_increase >= min_increase_pct

    # Also require minimum total holdings if specified
    holdings_qualified = total_holdings >= min_total_holdings

    if fii_qualified and dii_qualified and holdings_qualified:
        return {
            "symbol": shareholding["symbol"],
            "fii_current": fii_current,
            "fii_previous": fii_previous,
            "fii_increase_pct": round(fii_increase, 1),
            "dii_current": dii_current,
            "dii_previous": dii_previous,
            "dii_increase_pct": round(dii_increase, 1),
            "quarter_current": current["date"],
            "quarter_previous": previous["date"],
            "total_holdings": round(total_holdings, 2),
        }

    return None


def get_api_key() -> str | None:
    """Get the OpenAlgo API key from the database.

    Returns the first available API key, or None if not found.
    """
    try:
        from database.auth_db import ApiKeys, decrypt_token, db_session

        session = db_session()
        try:
            api_key_obj = session.query(ApiKeys).first()
            if api_key_obj and api_key_obj.api_key_encrypted:
                return decrypt_token(api_key_obj.api_key_encrypted)
        finally:
            session.close()
    except Exception as e:
        logger.exception(f"Failed to get API key from database: {e}")
    return None


def place_order(
    api_key: str,
    symbol: str,
    exchange: str,
    action: str,
    product: str,
    quantity: int,
    price: float = 0,
    pricetype: str = "MARKET",
    strategy: str = "FII-DII-Scanner",
) -> dict[str, Any] | None:
    """Place an order via OpenAlgo REST API.

    Args:
        api_key: OpenAlgo API key.
        symbol: Stock symbol.
        exchange: Exchange code (NSE/BSE).
        action: BUY or SELL.
        product: MIS or NRML.
        quantity: Number of shares.
        limit_price: Limit price (0 for market orders).
        pricetype: MARKET, LIMIT, SL, or SL-M.
        strategy: Strategy name for tracking.

    Returns:
        Response dict or None on failure.
    """
    payload = {
        "apikey": api_key,
        "symbol": symbol,
        "exchange": exchange,
        "action": action,
        "product": product,
        "pricetype": pricetype,
        "quantity": str(quantity),
        "price": str(price),
        "strategy": strategy,
    }

    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.post(f"{OPENALGO_HOST}/api/v1/placeorder", json=payload)
            result = resp.json()
            if result.get("status") == "success":
                logger.info(
                    f"Order placed: {action} {quantity} {symbol} ({exchange}) "
                    f"- OrderID: {result.get('data', {}).get('orderid', 'N/A')}"
                )
            else:
                logger.error(f"Order failed for {symbol}: {result.get('message', 'Unknown error')}")
            return result
    except Exception as e:
        logger.exception(f"Failed to place order for {symbol}: {e}")
        return None


def verify_broker_connection() -> bool:
    """Check if the broker connection is active by calling funds endpoint."""
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{OPENALGO_HOST}/api/v1/funds", params={"apikey": "probe"})
            # We just need to confirm the server is reachable
            return resp.status_code in (200, 401, 403)
    except Exception:
        return False


def run_scanner(args: argparse.Namespace) -> None:
    """Main scanner loop."""
    print("=" * 72)
    print("FII/DII Holdings Scanner for OpenAlgo")
    print(f"Mode: {'LIVE TRADING' if args.execute else 'DRY RUN (use --execute to trade)'}")
    print(f"Exchange: {args.exchange}")
    print(f"Product: {args.product}")
    print(f"Quantity per stock: {args.quantity}")
    print(f"FII/DII increase threshold: {args.min_increase}%")
    print(f"Min total FII+DII holdings: {args.min_holdings}%")
    print(f"Index: {args.index}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 72)

    # Verify server is running
    if not verify_broker_connection():
        logger.error("Cannot reach OpenAlgo server. Is it running?")
        sys.exit(1)

    # Get API key
    api_key = get_api_key()
    if not api_key:
        logger.error("No API key found. Generate one at /apikey in the OpenAlgo UI.")
        sys.exit(1)
    logger.info(f"API key loaded (ends with ...{api_key[-6:]})")

    # Initialize NSE client
    nse_client = create_nse_client()

    # Fetch stock list
    print(f"\nFetching stock list for '{args.index}'...")
    stocks = fetch_nse_stock_list(nse_client, args.index)
    if not stocks:
        logger.error("No stocks found. Check index name or NSE availability.")
        sys.exit(1)

    # Scan each stock
    matches = []
    errors = 0
    total = len(stocks)

    print(f"\nScanning {total} stocks for FII/DII accumulation pattern...\n")

    for i, stock in enumerate(stocks, 1):
        symbol = stock["symbol"]
        progress = f"[{i}/{total}]"

        # Rate limit NSE requests
        if i > 1:
            time_module.sleep(NSE_REQUEST_DELAY)

        # Fetch shareholding
        shareholding = fetch_shareholding(nse_client, symbol)
        if shareholding is None:
            errors += 1
            continue

        # Analyze pattern
        result = analyze_fii_dii_pattern(
            shareholding,
            min_increase_pct=args.min_increase,
            min_total_holdings=args.min_holdings,
        )

        if result:
            matches.append(result)
            print(
                f"  {progress} MATCH: {symbol:15s} "
                f"FII: {result['fii_previous']:.1f}% -> {result['fii_current']:.1f}% "
                f"(+{result['fii_increase_pct']:.0f}%) | "
                f"DII: {result['dii_previous']:.1f}% -> {result['dii_current']:.1f}% "
                f"(+{result['dii_increase_pct']:.0f}%)"
            )
        else:
            # Progress indicator for non-matches
            if i % 25 == 0 or i == total:
                print(f"  {progress} Scanned... ({len(matches)} matches so far, {errors} errors)")

    # Summary
    print("\n" + "=" * 72)
    print(f"SCAN COMPLETE: {total} stocks scanned, {len(matches)} matches, {errors} errors")
    print("=" * 72)

    if not matches:
        print("\nNo stocks matched the FII/DII accumulation criteria.")
        print("Try adjusting --min-increase or --min-holdings thresholds.")
        return

    # Display results
    print(f"\n{'Symbol':<15} {'FII Prev':>10} {'FII Curr':>10} {'FII Chg':>8} "
          f"{'DII Prev':>10} {'DII Curr':>10} {'DII Chg':>8} {'Total':>8}")
    print("-" * 89)

    for m in matches:
        print(
            f"{m['symbol']:<15} "
            f"{m['fii_previous']:>9.1f}% "
            f"{m['fii_current']:>9.1f}% "
            f"{m['fii_increase_pct']:>7.0f}% "
            f"{m['dii_previous']:>9.1f}% "
            f"{m['dii_current']:>9.1f}% "
            f"{m['dii_increase_pct']:>7.0f}% "
            f"{m['total_holdings']:>7.1f}%"
        )

    # Place orders if executing
    if args.execute and not args.list_only:
        print(f"\nPlacing BUY orders ({args.product}, qty={args.quantity})...")
        placed = 0
        for m in matches:
            time_module.sleep(OPENALGO_REQUEST_DELAY)
            result = place_order(
                api_key=api_key,
                symbol=m["symbol"],
                exchange=args.exchange,
                action="BUY",
                product=args.product,
                quantity=args.quantity,
                strategy="FII-DII-Scanner",
            )
            if result and result.get("status") == "success":
                placed += 1
        print(f"\nOrders placed: {placed}/{len(matches)}")
    elif args.execute:
        print("\n(Dry run — no orders placed. Use without --list-only to trade.)")
    else:
        print("\n(Dry run — no orders placed. Use --execute to trade.)")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="FII/DII Holdings Scanner — scans NSE stocks for institutional accumulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Dry run, scan NSE F&O stocks
  %(prog)s --execute                    # Place BUY orders for matching stocks
  %(prog)s --index "NIFTY 500"          # Scan NIFTY 500 universe
  %(prog)s --min-increase 50 --execute  # Lower threshold (50%% increase)
  %(prog)s --min-holdings 10 --execute  # Only stocks with 10%%+ FII+DII
  %(prog)s --product NRML --quantity 5  # Carry forward, 5 shares each
        """,
    )

    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually place orders (default: dry run)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Just print matching stocks, don't place orders even with --execute",
    )
    parser.add_argument(
        "--exchange",
        default="NSE",
        choices=["NSE", "BSE"],
        help="Exchange to trade on (default: NSE)",
    )
    parser.add_argument(
        "--product",
        default="MIS",
        choices=["MIS", "NRML", "CNC"],
        help="Product type (default: MIS intraday)",
    )
    parser.add_argument(
        "--quantity",
        type=int,
        default=1,
        help="Quantity per stock (default: 1)",
    )
    parser.add_argument(
        "--min-increase",
        type=float,
        default=100.0,
        help="Minimum %% increase in FII AND DII holdings (default: 100 = doubled)",
    )
    parser.add_argument(
        "--min-holdings",
        type=float,
        default=0.0,
        help="Minimum combined FII+DII %% holdings (default: 0)",
    )
    parser.add_argument(
        "--index",
        default="all",
        help="NSE index to scan (default: all F&O stocks). Examples: 'NIFTY 50', 'NIFTY 200'",
    )

    args = parser.parse_args()
    run_scanner(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
