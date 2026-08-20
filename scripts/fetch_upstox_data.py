"""Fetch Upstox company data on demand and populate the MCP file caches.

Thin CLI over ``mcp/upstox_company_data.py`` - the fetch-on-miss provider
that the ``company_snapshot`` MCP tool uses. It resolves the broker auth
token for the API key and asks the provider to refresh every requested
section, which owns the cache layout, per-kind TTLs and atomic writes. This
script only selects the kinds and prints the resulting envelope statuses.

The MCP server never needs this script: ``company_snapshot`` fetches on miss
itself. This CLI exists for manual warm-up, scheduled cache priming and
troubleshooting.

PRECONDITIONS:
  - Run inside the OpenAlgo venv:    uv run python scripts/fetch_upstox_data.py --symbol RELIANCE
  - .env must contain the same API_KEY_PEPPER used to encrypt the auth row
  - db/openalgo.db must contain an active Auth row (i.e. you've logged in today
    via the OpenAlgo UI - Indian broker tokens expire daily ~3:00 AM IST)
  - The broker master contract must be loaded (the backend does this at startup)

Usage:
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE --mode all
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE --mode all --force
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE --mode income-statement
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE --mode news
  uv run python scripts/fetch_upstox_data.py --symbol RELIANCE --mode history --interval D --days 365

Modes (default all):
  history            Historical candles -> mcp/cache/company_history/{SYMBOL}_{EXCHANGE}.json
  news               Upstox news (past 7 days) -> mcp/cache/company_news/{SYMBOL}_{EXCHANGE}.json
  income-statement   Annual consolidated P&L -> mcp/cache/company_fundamentals/{MODE}_{SYMBOL}_{EXCHANGE}.json
  balance-sheet      Annual consolidated balance sheet
  cash-flow          Annual consolidated cash flow
  key-ratios         Valuation and profitability ratios
  share-holdings     Promoter/FII/DII/mutual-fund quarterly holdings
  corporate-actions  Dividend/bonus/split/rights calendar
  company-profile    Company overview and contact details
  competitors        Peer companies for the same sector
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys

# Make sure we can import OpenAlgo's modules from repo root.
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

# Load .env so API_KEY_PEPPER is available before importing auth_db (which
# fails-fast if the pepper is missing).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(REPO_ROOT, ".env"))
except ImportError:
    pass  # python-dotenv not strictly required if API_KEY_PEPPER is already exported

if not os.getenv("API_KEY_PEPPER"):
    sys.stderr.write(
        "ERROR: API_KEY_PEPPER not set. Either source the .env that holds it,\n"
        "or export it manually before running this script.\n"
    )
    sys.exit(2)

# Import after env is loaded - auth_db.py hard-fails on a missing pepper.
from database.auth_db import get_auth_token_broker  # noqa: E402

_PROVIDER_PATH = os.path.join(REPO_ROOT, "mcp", "upstox_company_data.py")
_PROVIDER_MODULE = "openalgo_upstox_company_data"


def _load_provider():
    """Load ``mcp/upstox_company_data.py`` via the importlib file-path loader.

    mcp/ is not a Python package, so the provider is loaded from its file
    path (the same pattern the mcpserver tool registry uses for mcpserver).
    """
    spec = importlib.util.spec_from_file_location(_PROVIDER_MODULE, _PROVIDER_PATH)
    if spec is None or spec.loader is None:
        sys.stderr.write(f"ERROR: could not build import spec for {_PROVIDER_PATH}\n")
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[_PROVIDER_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _status_label(envelope: dict) -> str:
    """Extract the status string from a provider envelope."""
    status = envelope.get("status")
    if isinstance(status, dict):
        return status.get("status", "unknown")
    return str(status or "unknown")


def _count_label(kind: str, data: dict) -> str | None:
    """Best-effort record count for a section's data block."""
    if not isinstance(data, dict):
        return None
    if kind == "history":
        count = data.get("candle_count")
    elif kind == "news":
        count = data.get("news_count")
    else:
        count = data.get("count")
    return f"{count} records" if isinstance(count, int) else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Upstox company data (history/news/8 fundamentals) on demand "
            "and populate the MCP file caches."
        )
    )
    parser.add_argument("--symbol", required=True, help="Trading symbol (e.g., RELIANCE)")
    parser.add_argument("--exchange", default="NSE", help="Exchange (default NSE)")
    parser.add_argument(
        "--mode",
        default="all",
        help=(
            "Section to fetch: history, news, one of the 8 fundamentals kinds "
            "(income-statement, balance-sheet, cash-flow, key-ratios, "
            "share-holdings, corporate-actions, company-profile, competitors), "
            "or 'all' (default all)"
        ),
    )
    parser.add_argument(
        "--interval",
        default="D",
        help="Candle interval for --mode history (D, W, M, 1m..60m, 1h..4h)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of history to fetch on a miss (default 365)",
    )
    parser.add_argument(
        "--apikey", default=None, help="OpenAlgo API key (default: OPENALGO_API_KEY env)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Skip the freshness check and refetch unconditionally",
    )
    args = parser.parse_args()

    provider = _load_provider()
    if provider is None:
        return 2

    if args.mode == "all":
        kinds = list(provider.ALL_KINDS)
    elif args.mode in provider.ALL_KINDS:
        kinds = [args.mode]
    else:
        sys.stderr.write(
            f"ERROR: unknown mode {args.mode!r}. Expected one of: all, "
            f"{', '.join(provider.ALL_KINDS)}.\n"
        )
        return 2

    api_key = args.apikey or os.getenv("OPENALGO_API_KEY")
    if not api_key:
        sys.stderr.write(
            "ERROR: no API key. Pass --apikey or set OPENALGO_API_KEY in .env.\n"
        )
        return 2

    auth_token, broker = get_auth_token_broker(api_key)
    if not auth_token:
        sys.stderr.write(
            "ERROR: no active broker token found for this API key. Log in via "
            "the OpenAlgo UI to create one (broker tokens expire daily ~3:00 AM IST).\n"
        )
        return 1
    if broker and broker.lower() != "upstox":
        sys.stderr.write(
            f"WARNING: active broker is '{broker}', but company data is fetched "
            "from Upstox. Instrument resolution may fail.\n"
        )

    misses = 0
    for kind in kinds:
        envelope = provider.get_section(
            kind,
            args.symbol,
            args.exchange,
            api_key=api_key,
            interval=args.interval,
            days=args.days,
            force=args.force,
        )
        status_label = _status_label(envelope)
        count = _count_label(kind, envelope.get("data") or {})
        detail = f", {count}" if count else ""
        print(
            f"{kind}: {status_label} (freshness={envelope.get('freshness')}, "
            f"source={envelope.get('source')}){detail}"
        )
        data = envelope.get("data")
        if isinstance(data, dict) and data.get("note"):
            print(f"  note: {data['note']}")
        if status_label == "cache_miss":
            misses += 1

    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
