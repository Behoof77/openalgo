#!/usr/bin/env python3
"""Discover NIFTY futures contracts via OpenAlgo API."""
import os, json, sys
sys.path.insert(0, "strategies/xgb_reversals_NIFTY")
from dotenv import find_dotenv, load_dotenv
from openalgo import api

load_dotenv(find_dotenv())
API_KEY = os.getenv("OPENALGO_API_KEY", "")
API_HOST = os.getenv("HOST_SERVER") or os.getenv("OPENALGO_HOST", "https://skopaq.duckdns.org")
client = api(api_key=API_KEY, host=API_HOST)

# Search for all NIFTY futures
print("=== NIFTY FUT === via search")
r = client.search(query="NIFTY FUT")
if isinstance(r, dict) and "data" in r:
    for item in r["data"]:
        print(json.dumps(item, indent=2))
else:
    print(r)

print("\n=== Expiry dates ===")
try:
    r = client.expiry(symbol="NIFTY")
    if isinstance(r, dict):
        print(json.dumps(r, indent=2)[:2000])
    else:
        print(r)
except Exception as e:
    print(f"expiry() error: {e}")
