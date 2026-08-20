"""
ML CHoCH Strategy -- OpenAlgo Python Strategy Host ready script.

Runs the ML CHoCH indicator on a schedule during market hours and places
orders based on signals. Supports both live and paper (sandbox) modes.

The model retrains from scratch each cycle using a rolling window of
historical CHoCH events, ensuring no future data leaks into predictions.

Environment Variables:
    OPENALGO_API_KEY: Your OpenAlgo API key
    OPENALGO_HOST: OpenAlgo server URL (default: http://127.0.0.1:5000)
    SYMBOL: Trading symbol (default: NIFTY)
    EXCHANGE: Exchange code (default: NSE_INDEX)
    INTERVAL: Candle interval (default: 15m)
    PAPER_MODE: Set to "true" for sandbox/paper trading (default: true)
    LOOKBACK_DAYS: Days of history to fetch (default: 90)
    LOOKAHEAD: Lookahead window for ML labeling (default: 20)
    SWING_LEN: Pivot detection period (default: 5)
    ATR_LEN: ATR period (default: 14)
    MIN_SCORE: Minimum probability for valid signal (default: 65)
    N_TREES: Number of Random Forest trees (default: 100)
    MIN_EVENTS: Minimum events before model can predict (default: 10)
    WINDOW: Rolling window size in bars (default: 1500)
    QUANTITY: Order quantity (default: 1)
    PRODUCT: Order product type (default: MIS)
    SLEEP_SECONDS: Seconds between runs (default: 300)

Usage in OpenAlgo Python Strategy Host:
    Upload this file to /python page, configure environment variables,
    set start/stop times to market hours.
"""

import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Graceful shutdown flag
_shutdown = False


def _handle_sigterm(signum, frame):
    """Handle SIGTERM for graceful shutdown."""
    global _shutdown
    _shutdown = True
    print("[ML CHoCH] SIGTERM received, shutting down gracefully...")


signal.signal(signal.SIGTERM, _handle_sigterm)


def log(msg: str):
    """Print log message to stdout (required for OpenAlgo Strategy Host)."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [ML CHoCH] {msg}", flush=True)


def get_config() -> dict:
    """Read configuration from environment variables."""
    return {
        "api_key": os.getenv("OPENALGO_API_KEY", ""),
        "host": os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
        "symbol": os.getenv("SYMBOL", "NIFTY"),
        "exchange": os.getenv("EXCHANGE", "NSE_INDEX"),
        "interval": os.getenv("INTERVAL", "15m"),
        "paper_mode": os.getenv("PAPER_MODE", "true").lower() == "true",
        "lookback_days": int(os.getenv("LOOKBACK_DAYS", "90")),
        "lookahead": int(os.getenv("LOOKAHEAD", "20")),
        "swing_len": int(os.getenv("SWING_LEN", "5")),
        "atr_len": int(os.getenv("ATR_LEN", "14")),
        "min_score": float(os.getenv("MIN_SCORE", "65")),
        "n_trees": int(os.getenv("N_TREES", "100")),
        "min_events": int(os.getenv("MIN_EVENTS", "10")),
        "window": int(os.getenv("WINDOW", "1500")),
        "quantity": int(os.getenv("QUANTITY", "1")),
        "product": os.getenv("PRODUCT", "MIS"),
        "sleep_seconds": int(os.getenv("SLEEP_SECONDS", "300")),
        "model_path": os.getenv("MODEL_PATH", "choch_model.joblib"),
    }


def fetch_data(cfg: dict):
    """Fetch OHLCV data from OpenAlgo API."""
    import pandas as pd
    from openalgo import api

    client = api(api_key=cfg["api_key"], host=cfg["host"])

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=cfg["lookback_days"])

    df = client.history(
        symbol=cfg["symbol"],
        exchange=cfg["exchange"],
        interval=cfg["interval"],
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df


def place_order(cfg: dict, direction: bool, tp1: float):
    """Place an order via OpenAlgo API (live or paper mode).

    Args:
        cfg: Configuration dict
        direction: True for BUY (bullish), False for SELL (bearish)
        tp1: TP1 price for reference logging
    """
    from openalgo import api

    client = api(api_key=cfg["api_key"], host=cfg["host"])

    action = "BUY" if direction else "SELL"

    if cfg["paper_mode"]:
        log(f"PAPER MODE: Would place {action} {cfg['symbol']} "
            f"x{cfg['quantity']} @ market (TP1={tp1:.2f})")
        # Uncomment below for actual paper execution via analyzer:
        # result = client.placeorder(
        #     symbol=cfg["symbol"], exchange=cfg["exchange"],
        #     action=action, quantity=cfg["quantity"],
        #     pricetype="MARKET", product=cfg["product"],
        #     strategy="ml_choch",
        # )
    else:
        log(f"LIVE: Placing {action} {cfg['symbol']} "
            f"x{cfg['quantity']} @ market (TP1={tp1:.2f})")
        result = client.placeorder(
            symbol=cfg["symbol"], exchange=cfg["exchange"],
            action=action, quantity=cfg["quantity"],
            pricetype="MARKET", product=cfg["product"],
            strategy="ml_choch",
        )
        log(f"Order result: {result}")


def run_once(cfg: dict) -> bool:
    """Execute one cycle of the strategy.

    Each cycle:
    1. Fetch fresh OHLCV data
    2. Detect CHoCH events and extract features
    3. Label historical events with outcomes
    4. For each event chronologically: train RF on rolling window, predict, compute targets
    5. Return latest valid signal for order placement

    Returns:
        True if a signal was generated, False otherwise
    """
    import pandas as pd

    from custom_indicators.ml_choch.signal_generator import generate_signals

    # Fetch data
    log(f"Fetching {cfg['symbol']} ({cfg['exchange']}) {cfg['interval']}...")
    df = fetch_data(cfg)
    log(f"Loaded {len(df)} bars")

    if len(df) < 100:
        log("Insufficient data (need at least 100 bars)")
        return False

    # Run ML pipeline with rolling window training
    signals, model = generate_signals(
        df,
        lookahead=cfg["lookahead"],
        swing_len=cfg["swing_len"],
        atr_len=cfg["atr_len"],
        min_score=cfg["min_score"],
        n_trees=cfg["n_trees"],
        min_events=cfg["min_events"],
        window=cfg["window"],
        model_path=cfg["model_path"],
    )

    valid = [s for s in signals if s.is_valid]
    log(f"Signals: {len(signals)} total, {len(valid)} valid (>= {cfg['min_score']}%)")

    if valid:
        last = valid[-1]
        direction = "BULLISH" if last.direction else "BEARISH"
        log(f"Latest: {direction} {last.probability:.1f}% "
            f"TP1={last.tp1:.2f} TP2={last.tp2:.2f} TP3={last.tp3:.2f}")
        log(f"Rolling DB size: {last.db_size} events")

        # Place order based on the latest valid signal
        place_order(cfg, last.direction, last.tp1)
        return True
    else:
        log("No valid signals this cycle")
        return False


def main():
    """Main strategy loop."""
    global _shutdown

    cfg = get_config()

    if not cfg["api_key"]:
        log("ERROR: OPENALGO_API_KEY not set. Exiting.")
        sys.exit(1)

    mode = "PAPER" if cfg["paper_mode"] else "LIVE"
    log(f"Starting ML CHoCH Strategy ({mode} mode)")
    log(f"Symbol: {cfg['symbol']} | Exchange: {cfg['exchange']} | Interval: {cfg['interval']}")
    log(f"Trees={cfg['n_trees']} | MinEvents={cfg['min_events']} | Window={cfg['window']}")
    log(f"Lookahead={cfg['lookahead']} | MinScore={cfg['min_score']}")
    log(f"Run interval: {cfg['sleep_seconds']}s")

    # Add project root to path for imports
    project_root = Path(__file__).resolve().parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    run_count = 0
    while not _shutdown:
        run_count += 1
        log(f"--- Run #{run_count} ---")

        try:
            run_once(cfg)
        except Exception as e:
            log(f"ERROR: {e}")

        if _shutdown:
            break

        log(f"Sleeping {cfg['sleep_seconds']}s...")
        # Sleep in small increments to allow SIGTERM to break through
        for _ in range(cfg["sleep_seconds"]):
            if _shutdown:
                break
            time.sleep(1)

    log("Strategy stopped.")


if __name__ == "__main__":
    main()
