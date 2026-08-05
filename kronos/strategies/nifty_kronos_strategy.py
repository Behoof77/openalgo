"""
Nifty Kronos Strategy -- Nifty Futures + ATM Options Hedge

Uses the Kronos time-series model fused with technical indicators (RSI,
Bollinger Bands, ATR, ADX) and optionally option-market sentiment (PCR,
IV skew) to generate directional signals on NIFTY 3-minute bars.

Trade logic
-----------
BUY signal   -> Long 1 Nifty future + Buy 1 ATM Put  (current expiry)
SELL signal  -> Short 1 Nifty future + Buy 1 ATM Call (current expiry)
Target: +40 points on the futures position.
Stoploss: -20 points on the futures position.

The ATM option leg is a gap hedge -- it protects against overnight / gap
moves but decays with time.  In the live runner the option is bought at
MARKET for quick fill; the futures leg enters at LIMIT.

Usage
-----
    # Backtest on 3m NIFTY index data
    python -m kronos.strategies.nifty_kronos_strategy --mode backtest

    # Backtest on different interval
    python -m kronos.strategies.nifty_kronos_strategy --mode backtest --interval 5m

    # Live with dry-run (no real orders)
    python -m kronos.strategies.nifty_kronos_strategy --mode live --dry-run

    # Live full execution (analyzer mode first, then --no-analyzer for live)
    python -m kronos.strategies.nifty_kronos_strategy --mode live --no-dry-run --no-analyzer
"""

from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

from kronos.backtest.metrics import BacktestMetricsCalculator, BacktestResult
from kronos.broker.bridge import OpenAlgoBrokerBridge, PlaceOrderRequest
from kronos.data.openalgo_provider import OpenAlgoDataProvider
from kronos.data.option_provider import OptionDataProvider
from kronos.data.ws_client import KronosWebSocketClient
from kronos.features.indicators import compute_all_technicals
from kronos.features.signal_fusion import (
    FusionFactors,
    SignalFusionConfig,
    fuse_signals,
)
from kronos.client import KronosClient, KronosPrediction
from kronos.utils.config import KronosConfig
from kronos.utils.helpers import get_logger, ohlcv_df_to_kronos_df, setup_logging

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NIFTY_LOT_SIZE = 50             # 1 futures lot = 50 units
NIFTY_INDEX_SYMBOL = "NIFTY"
NIFTY_INDEX_EXCHANGE = "NSE"
NIFTY_FUT_EXCHANGE = "NFO"

TARGET_POINTS = 40.0
STOPLOSS_POINTS = 20.0

# Backtest cost modelling -------------------------------------------------
# The ATM option premium is *not* modelled dynamically because we don't have
# historical option prices in the NIFTY index data.  We deduct a fixed per-
# trade cost as a conservative approximation.  Adjust this estimate based on
# prevailing ATM implied vol at the time of the backtest.
BACKTEST_OPTION_COST = 250.0    # fixed INR per leg (entry only)
BACKTEST_FUTURES_STT = 0.0001   # 0.01 % STT on sell side of futures
BACKTEST_BROKERAGE = 20.0       # INR per order (each entry + exit leg)

DEFAULT_INTERVAL = "3m"         # 3-minute bars
BACKTEST_DEFAULT_CAPITAL = 100000.0

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class NiftyTrade:
    """A single futures + options hedge trade."""

    entry_time: datetime
    direction: str                     # "LONG" | "SHORT"
    entry_price: float
    target_price: float
    stoploss_price: float
    confidence: float
    fused_score: float = 0.0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: str = ""
    future_pnl: float = 0.0
    option_cost: float = 0.0
    transaction_costs: float = 0.0     # STT + brokerage
    net_pnl: float = 0.0

    @property
    def bars_held(self) -> int:
        if self.exit_time and self.entry_time:
            return max(1, int((self.exit_time - self.entry_time).total_seconds() / 180))
        return 0


@dataclass
class BacktestSummary:
    """Aggregated backtest performance."""

    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    total_future_pnl: float = 0.0
    total_option_cost: float = 0.0
    total_txn_costs: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_losses: int = 0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    cagr: float = 0.0

    # Benchmark
    bench_total_return: float = 0.0
    bench_sharpe: float = 0.0
    bench_max_dd: float = 0.0
    bench_cagr: float = 0.0

    trades: list[NiftyTrade] = field(default_factory=list)
    equity_curve: pd.Series | None = None


# ---------------------------------------------------------------------------
# Signal handler for graceful shutdown
# ---------------------------------------------------------------------------

SHUTDOWN_REQUESTED = False


def _handle_sigterm(signum: int, _frame: Any) -> None:
    global SHUTDOWN_REQUESTED
    logger.warning("Shutdown signal received. Finishing current cycle...")
    SHUTDOWN_REQUESTED = True


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class NiftyKronosStrategy:
    """Nifty futures + ATM options hedge strategy.

    Parameters
    ----------
    config:
        Application config (reads from ``.env`` by default).
    mode:
        ``"backtest"`` or ``"live"``.
    analyzer_mode:
        When True (default), the live runner toggles OpenAlgo analyzer
        mode so orders are simulated rather than sent to the broker.
    """

    def __init__(
        self,
        config: KronosConfig | None = None,
        mode: str = "backtest",
        analyzer_mode: bool = True,
    ):
        self.config = config or KronosConfig.from_env()
        self.mode = mode
        self.analyzer_mode = analyzer_mode
        self._kronos_client: KronosClient | None = None
        self._fusion_cfg = SignalFusionConfig()
        self._ws: KronosWebSocketClient | None = None

        # Live state
        self._current_trade: NiftyTrade | None = None
        self._trade_history: list[NiftyTrade] = []
        self._bridge: OpenAlgoBrokerBridge | None = None
        self._option_provider: OptionDataProvider | None = None
        self._data_provider: OpenAlgoDataProvider | None = None

    # ------------------------------------------------------------------
    # Predict pipeline (shared by backtest + live)
    # ------------------------------------------------------------------

    def _fused_predict(
        self, hist_df: pd.DataFrame
    ) -> tuple[int, float, dict[str, Any]]:
        """Run Kronos + technical fusion on *hist_df*.

        Returns (signal, conviction, breakdown).
        """
        # 1. Kronos factors (via HTTP inference server)
        kronos_signal = 0
        kronos_conf = 0.0
        if self._kronos_client is not None:
            try:
                kdf = ohlcv_df_to_kronos_df(hist_df)
                if len(kdf) > self.config.max_context:
                    kdf = kdf.iloc[-self.config.max_context :]

                pred = self._kronos_client.predict(
                    df=kdf,
                    pred_len=self.config.pred_len,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    sample_count=self.config.sample_count,
                )
                kronos_signal = pred.signal
                kronos_conf = pred.confidence
            except Exception:
                logger.exception("Kronos prediction failed, proceeding without it")
                kronos_signal = 0
                kronos_conf = 0.0

        # 2. Technical factors (only when Kronos has a view)
        tech = compute_all_technicals(hist_df) if kronos_signal != 0 else None

        # 3. Fuse
        factors = FusionFactors(
            kronos_signal=kronos_signal, kronos_confidence=kronos_conf, technicals=tech
        )
        signal, conviction, breakdown = fuse_signals(factors, self._fusion_cfg)
        return signal, conviction, breakdown

    # ------------------------------------------------------------------
    # Backtest mode
    # ------------------------------------------------------------------

    def run_backtest(
        self,
        start_date: str = "2024-01-01",
        end_date: str | None = None,
        interval: str = DEFAULT_INTERVAL,
        initial_capital: float = BACKTEST_DEFAULT_CAPITAL,
    ) -> BacktestSummary:
        """Walk-forward backtest with realistic Indian costs and NIFTY benchmark."""
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        logger.info(
            "Backtest: %s..%s interval=%s capital=%.0f",
            start_date, end_date, interval, initial_capital,
        )

        provider = OpenAlgoDataProvider(self.config)
        df = provider.fetch_history(
            symbol=NIFTY_INDEX_SYMBOL,
            exchange=NIFTY_INDEX_EXCHANGE,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
        )
        provider.close()

        if df.empty:
            logger.error("No data returned.")
            return BacktestSummary()

        logger.info("Fetched %d bars", len(df))

        min_bars = self.config.max_context + 10
        if len(df) < min_bars:
            logger.error("Need >= %d bars, got %d", min_bars, len(df))
            return BacktestSummary()

        # Connect to Kronos inference server
        kronos_host = self.config.openalgo_host  # or a dedicated KRONOS_SERVER env var
        try:
            self._kronos_client = KronosClient(base_url=kronos_host)
            # Verify server is alive
            h = self._kronos_client.health()
            logger.info("Kronos server connected: model_loaded=%s device=%s", h.get("model_loaded"), h.get("device"))
        except Exception as exc:
            logger.warning("Kronos server unreachable (%s) -- technicals-only signals", exc)
            self._kronos_client = None

        # Walk-forward state
        capital = initial_capital
        trades: list[NiftyTrade] = []
        open_trade: NiftyTrade | None = None
        equity_curve: list[tuple[pd.Timestamp, float]] = []
        last_signal = 0

        for window_end in range(self.config.max_context, len(df)):
            hist = df.iloc[:window_end]
            current_bar = df.iloc[window_end]
            current_price = float(current_bar["close"])
            current_time = current_bar.name

            signal, conviction, breakdown = self._fused_predict(hist)

            if window_end % 200 == 0:
                logger.debug(
                    "Bar %d/%d: signal=%d conv=%.2f",
                    window_end, len(df) - self.config.max_context, signal, conviction,
                )

            # -- Check SL/TP on existing trade --
            if open_trade is not None:
                hit_sl, hit_tp = False, False
                bar_high = float(current_bar["high"])
                bar_low = float(current_bar["low"])

                if open_trade.direction == "LONG":
                    if bar_low <= open_trade.stoploss_price:
                        exit_px = open_trade.stoploss_price
                        open_trade.exit_price = exit_px
                        open_trade.exit_reason = "stoploss"
                        open_trade.exit_time = current_time
                        hit_sl = True
                    elif bar_high >= open_trade.target_price:
                        exit_px = open_trade.target_price
                        open_trade.exit_price = exit_px
                        open_trade.exit_reason = "target"
                        open_trade.exit_time = current_time
                        hit_tp = True
                else:  # SHORT
                    if bar_high >= open_trade.stoploss_price:
                        exit_px = open_trade.stoploss_price
                        open_trade.exit_price = exit_px
                        open_trade.exit_reason = "stoploss"
                        open_trade.exit_time = current_time
                        hit_sl = True
                    elif bar_low <= open_trade.target_price:
                        exit_px = open_trade.target_price
                        open_trade.exit_price = exit_px
                        open_trade.exit_reason = "target"
                        open_trade.exit_time = current_time
                        hit_tp = True

                if hit_sl or hit_tp:
                    _close_trade(open_trade, NIFTY_LOT_SIZE, BACKTEST_OPTION_COST, BACKTEST_FUTURES_STT, BACKTEST_BROKERAGE)
                    capital += open_trade.net_pnl
                    trades.append(open_trade)
                    open_trade = None
                    last_signal = 0

            # -- Check reversal --
            if open_trade is not None and signal != 0 and signal != last_signal and last_signal != 0:
                direction = open_trade.direction
                entry = open_trade.entry_price
                fp = (current_price - entry) * NIFTY_LOT_SIZE if direction == "LONG" else (entry - current_price) * NIFTY_LOT_SIZE
                open_trade.exit_price = current_price
                open_trade.exit_time = current_time
                open_trade.exit_reason = "signal_reversal"
                open_trade.future_pnl = fp
                open_trade.option_cost = BACKTEST_OPTION_COST
                open_trade.transaction_costs = _calc_txn_costs(fp, BACKTEST_FUTURES_STT, BACKTEST_BROKERAGE)
                open_trade.net_pnl = fp - BACKTEST_OPTION_COST - open_trade.transaction_costs
                capital += open_trade.net_pnl
                trades.append(open_trade)
                open_trade = None
                last_signal = 0

            # -- Enter new trade --
            if open_trade is None and signal != 0 and conviction >= 0.25:
                direction = "LONG" if signal == 1 else "SHORT"
                target = current_price + (TARGET_POINTS if signal == 1 else -TARGET_POINTS)
                sl = current_price - (STOPLOSS_POINTS if signal == 1 else -STOPLOSS_POINTS)

                open_trade = NiftyTrade(
                    entry_time=current_time,
                    direction=direction,
                    entry_price=current_price,
                    target_price=target,
                    stoploss_price=sl,
                    confidence=conviction,
                    fused_score=breakdown.get("fused_score", 0.0),
                )
                last_signal = signal

            # Record equity
            unrealised = 0.0
            if open_trade is not None:
                upnl = (current_price - open_trade.entry_price) * NIFTY_LOT_SIZE if open_trade.direction == "LONG" else (open_trade.entry_price - current_price) * NIFTY_LOT_SIZE
                unrealised = upnl - BACKTEST_OPTION_COST
            equity_curve.append((current_time, capital + unrealised))

        # Close any remaining trade
        if open_trade is not None:
            last_bar = df.iloc[-1]
            last_price = float(last_bar["close"])
            _close_trade(
                open_trade, NIFTY_LOT_SIZE, BACKTEST_OPTION_COST,
                BACKTEST_FUTURES_STT, BACKTEST_BROKERAGE,
                exit_price=last_price, exit_time=last_bar.name, exit_reason="end_of_data",
            )
            capital += open_trade.net_pnl
            trades.append(open_trade)

        # Build equity curve
        eq_df = pd.DataFrame(equity_curve, columns=["time", "equity"]).set_index("time")
        equity_series = eq_df["equity"]

        # Compute metrics
        summary = self._compute_summary(trades, equity_series, initial_capital)

        # Benchmark: NIFTY buy-and-hold
        bench_df = df.copy()
        if not bench_df.empty:
            bench_returns = bench_df["close"].pct_change().dropna()
            bench_equity = initial_capital * (1 + bench_returns).cumprod()
            bench_equity = pd.concat([pd.Series([initial_capital], index=[bench_df.index[0]]), bench_equity]).sort_index()
            try:
                bench_calc = BacktestMetricsCalculator(
                    equity_curve=bench_equity, trades=[], initial_capital=initial_capital,
                )
                bench_res = bench_calc.compute_all()
                summary.bench_total_return = bench_res.total_return_pct
                summary.bench_sharpe = bench_res.sharpe_ratio
                summary.bench_max_dd = bench_res.max_drawdown_pct
                summary.bench_cagr = bench_res.cagr
            except Exception:
                logger.warning("Benchmark computation failed (non-critical)")

        logger.info(
            "Backtest done: %d trades, net P&L=%.0f, final=%.0f",
            summary.total_trades, summary.net_pnl, capital,
        )
        # Export trades
        _export_trades(trades, self.mode)
        return summary

    # ------------------------------------------------------------------
    # Live mode
    # ------------------------------------------------------------------

    def run_live(
        self,
        dry_run: bool = True,
        interval: str = DEFAULT_INTERVAL,
    ) -> None:
        """Run the live trading loop with WebSocket price monitoring.

        Parameters
        ----------
        dry_run:
            Log orders without placing them.
        interval:
            Data interval for prediction candles.
        """
        signal.signal(signal.SIGTERM, _handle_sigterm)
        signal.signal(signal.SIGINT, _handle_sigterm)

        # Initialise providers
        self._bridge = OpenAlgoBrokerBridge(self.config)
        self._data_provider = OpenAlgoDataProvider(self.config)
        self._option_provider = OptionDataProvider(self.config)

        # Start WebSocket for real-time LTP
        self._ws = KronosWebSocketClient(self.config)
        self._ws.set_on_ltp(self._on_ws_ltp)
        self._ws.subscribe(NIFTY_INDEX_SYMBOL, NIFTY_INDEX_EXCHANGE)
        self._ws.start()
        logger.info("WebSocket client started for NIFTY LTP")

        # Toggle analyzer mode if requested
        if self.analyzer_mode:
            self._toggle_analyzer(True)

        # Connect to Kronos inference server
        try:
            self._kronos_client = KronosClient(base_url=kronos_host)
            h = self._kronos_client.health()
            logger.info(
                "Kronos server connected: model_loaded=%s device=%s",
                h.get("model_loaded"), h.get("device"),
            )
        except Exception as exc:
            logger.error("Kronos server unreachable (%s). Live mode requires Kronos. Aborting.", exc)
            self._cleanup()
            return

        logger.info(
            "Live runner started (dry_run=%s analyzer=%s interval=%s poll=%ds)",
            dry_run, self.analyzer_mode, interval, self.config.poll_interval_seconds,
        )

        cycle = 0
        while not SHUTDOWN_REQUESTED:
            cycle += 1
            try:
                self._live_cycle(cycle, interval, dry_run)
            except Exception:
                logger.exception("Live cycle %d failed", cycle)

            if SHUTDOWN_REQUESTED:
                break
            logger.debug("Sleeping %d s...", self.config.poll_interval_seconds)
            time.sleep(self.config.poll_interval_seconds)

        # Shutdown
        logger.info("Shutdown complete. %d trades executed.", len(self._trade_history))
        if self.analyzer_mode:
            self._toggle_analyzer(False)
        self._cleanup()

    # ------------------------------------------------------------------
    # Live helpers
    # ------------------------------------------------------------------

    def _live_cycle(self, cycle: int, interval: str, dry_run: bool) -> None:
        """Single live cycle: fetch data, predict, manage position."""
        now = datetime.now()
        start = (now - timedelta(days=max(30, int(self.config.max_context * 0.01)))).strftime("%Y-%m-%d")
        end = now.strftime("%Y-%m-%d")

        # 1. Fetch data (enough lookback for Kronos context)
        df = self._data_provider.fetch_history(
            symbol=NIFTY_INDEX_SYMBOL,
            exchange=NIFTY_INDEX_EXCHANGE,
            interval=interval,
            start_date=start,
            end_date=end,
            source="api",
        )
        if df.empty:
            logger.warning("No data fetched. Skipping.")
            return

        # 2. Get current price from WebSocket (falls back to last bar close)
        ws_price = self._ws.latest_ltp(NIFTY_INDEX_SYMBOL, NIFTY_INDEX_EXCHANGE)
        current_price = ws_price if ws_price is not None else float(df["close"].iloc[-1])

        # 3. Fused signal using *penultimate* bar (forming bar convention -- iloc[-2])
        df_for_pred = df.iloc[:-2] if len(df) > 2 else df.iloc[:-1] if len(df) > 1 else df
        signal, conviction, breakdown = self._fused_predict(df_for_pred)

        logger.info(
            "[Cycle %d] NIFTY=%.0f signal=%d conv=%.3f",
            cycle, current_price, signal, conviction,
        )

        # 4. Fetch option sentiment (non-critical)
        sentiment = None
        try:
            expiry_normalized = ""
            if self._option_provider:
                expiry_normalized = self._option_provider.get_normalized_expiry(NIFTY_INDEX_SYMBOL, "NFO")
            sentiment = self._option_provider.compute_sentiment(
                underlying=NIFTY_INDEX_SYMBOL,
                exchange="NFO",
                spot_price=current_price,
                expiry_date=expiry_normalized if expiry_normalized else "",
            ) if self._option_provider else None
            if sentiment:
                logger.info(
                    "Option: PCR=%.2f IV_skew=%.1f ATM=%.0f expiry=%s",
                    sentiment.pcr, sentiment.iv_skew, sentiment.atm_strike,
                    sentiment.near_expiry,
                )
        except Exception:
            logger.warning("Option data fetch failed (non-critical)")

        # 5. Check SL/TP on current trade
        if self._current_trade is not None:
            self._check_live_sl_tp(current_price, dry_run)
            if self._current_trade is not None:
                # Signal reversed?
                current_dir = 1 if self._current_trade.direction == "LONG" else -1
                if signal != 0 and signal != current_dir and conviction >= 0.3:
                    self._close_live_position(current_price, "signal_reversal", dry_run)

        # 6. Open new position
        if self._current_trade is None and signal != 0 and conviction >= 0.3:
            self._open_live_position(signal, current_price, sentiment, dry_run)

    def _open_live_position(
        self,
        signal: int,
        current_price: float,
        sentiment: Any,
        dry_run: bool,
    ) -> None:
        """Resolve symbols, preview, confirm, and place entry orders."""
        direction = "LONG" if signal == 1 else "SHORT"
        target = current_price + (TARGET_POINTS if signal == 1 else -TARGET_POINTS)
        sl = current_price - (STOPLOSS_POINTS if signal == 1 else -STOPLOSS_POINTS)

        # -- Resolve symbols using API --
        try:
            expiry_date = ""
            if sentiment and sentiment.near_expiry:
                expiry_date = sentiment.near_expiry
            if not expiry_date and self._option_provider:
                expiry_date = self._option_provider.get_normalized_expiry(
                    NIFTY_INDEX_SYMBOL, "NFO"
                )

            option_type = "PE" if signal == 1 else "CE"

            # Resolve ATM option symbol
            opt_res = self._option_provider.resolve_option_symbol(
                underlying=NIFTY_INDEX_SYMBOL,
                exchange=NIFTY_FUT_EXCHANGE,
                expiry_date=expiry_date,
                offset="ATM",
                option_type=option_type,
            ) if self._option_provider else {"status": "error"}

            if opt_res.get("status") != "success":
                logger.error("Option symbol resolution failed: %s", opt_res.get("message", "unknown"))
                return
            opt_symbol = opt_res["symbol"]
            opt_lotsize = int(opt_res.get("lotsize", 50))
            underlying_ltp = float(opt_res.get("underlying_ltp", current_price))

            # Build futures symbol
            if expiry_date:
                fut_symbol = f"{NIFTY_INDEX_SYMBOL}{expiry_date}FUT"
            else:
                logger.error("No expiry date available -- cannot build futures symbol")
                return

            # Validate futures symbol
            fut_res = self._option_provider.resolve_futures_symbol(
                underlying=NIFTY_INDEX_SYMBOL, exchange=NIFTY_FUT_EXCHANGE,
            ) if self._option_provider else {"status": "error"}
            if not isinstance(fut_res, dict) or fut_res.get("status") != "success":
                logger.warning("Futures symbol validation returned: %s", fut_res)
                # Proceed anyway -- the symbol format is standard
        except Exception:
            logger.exception("Symbol resolution failed")
            return

        # -- Preview --
        notional = current_price * NIFTY_LOT_SIZE
        logger.info(
            "ORDER PREVIEW --- %s %d lot NIFTY FUT @ LIMIT ~%.0f (notional Rs %.0f)",
            direction, NIFTY_LOT_SIZE, current_price, notional,
        )
        logger.info("  Futures: %s @ ~%.0f", fut_symbol, current_price)
        logger.info("  Option:  BUY 1 %s @ MARKET (ATM %s)", opt_symbol, option_type)
        logger.info("  Target: %.0f  Stoploss: %.0f", target, sl)
        logger.info("  Expiry: %s", expiry_date)

        # -- Confirm (skip in dry-run) --
        if not dry_run:
            try:
                ans = input("Place these orders? [y/N] ").strip().lower()
            except EOFError:
                ans = "n"
            if ans != "y":
                logger.info("Entry cancelled by user.")
                return

        # -- Place orders --
        if dry_run:
            logger.info("[DRY] %s %d %s @ LIMIT %.0f", "BUY" if direction == "LONG" else "SELL", NIFTY_LOT_SIZE, fut_symbol, current_price)
            logger.info("[DRY] BUY %d %s @ MARKET", opt_lotsize, opt_symbol)
            self._current_trade = NiftyTrade(
                entry_time=datetime.now(),
                direction=direction,
                entry_price=current_price,
                target_price=target,
                stoploss_price=sl,
                confidence=0.0,
            )
            return

        # Live execution
        bridge = self._bridge
        assert bridge is not None

        fut_action = "BUY" if direction == "LONG" else "SELL"
        fut_req = PlaceOrderRequest(
            strategy="kronos",
            exchange=NIFTY_FUT_EXCHANGE,
            symbol=fut_symbol,
            action=fut_action,
            quantity=NIFTY_LOT_SIZE,
            product="MIS",
            pricetype="LIMIT",
            price=current_price,
        )
        fut_res = bridge.place_order(fut_req)
        logger.info("Futures order: %s", fut_res)
        if not fut_res.success:
            logger.error("Futures order failed. Aborting entry.")
            return

        opt_req = PlaceOrderRequest(
            strategy="kronos",
            exchange=NIFTY_FUT_EXCHANGE,
            symbol=opt_symbol,
            action="BUY",
            quantity=opt_lotsize,
            product="MIS",
            pricetype="MARKET",
        )
        opt_res = bridge.place_order(opt_req)
        logger.info("Option order: %s", opt_res)

        self._current_trade = NiftyTrade(
            entry_time=datetime.now(),
            direction=direction,
            entry_price=current_price,
            target_price=target,
            stoploss_price=sl,
            confidence=0.0,
        )

    def _check_live_sl_tp(self, current_price: float, dry_run: bool) -> None:
        """Check if SL or TP has been hit (WebSocket prices are checked each cycle)."""
        t = self._current_trade
        if t is None:
            return

        hit, reason = False, ""
        if t.direction == "LONG":
            if current_price <= t.stoploss_price:
                hit, reason = True, "stoploss"
            elif current_price >= t.target_price:
                hit, reason = True, "target"
        else:
            if current_price >= t.stoploss_price:
                hit, reason = True, "stoploss"
            elif current_price <= t.target_price:
                hit, reason = True, "target"

        if hit:
            self._close_live_position(current_price, reason, dry_run)

    def _close_live_position(self, exit_price: float, reason: str, dry_run: bool) -> None:
        """Close current position."""
        t = self._current_trade
        if t is None:
            return

        t.exit_time = datetime.now()
        t.exit_price = exit_price
        t.exit_reason = reason
        fp = (exit_price - t.entry_price) * NIFTY_LOT_SIZE if t.direction == "LONG" else (t.entry_price - exit_price) * NIFTY_LOT_SIZE
        t.future_pnl = fp
        t.option_cost = BACKTEST_OPTION_COST
        t.transaction_costs = _calc_txn_costs(fp, BACKTEST_FUTURES_STT, BACKTEST_BROKERAGE)
        t.net_pnl = fp - BACKTEST_OPTION_COST - t.transaction_costs

        logger.info(
            "CLOSE %s @ %.0f | reason=%s | fut P&L=%.0f | net=%.0f",
            t.direction, exit_price, reason, fp, t.net_pnl,
        )

        if not dry_run and self._bridge is not None:
            bridge = self._bridge
            close_action = "SELL" if t.direction == "LONG" else "BUY"
            logger.info("  Closing futures: %s %d", close_action, NIFTY_LOT_SIZE)
            logger.info("  Closing option leg (sell at MARKET)")

        self._trade_history.append(t)
        self._current_trade = None

    # ------------------------------------------------------------------
    # WebSocket callback
    # ------------------------------------------------------------------

    def _on_ws_ltp(self, symbol: str, exchange: str, ltp: float) -> None:
        """Called on every LTP tick from WebSocket."""
        # Check SL/TP intra-cycle if we have an active trade
        if self._current_trade is not None:
            self._check_live_sl_tp(ltp, False)

    # ------------------------------------------------------------------
    # Analyzer mode toggle
    # ------------------------------------------------------------------

    def _toggle_analyzer(self, activate: bool) -> None:
        """Toggle OpenAlgo analyzer mode on/off."""
        try:
            # Use httpx directly -- the bridge doesn't expose this
            import httpx
            url = f"{self.config.openalgo_host}/api/v1/analyzertoggle/"
            resp = httpx.post(
                url,
                json={"apikey": self.config.openalgo_api_key, "data": {"analyze_mode": activate}},
                timeout=10,
            )
            data = resp.json()
            if data.get("status") == "success":
                logger.info("Analyzer mode toggled to %s", activate)
            else:
                logger.warning("Analyzer toggle response: %s", data)
        except Exception as exc:
            logger.warning("Analyzer toggle failed: %s", exc)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        trades: list[NiftyTrade],
        equity_series: pd.Series,
        initial_capital: float,
    ) -> BacktestSummary:
        """Compute aggregate metrics."""
        summary = BacktestSummary()
        summary.trades = trades
        summary.equity_curve = equity_series
        summary.total_trades = len(trades)
        summary.total_future_pnl = sum(t.future_pnl for t in trades)
        summary.total_option_cost = sum(t.option_cost for t in trades)
        summary.total_txn_costs = sum(t.transaction_costs for t in trades)
        summary.net_pnl = sum(t.net_pnl for t in trades)

        winning = [t for t in trades if t.net_pnl > 0]
        losing = [t for t in trades if t.net_pnl <= 0]
        summary.winning_trades = len(winning)
        summary.losing_trades = len(losing)
        summary.win_rate = (len(winning) / max(len(trades), 1)) * 100

        if winning:
            summary.gross_profit = sum(t.net_pnl for t in winning)
            summary.avg_win = summary.gross_profit / len(winning)
        if losing:
            summary.gross_loss = abs(sum(t.net_pnl for t in losing))
            summary.avg_loss = summary.gross_loss / len(losing)
        if summary.gross_loss > 0:
            summary.profit_factor = summary.gross_profit / max(summary.gross_loss, 1)

        # Max consecutive losses
        max_cl = cur_cl = 0
        for t in trades:
            if t.net_pnl <= 0:
                cur_cl += 1
                max_cl = max(max_cl, cur_cl)
            else:
                cur_cl = 0
        summary.max_consecutive_losses = max_cl

        # Sharpe, Sortino, drawdown, CAGR
        if len(equity_series) > 1:
            try:
                calc = BacktestMetricsCalculator(
                    equity_curve=equity_series, trades=[], initial_capital=initial_capital,
                )
                result = calc.compute_all()
                summary.sharpe = result.sharpe_ratio
                summary.sortino = result.sortino_ratio
                summary.max_drawdown_pct = result.max_drawdown_pct
                summary.cagr = result.cagr
            except Exception:
                pass

        return summary

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Shut down all providers and connections."""
        if self._ws:
            self._ws.stop()
        if self._data_provider:
            self._data_provider.close()
        if self._bridge:
            self._bridge.close()
        if self._option_provider:
            self._option_provider.close()
        _export_trades(self._trade_history, self.mode)


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _calc_txn_costs(future_pnl: float, stt_rate: float, brokerage: float) -> float:
    """Calculate transaction costs for a round-trip futures trade.

    STT is only on the sell side.  Brokerage is charged per order
    (entry futures + exit futures = 2 orders).  The option leg STT
    is included in the fixed BACKTEST_OPTION_COST for backtest mode.
    """
    sell_side = future_pnl if future_pnl > 0 else 0.0
    stt = sell_side * stt_rate
    brk = brokerage * 2  # entry + exit futures
    return stt + brk


def _close_trade(
    t: NiftyTrade,
    lot_size: int,
    option_cost: float,
    stt_rate: float,
    brokerage: float,
    exit_price: float | None = None,
    exit_time: Any = None,
    exit_reason: str = "",
) -> None:
    """Close a trade and compute costs."""
    if exit_price is not None:
        t.exit_price = exit_price
    if exit_time is not None:
        t.exit_time = exit_time
    if exit_reason:
        t.exit_reason = exit_reason

    direction = t.direction
    entry = t.entry_price
    exit_ = t.exit_price if t.exit_price is not None else entry

    if direction == "LONG":
        fp = (exit_ - entry) * lot_size
    else:
        fp = (entry - exit_) * lot_size

    t.future_pnl = fp
    t.option_cost = option_cost
    t.transaction_costs = _calc_txn_costs(fp, stt_rate, brokerage)
    t.net_pnl = fp - option_cost - t.transaction_costs


def _export_trades(trades: list[NiftyTrade], mode: str) -> None:
    """Export trade list to CSV next to the script."""
    if not trades:
        return
    script_dir = Path(__file__).resolve().parent
    fname = script_dir / f"trades_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "entry_time", "exit_time", "direction", "entry_price", "exit_price",
        "target_price", "stoploss_price", "exit_reason", "confidence",
        "future_pnl", "option_cost", "transaction_costs", "net_pnl",
    ]
    with open(fname, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in trades:
            w.writerow({
                "entry_time": t.entry_time.isoformat() if t.entry_time else "",
                "exit_time": t.exit_time.isoformat() if t.exit_time else "",
                "direction": t.direction,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price or 0,
                "target_price": t.target_price,
                "stoploss_price": t.stoploss_price,
                "exit_reason": t.exit_reason,
                "confidence": round(t.confidence, 3),
                "future_pnl": round(t.future_pnl, 2),
                "option_cost": round(t.option_cost, 2),
                "transaction_costs": round(t.transaction_costs, 2),
                "net_pnl": round(t.net_pnl, 2),
            })
    logger.info("Trades exported to %s", fname)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_backtest_report(summary: BacktestSummary, initial_capital: float) -> None:
    """Print a formatted backtest report with benchmark comparison."""
    line = "\u2500" * 62
    print(f"\n{line}")
    print("  NIFTY KRONOS BACKTEST REPORT")
    print(f"{line}")

    print(f"  {'':30s} {'Strategy':>10s} {'Buy & Hold':>10s}")
    print(f"  {'---':30s} {'---':>10s} {'---':>10s}")
    print(f"  {'Total Return':30s} {summary.net_pnl / initial_capital * 100:>9.1f}% {summary.bench_total_return:>9.1f}%")
    print(f"  {'CAGR':30s} {summary.cagr:>9.1f}% {summary.bench_cagr:>9.1f}%")
    print(f"  {'Sharpe Ratio':30s} {summary.sharpe:>10.2f} {summary.bench_sharpe:>10.2f}")
    print(f"  {'Max Drawdown':30s} {summary.max_drawdown_pct:>9.1f}% {summary.bench_max_dd:>9.1f}%")
    print(f"{line}")
    print(f"  Total Trades         : {summary.total_trades}")
    print(f"  Win Rate             : {summary.win_rate:.1f}%")
    print(f"  Winning / Losing     : {summary.winning_trades} / {summary.losing_trades}")
    print(f"  Profit Factor        : {summary.profit_factor:.2f}")
    print(f"  Avg Win / Avg Loss   : {summary.avg_win:.0f} / {summary.avg_loss:.0f}")
    print(f"  Max Consecutive Loss : {summary.max_consecutive_losses}")
    print(f"  Net P&L              : Rs {summary.net_pnl:>8,.0f}")
    print(f"  Future P&L           : Rs {summary.total_future_pnl:>8,.0f}")
    print(f"  Option Cost          : Rs {summary.total_option_cost:>8,.0f}")
    print(f"  Transaction Costs    : Rs {summary.total_txn_costs:>8,.0f}")
    print(f"  Sortino Ratio        : {summary.sortino:.2f}")
    print(f"{line}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nifty Kronos Strategy -- AI-driven futures + options hedge",
    )
    parser.add_argument(
        "--mode", choices=["backtest", "live"], default="backtest",
    )
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--interval", default=DEFAULT_INTERVAL)
    parser.add_argument(
        "--dry-run", action="store_true", default=True,
        help="Live: log orders without executing (default)",
    )
    parser.add_argument(
        "--no-dry-run", action="store_false", dest="dry_run",
        help="Live: execute real orders",
    )
    parser.add_argument(
        "--analyzer", action="store_true", default=True,
        help="Live: enable OpenAlgo analyzer mode (default, simulated orders)",
    )
    parser.add_argument(
        "--no-analyzer", action="store_false", dest="analyzer",
        help="Live: disable analyzer mode (real broker orders)",
    )
    parser.add_argument(
        "--capital", type=float, default=BACKTEST_DEFAULT_CAPITAL,
    )

    args = parser.parse_args()

    setup_logging()
    config = KronosConfig.from_env()
    strategy = NiftyKronosStrategy(
        config, mode=args.mode, analyzer_mode=args.analyzer,
    )

    if args.mode == "backtest":
        summary = strategy.run_backtest(
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval,
            initial_capital=args.capital,
        )
        print_backtest_report(summary, args.capital)
    else:
        strategy.run_live(
            dry_run=args.dry_run,
            interval=args.interval,
        )


if __name__ == "__main__":
    main()
