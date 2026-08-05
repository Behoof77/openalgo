"""
Live Trading Runner

Periodically fetches OHLCV data from OpenAlgo, runs Kronos model
predictions, and executes trades through the OpenAlgo broker bridge.
"""

from __future__ import annotations

import time
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime, timedelta, timezone

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import (
    ohlcv_df_to_kronos_df,
    kronos_prediction_to_signal,
    get_logger,
)
from kronos.data.openalgo_provider import OpenAlgoDataProvider
from kronos.broker.bridge import OpenAlgoBrokerBridge, PlaceOrderRequest
from kronos.model.model import (
    load_kronos_model,
    KronosPredictor,
    KronosUnavailableError,
)

logger = get_logger(__name__)


@dataclass
class LiveTradeRecord:
    """Record of a live trade."""

    entry_time: datetime
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    quantity: int
    confidence: float
    order_id: str = ""
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    status: str = "open"  # "open" | "closed" | "cancelled"


@dataclass
class LiveRunnerState:
    """Snapshot of runner state for status reporting."""

    is_running: bool = False
    current_position: Optional[LiveTradeRecord] = None
    last_prediction_time: Optional[datetime] = None
    last_signal: int = 0
    last_confidence: float = 0.0
    total_trades: int = 0
    total_pnl: float = 0.0
    poll_count: int = 0
    error_count: int = 0
    start_time: Optional[datetime] = None


class KronosLiveRunner:
    """Live trading runner that polls OpenAlgo and executes Kronos signals.

    Parameters
    ----------
    config:
        Kronos configuration (reads from .env by default).
    provider:
        OpenAlgo data provider.  Created from *config* if not given.
    bridge:
        OpenAlgo broker bridge.  Created from *config* if not given.
    """

    def __init__(
        self,
        config: KronosConfig,
        provider: Optional[OpenAlgoDataProvider] = None,
        bridge: Optional[OpenAlgoBrokerBridge] = None,
    ):
        self.config = config
        self.provider = provider or OpenAlgoDataProvider(config)
        self.bridge = bridge or OpenAlgoBrokerBridge(config)
        self.state = LiveRunnerState()
        self._running = False
        self._signal_fn: Optional[Callable] = None
        self._predictor: Optional[KronosPredictor] = None
        self._trading_symbol: str = ""
        self._trading_exchange: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        """Whether the runner loop is active."""
        return self._running

    def run_loop(
        self,
        symbol: str,
        exchange: str,
        interval: str = "5m",
        *,
        confidence_threshold: float = 0.7,
        max_context: Optional[int] = None,
        pred_len: Optional[int] = None,
        poll_interval: Optional[int] = None,
        dry_run: bool = True,
        signal_fn: Optional[Callable] = None,
    ) -> None:
        """Start the main polling loop.

        Parameters
        ----------
        symbol:
            Trading symbol (e.g. ``"NIFTY"``, ``"SBIN"``).
        exchange:
            Exchange code (e.g. ``"NSE"``, ``"NFO"``).
        interval:
            Candle interval (see :meth:`OpenAlgoDataProvider.fetch_history`
            for valid values).  Default ``"5m"``.
        confidence_threshold:
            Minimum prediction confidence to place a trade (0-1).
        max_context:
            Number of bars to pass to the model.  Default: config default.
        pred_len:
            Number of bars the model predicts ahead.  Default: config default.
        poll_interval:
            Seconds between polls.  Default: config ``poll_interval_seconds``.
        dry_run:
            If ``True`` (default), log orders but do not send them to the
            broker.  Set ``False`` for live execution.
        signal_fn:
            Optional override for signal generation.  Signature:
            ``signal_fn(pred_output, current_price) -> (signal, confidence)``.
        """
        max_context = max_context or self.config.max_context
        pred_len = pred_len or self.config.pred_len
        poll_interval = poll_interval or self.config.poll_interval_seconds
        self._signal_fn = signal_fn
        self._trading_symbol = symbol
        self._trading_exchange = exchange

        self.state = LiveRunnerState()
        self.state.is_running = True
        self.state.start_time = datetime.now(timezone.utc)
        self._running = True

        logger.info(
            "Live runner started: %s %s %s | dry=%s | poll=%ds | ctx=%d | pred=%d",
            symbol, exchange, interval, dry_run, poll_interval, max_context, pred_len,
        )

        try:
            while self._running:
                self.state.poll_count += 1
                try:
                    self._tick(
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        confidence_threshold=confidence_threshold,
                        max_context=max_context,
                        pred_len=pred_len,
                        dry_run=dry_run,
                    )
                except Exception:
                    logger.exception("Error during polling tick")
                    self.state.error_count += 1

                time.sleep(poll_interval)
        except KeyboardInterrupt:
            logger.info("Runner interrupted by user")
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Signal the runner loop to stop gracefully."""
        logger.info("Stopping live runner...")
        self._running = False

    def status(self) -> LiveRunnerState:
        """Return current runner state snapshot."""
        return self.state

    # ------------------------------------------------------------------
    # Model loading (cached)
    # ------------------------------------------------------------------

    def _get_predictor(self) -> KronosPredictor:
        """Return the cached KronosPredictor, loading it on first call."""
        if self._predictor is not None:
            return self._predictor

        logger.info("Loading Kronos model (first call)...")
        model = load_kronos_model(self.config)
        self._predictor = KronosPredictor(
            model=model,
            tokenizer=None,
            device=self.config.resolved_device,
        )
        return self._predictor

    # ------------------------------------------------------------------
    # Polling tick
    # ------------------------------------------------------------------

    def _tick(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        confidence_threshold: float,
        max_context: int,
        pred_len: int,
        dry_run: bool,
    ) -> None:
        """Single polling iteration."""
        # 1. Fetch recent data (need max_context + pred_len bars)
        bars_needed = max_context + pred_len
        lookback = self._lookback_for_interval(interval, bars_needed)
        start_date = (datetime.now() - lookback).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")

        df = self.provider.fetch_history(
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            source="api",
        )

        if df is None or len(df) < max_context:
            logger.warning(
                "Insufficient data: got %d rows, need %d",
                len(df) if df is not None else 0,
                max_context,
            )
            return

        # 2. Latest price reference
        current_price = float(df["close"].iloc[-1])
        current_time = df.index[-1] if hasattr(df.index, "dtype") else datetime.now(timezone.utc)
        self.state.last_prediction_time = current_time

        # 3. Prepare model input
        hist = df.iloc[-max_context:]
        kronos_df = ohlcv_df_to_kronos_df(hist)

        # 4. Run prediction (model is cached after first tick)
        try:
            predictor = self._get_predictor()
            pred_output = predictor.predict(
                df=kronos_df,
                x_timestamp=hist.index[-1],
                y_timestamp=None,
                pred_len=pred_len,
                T=self.config.temperature,
                top_p=self.config.top_p,
                sample_count=self.config.sample_count,
            )
        except KronosUnavailableError:
            logger.error(
                "Kronos model package not installed. "
                "Run: pip install git+https://github.com/shiyu-coder/Kronos.git"
            )
            self._running = False
            return
        except Exception:
            logger.exception("Kronos prediction failed")
            return

        # 5. Generate signal
        if self._signal_fn is not None:
            signal, confidence = self._signal_fn(pred_output, current_price)
        else:
            signal, confidence = kronos_prediction_to_signal(pred_output)

        self.state.last_signal = signal
        self.state.last_confidence = confidence

        pos_label = (
            self.state.current_position.direction
            if self.state.current_position
            else "NONE"
        )
        logger.info(
            "Tick #%d | signal=%d conf=%.3f price=%.2f position=%s",
            self.state.poll_count, signal, confidence, current_price, pos_label,
        )

        # 6. Position management
        if self.state.current_position is not None:
            self._manage_open_position(signal, confidence, confidence_threshold, current_price, dry_run)
        else:
            self._check_entry(signal, confidence, confidence_threshold, current_price, dry_run)

    def _manage_open_position(
        self,
        signal: int,
        confidence: float,
        threshold: float,
        current_price: float,
        dry_run: bool,
    ) -> None:
        """Decide whether to close the open position."""
        pos = self.state.current_position
        assert pos is not None

        should_exit = False
        reason = ""

        if signal == 0 or confidence < threshold:
            should_exit = True
            reason = "signal_dropped"
        elif pos.direction == "LONG" and signal == -1:
            should_exit = True
            reason = "reversal_signal"
        elif pos.direction == "SHORT" and signal == 1:
            should_exit = True
            reason = "reversal_signal"

        if should_exit:
            self._close_position(current_price, dry_run, reason=reason)

    def _check_entry(
        self,
        signal: int,
        confidence: float,
        threshold: float,
        current_price: float,
        dry_run: bool,
    ) -> None:
        """Open a position if the signal is strong enough."""
        if abs(signal) > 0 and confidence >= threshold:
            direction = "LONG" if signal == 1 else "SHORT"
            self._open_position(
                direction=direction,
                price=current_price,
                quantity=self._estimate_quantity(current_price),
                confidence=confidence,
                dry_run=dry_run,
            )

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def _open_position(
        self,
        direction: str,
        price: float,
        quantity: int,
        confidence: float,
        dry_run: bool,
    ) -> None:
        """Open a new position."""
        action = "BUY" if direction == "LONG" else "SELL"
        logger.info(
            "OPEN %s | price=%.2f qty=%d conf=%.3f dry=%s",
            direction, price, quantity, confidence, dry_run,
        )

        order_id = ""
        if not dry_run:
            req = PlaceOrderRequest(
                strategy="KronosLive",
                exchange=self._trading_exchange,
                symbol=self._trading_symbol,
                action=action,
                quantity=quantity,
                product="MIS",
                pricetype="MARKET",
            )
            resp = self.bridge.place_order(req)
            if resp:
                data = resp.get("data", {}) or {}
                order_id = data.get("orderid", "")
        else:
            order_id = f"dry_run_{datetime.now().strftime('%H%M%S')}"

        record = LiveTradeRecord(
            entry_time=datetime.now(timezone.utc),
            direction=direction,
            entry_price=price,
            quantity=quantity,
            confidence=confidence,
            order_id=order_id,
        )
        self.state.current_position = record
        self.state.total_trades += 1

    def _close_position(
        self,
        exit_price: float,
        dry_run: bool,
        reason: str = "",
    ) -> None:
        """Close the currently open position."""
        pos = self.state.current_position
        if pos is None:
            return

        if pos.direction == "LONG":
            pnl = (exit_price - pos.entry_price) * pos.quantity
        else:
            pnl = (pos.entry_price - exit_price) * pos.quantity

        logger.info(
            "CLOSE %s | entry=%.2f exit=%.2f pnl=%.2f reason=%s dry=%s",
            pos.direction, pos.entry_price, exit_price, pnl, reason, dry_run,
        )

        action = "SELL" if pos.direction == "LONG" else "BUY"
        if not dry_run:
            req = PlaceOrderRequest(
                strategy="KronosLive",
                exchange=self._trading_exchange,
                symbol=self._trading_symbol,
                action=action,
                quantity=pos.quantity,
                product="MIS",
                pricetype="MARKET",
            )
            self.bridge.place_order(req)

        pos.exit_time = datetime.now(timezone.utc)
        pos.exit_price = exit_price
        pos.pnl = pnl
        pos.status = "closed"

        self.state.total_pnl += pnl
        self.state.current_position = None

    @staticmethod
    def _estimate_quantity(price: float) -> int:
        """Estimate a conservative quantity (fixed small lot for safety)."""
        capital_per_trade = 20_000.0
        qty = int(capital_per_trade / price)
        return max(qty, 1)

    @staticmethod
    def _lookback_for_interval(interval: str, bars_needed: int) -> timedelta:
        """Calculate a conservative lookback for fetching enough history bars.

        For intraday intervals we need fewer days; for daily/weekly we need
        more.  Multiplies by 2 for a safety buffer against market holidays.
        """
        unit = interval[-1]
        mult = int(interval[:-1]) if len(interval) > 1 else 1

        if unit == "s":
            secs = mult * bars_needed * 2
        elif unit == "m":
            secs = mult * 60 * bars_needed * 2
        elif unit == "h":
            secs = mult * 3600 * bars_needed * 2
        elif unit == "D":
            secs = mult * 86400 * bars_needed * 2
        elif unit == "W":
            secs = mult * 604800 * bars_needed * 2
        elif unit == "M":
            secs = mult * 2592000 * bars_needed * 2
        else:
            secs = 86400 * 365 * 2  # 2 years fallback

        days = max(secs // 86400, 7)
        return timedelta(days=days)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def _cleanup(self) -> None:
        """Clean up on shutdown."""
        self._running = False
        self.state.is_running = False
        self._predictor = None  # allow GC of model
        elapsed = (
            datetime.now(timezone.utc) - self.state.start_time
            if self.state.start_time
            else datetime.min
        )
        logger.info(
            "Runner stopped | polls=%d errors=%d trades=%d pnl=%.2f elapsed=%s",
            self.state.poll_count,
            self.state.error_count,
            self.state.total_trades,
            self.state.total_pnl,
            elapsed,
        )
