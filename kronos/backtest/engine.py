"""
Kronos Backtesting Engine

Walk-forward backtesting engine driven by a user-supplied predict_fn.
The engine handles position tracking, PnL calculation, and equity curve
construction -- the caller provides model-specific prediction logic.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from datetime import datetime

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import get_logger
from kronos.backtest.metrics import BacktestResult, BacktestMetricsCalculator

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """A single completed trade."""

    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "LONG" | "SHORT"
    entry_price: float
    exit_price: float
    shares: float
    pnl: float
    pnl_pct: float
    confidence: float
    entry_reason: str = ""
    exit_reason: str = ""


@dataclass
class BacktestResult:
    """Aggregated backtest results.

    The first 9 fields are summary stats; ``trades`` and ``equity_curve``
    carry the full time-series data.
    """

    total_return_pct: float
    cagr_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_trade_pnl_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    trades: list[BacktestTrade]
    equity_curve: pd.Series | None = None
    config_snapshot: dict = field(default_factory=dict)


class KronosBacktestEngine:
    """Walk-forward backtesting engine.

    Usage
    -----
    >>> engine = KronosBacktestEngine(config)
    >>> def my_predict(hist_df):
    ...     # run Kronos model on hist_df, return (signal, confidence)
    ...     return (1, 0.85)
    >>> result = engine.run(df, predict_fn=my_predict)
    >>> print(result.sharpe_ratio, result.max_drawdown_pct)
    """

    def __init__(self, config: KronosConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        predict_fn: Callable[[pd.DataFrame], tuple[int, float]],
        *,
        step_size: Optional[int] = None,
        confidence_threshold: float = 0.6,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0002,
        max_context: Optional[int] = None,
        pred_len: Optional[int] = None,
    ) -> BacktestResult:
        """Execute a walk-forward backtest.

        Parameters
        ----------
        df:
            OHLCV DataFrame with columns ``open``, ``high``, ``low``,
            ``close`` (and optionally ``volume``).  A ``DatetimeIndex`` is
            expected but not required.
        predict_fn:
            **Signature**: ``predict_fn(hist_df: pd.DataFrame) -> (signal: int, confidence: float)``.

            * ``signal``: ``1`` = go long, ``-1`` = go short, ``0`` = stay flat.
            * ``confidence``: value in ``[0, 1]`` used against
              ``confidence_threshold``.  Trades below threshold are rejected.
        step_size:
            Number of bars to advance the window on each iteration.
            Defaults to ``pred_len``.
        confidence_threshold:
            Minimum ``confidence`` from ``predict_fn`` to accept a signal.
        initial_capital:
            Starting portfolio value in rupees.
        commission_pct:
            Round-trip commission as a fraction of trade value.
        max_context:
            Number of historical bars passed to ``predict_fn`` each window.
            Default: ``config.max_context``.
        pred_len:
            Number of forward bars each prediction covers (trades are
            held until the end of this window).  Default: ``config.pred_len``.
        """
        max_context = max_context or self.config.max_context
        pred_len = pred_len or self.config.pred_len
        step_size = step_size or pred_len

        # --- validate inputs ------------------------------------------------
        required_cols = {"open", "high", "low", "close"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        min_rows = max_context + pred_len
        if len(df) < min_rows:
            raise ValueError(
                f"DataFrame has {len(df)} rows, need at least {min_rows} "
                f"(max_context={max_context} + pred_len={pred_len})"
            )

        # --- state ----------------------------------------------------------
        capital = float(initial_capital)
        position: int = 0  # -1 short, 0 flat, 1 long
        open_trade: Optional[dict] = None
        trades: list[BacktestTrade] = []
        equity_records: list[dict] = []
        total_windows = (len(df) - max_context - pred_len) // step_size + 1

        logger.info(
            "Backtest started: %d windows, capital=%.2f, step=%d, ctx=%d, pred=%d",
            total_windows, initial_capital, step_size, max_context, pred_len,
        )

        # --- walk-forward loop ----------------------------------------------
        for window_i in range(total_windows):
            start = window_i * step_size
            mid = start + max_context
            end = mid + pred_len
            if end > len(df):
                break

            hist = df.iloc[start:mid]
            actual = df.iloc[mid:end]
            current_price = float(hist["close"].iloc[-1])
            entry_time = hist.index[-1]

            # 1. Get signal
            try:
                signal, confidence = predict_fn(hist)
            except Exception:
                logger.exception("predict_fn failed at window %d", window_i)
                signal, confidence = 0, 0.0

            # 2. Close any existing position when signal changes or drops out
            if open_trade is not None:
                close_position = False
                exit_reason = ""

                if position == 1 and signal != 1:
                    close_position = True
                    exit_reason = "signal_exit"
                elif position == -1 and signal != -1:
                    close_position = True
                    exit_reason = "signal_exit"
                elif confidence < confidence_threshold:
                    close_position = True
                    exit_reason = "below_confidence_threshold"

                if close_position:
                    pnl, pnl_pct = self._close_trade(
                        open_trade, current_price, direction_first=False,
                    )
                    trades.append(
                        BacktestTrade(
                            entry_time=open_trade["time"],
                            exit_time=entry_time,
                            direction=open_trade["direction"],
                            entry_price=open_trade["price"],
                            exit_price=current_price,
                            shares=open_trade["shares"],
                            pnl=pnl,
                            pnl_pct=pnl_pct,
                            confidence=open_trade["confidence"],
                            exit_reason=exit_reason,
                        )
                    )
                    capital += pnl
                    capital -= open_trade["cost"]
                    open_trade = None
                    position = 0

            # 3. Enter new position if signal is strong enough
            if abs(signal) > 0 and confidence >= confidence_threshold and open_trade is None:
                direction = "LONG" if signal == 1 else "SHORT"
                # Use 95 % of available capital per trade
                committed = capital * 0.95
                shares = committed / current_price
                cost = shares * current_price * commission_pct
                # We subtract commission from capital *now*; it's a sunk cost
                # regardless of trade outcome.
                capital -= cost

                open_trade = {
                    "time": entry_time,
                    "price": current_price,
                    "shares": shares,
                    "direction": direction,
                    "confidence": confidence,
                    "cost": cost,
                    "capital_at_entry": capital,
                }
                position = signal

            # 4. Record portfolio value (liquid + unrealised)
            unrealised = 0.0
            if open_trade is not None:
                if position == 1:
                    unrealised = (current_price - open_trade["price"]) * open_trade["shares"]
                else:
                    unrealised = (open_trade["price"] - current_price) * open_trade["shares"]

            equity_records.append({"time": entry_time, "equity": capital + unrealised})

        # --- close any position still open at end of data -------------------
        if open_trade is not None:
            last_bar = df.iloc[-1]
            final_price = float(last_bar["close"])
            pnl, pnl_pct = self._close_trade(
                open_trade, final_price, direction_first=False,
            )
            trades.append(
                BacktestTrade(
                    entry_time=open_trade["time"],
                    exit_time=last_bar.name,
                    direction=open_trade["direction"],
                    entry_price=open_trade["price"],
                    exit_price=final_price,
                    shares=open_trade["shares"],
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    confidence=open_trade["confidence"],
                    exit_reason="end_of_data",
                )
            )
            capital += pnl
            capital -= open_trade["cost"]
            open_trade = None
            position = 0
            equity_records.append({"time": last_bar.name, "equity": capital})

        # --- build equity curve ---------------------------------------------
        if equity_records:
            eq_df = pd.DataFrame(equity_records).set_index("time")
            equity_series = eq_df["equity"].sort_index()
        else:
            equity_series = pd.Series([initial_capital], index=[df.index[0]])

        logger.info(
            "Backtest complete: %d trades, final equity=%.2f",
            len(trades), equity_series.iloc[-1] if len(equity_series) else capital,
        )

        # --- compute & return results ---------------------------------------
        calc = BacktestMetricsCalculator(
            equity_curve=equity_series,
            trades=trades,
            initial_capital=initial_capital,
        )
        result = calc.compute_all()
        result.trades = trades
        result.equity_curve = equity_series
        result.config_snapshot = {
            "confidence_threshold": confidence_threshold,
            "commission_pct": commission_pct,
            "initial_capital": initial_capital,
            "step_size": step_size,
            "max_context": max_context,
            "pred_len": pred_len,
        }
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _close_trade(
        trade: dict,
        exit_price: float,
        *,
        direction_first: bool = False,  # unused, kept for signature compat
    ) -> tuple[float, float]:
        """Return (pnl, pnl_pct) for closing *trade* at *exit_price*."""
        direction = trade["direction"]
        price = trade["price"]
        shares = trade["shares"]

        if direction == "LONG":
            pnl = (exit_price - price) * shares
        else:
            pnl = (price - exit_price) * shares

        pnl_pct = ((exit_price - price) / price) * 100
        if direction == "SHORT":
            pnl_pct = -pnl_pct

        return pnl, pnl_pct
