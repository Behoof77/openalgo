"""
Backtest performance metrics.

Computes all standard financial statistics from an equity curve and
a list of completed trades.  Uses pandas vectorised operations where
possible to avoid slow Python loops.
"""

from __future__ import annotations

import math
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from kronos.utils.helpers import get_logger

logger = get_logger(__name__)


@dataclass
class BacktestResult:
    """Aggregated backtest results.

    Returned by :meth:`BacktestMetricsCalculator.compute_all`.
    """

    total_return_pct: float = 0.0
    cagr_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    avg_trade_pnl_pct: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    avg_holding_bars: float = 0.0
    trades: list = field(default_factory=list)
    equity_curve: pd.Series | None = None
    config_snapshot: dict = field(default_factory=dict)

    def summary(self) -> dict:
        """Return a flat dictionary suitable for printing or logging."""
        return {
            "Total Return %": f"{self.total_return_pct:.2f}",
            "CAGR %": f"{self.cagr_pct:.2f}",
            "Sharpe Ratio": f"{self.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.sortino_ratio:.2f}",
            "Calmar Ratio": f"{self.calmar_ratio:.2f}",
            "Max Drawdown %": f"{self.max_drawdown_pct:.2f}",
            "Win Rate %": f"{self.win_rate:.1f}",
            "Profit Factor": f"{self.profit_factor:.2f}",
            "Total Trades": self.total_trades,
            "Winning Trades": self.winning_trades,
            "Losing Trades": self.losing_trades,
            "Avg Trade %": f"{self.avg_trade_pnl_pct:.2f}",
            "Avg Win %": f"{self.avg_win_pct:.2f}",
            "Avg Loss %": f"{self.avg_loss_pct:.2f}",
            "Avg Holding (bars)": f"{self.avg_holding_bars:.1f}",
        }


class BacktestMetricsCalculator:
    """Compute financial metrics from an equity curve and trade list.

    Parameters
    ----------
    equity_curve:
        Time series of portfolio value (index = datetime, values = equity).
        Must be daily or higher frequency; annualisation assumes **252
        trading days per year**.
    trades:
        Iterable of trade records with ``pnl_pct`` and ``pnl`` attributes.
    initial_capital:
        Starting capital (used for return calculations).
    risk_free_rate:
        Annual risk-free rate for Sharpe / Sortino (default 0.05 = 5 %).
    """

    TRADING_DAYS = 252  # For daily data annualisation

    def __init__(
        self,
        equity_curve: pd.Series,
        trades: list,
        initial_capital: float,
        risk_free_rate: float = 0.05,
    ):
        self.equity = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        self.rf = risk_free_rate

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def compute_all(self) -> BacktestResult:
        """Compute every metric and return a populated :class:`BacktestResult`."""
        r = BacktestResult()
        if len(self.equity) < 2 and len(self.trades) == 0:
            logger.warning("No data to compute metrics")
            return r

        r.equity_curve = self.equity

        # --- returns ---------------------------------------------------
        final_equity = float(self.equity.iloc[-1])
        r.total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # CAGR
        years = self._years_elapsed(self.equity.index)
        if years > 0 and r.total_return_pct > -100:
            r.cagr_pct = ((final_equity / self.initial_capital) ** (1.0 / years) - 1) * 100

        # --- drawdown --------------------------------------------------
        dd, r.max_drawdown_pct = self._max_drawdown(self.equity)

        # --- risk-adjusted ratios --------------------------------------
        daily_returns = self.equity.pct_change().dropna()
        if len(daily_returns) > 1:
            r.sharpe_ratio = self._sharpe(daily_returns)
            r.sortino_ratio = self._sortino(daily_returns)
            if r.max_drawdown_pct > 0:
                r.calmar_ratio = r.cagr_pct / r.max_drawdown_pct

        # --- trade stats -----------------------------------------------
        r.total_trades = len(self.trades)
        if r.total_trades > 0:
            pnl_pcts = np.array([t.pnl_pct for t in self.trades], dtype=float)
            pnls = np.array([t.pnl for t in self.trades], dtype=float)

            winning = pnl_pcts > 0
            r.winning_trades = int(winning.sum())
            r.losing_trades = int((pnl_pcts <= 0).sum())
            r.win_rate = (r.winning_trades / r.total_trades) * 100 if r.total_trades else 0.0

            positive_pnl = pnl_pcts[pnl_pcts > 0]
            negative_pnl = pnl_pcts[pnl_pcts <= 0]

            r.avg_trade_pnl_pct = float(np.mean(pnl_pcts))
            r.avg_win_pct = float(np.mean(positive_pnl)) if len(positive_pnl) > 0 else 0.0
            r.avg_loss_pct = float(np.mean(negative_pnl)) if len(negative_pnl) > 0 else 0.0

            # Profit factor (ratio of gross wins to gross losses)
            gross_win = float(pnls[winning].sum())
            gross_loss = abs(float(pnls[~winning].sum()))
            r.profit_factor = gross_win / gross_loss if gross_loss > 1e-8 else float("inf")

            # Average holding period
            holding_bars = []
            for t in self.trades:
                if hasattr(t, "entry_time") and hasattr(t, "exit_time"):
                    # Estimate bars by time difference (rough)
                    delta = pd.Timestamp(t.exit_time) - pd.Timestamp(t.entry_time)
                    if hasattr(self.equity.index, "freq") and self.equity.index.freq is not None:
                        bars = delta.total_seconds() / self.equity.index.freq.n / (
                            self.equity.index.freq.seconds or 86400
                        )
                    else:
                        bars = delta.days  # fallback
                    holding_bars.append(bars)
            if holding_bars:
                r.avg_holding_bars = float(np.mean(holding_bars))

        return r

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _years_elapsed(self, index: pd.Index) -> float:
        """Return fractional years between first and last index entry."""
        if len(index) < 2:
            return 0.0
        delta = index[-1] - index[0]
        return delta.total_seconds() / (365.25 * 86400)

    def _max_drawdown(self, equity: pd.Series) -> tuple[pd.Series, float]:
        """Compute drawdown series and its maximum percentage."""
        rolling_max = equity.expanding().max()
        dd = (equity - rolling_max) / rolling_max * 100
        return dd, float(dd.min())

    def _sharpe(self, daily_returns: pd.Series) -> float:
        """Annualised Sharpe ratio."""
        excess = daily_returns - self.rf / self.TRADING_DAYS
        if excess.std() < 1e-10:
            return 0.0
        return float(np.sqrt(self.TRADING_DAYS) * excess.mean() / excess.std())

    def _sortino(self, daily_returns: pd.Series) -> float:
        """Annualised Sortino ratio (downside deviation only)."""
        excess = daily_returns - self.rf / self.TRADING_DAYS
        downside = excess[excess < 0]
        if len(downside) < 1 or downside.std() < 1e-10:
            return 0.0
        return float(np.sqrt(self.TRADING_DAYS) * excess.mean() / downside.std())
