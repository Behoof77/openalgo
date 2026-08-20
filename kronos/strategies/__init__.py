"""Kronos trading strategies.

Each submodule is a self-contained strategy that can run in ``backtest``
or ``live`` mode.
"""

from kronos.strategies.nifty_kronos_strategy import NiftyKronosStrategy

__all__ = ["NiftyKronosStrategy"]
