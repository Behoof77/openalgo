# kronos - AI-Powered Market Prediction for OpenAlgo
#
# Kronos integrates the Kronos time-series foundation model into OpenAlgo
# for OHLCV-based price prediction, backtesting, and live trading.
#
# Quick start:
#   from kronos.utils.config import KronosConfig
#   from kronos.data.openalgo_provider import OpenAlgoDataProvider
#
#   config = KronosConfig()
#   provider = OpenAlgoDataProvider(config)
#   df = provider.fetch_history("NIFTY", "NSE", "5m", ...)

VERSION = "0.1.0"

from kronos.utils.config import KronosConfig
from kronos.data.openalgo_provider import OpenAlgoDataProvider
from kronos.data.option_provider import OptionDataProvider, OptionSentiment
from kronos.data.ws_client import KronosWebSocketClient
from kronos.broker.bridge import OpenAlgoBrokerBridge, PlaceOrderRequest
from kronos.backtest.engine import KronosBacktestEngine, BacktestTrade
from kronos.backtest.metrics import BacktestResult, BacktestMetricsCalculator
from kronos.live.runner import KronosLiveRunner, LiveTradeRecord, LiveRunnerState

__all__ = [
    "KronosConfig",
    "OpenAlgoDataProvider",
    "OptionDataProvider",
    "OptionSentiment",
    "KronosWebSocketClient",
    "OpenAlgoBrokerBridge",
    "PlaceOrderRequest",
    "KronosBacktestEngine",
    "BacktestTrade",
    "BacktestResult",
    "BacktestMetricsCalculator",
    "KronosLiveRunner",
    "LiveTradeRecord",
    "LiveRunnerState",
    "VERSION",
]
