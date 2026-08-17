"""
strategies.shared — Shared infrastructure for self-hosted strategies.

Provides:
    execution_modes  - INTRADAY / SWING mode constants and per-strategy lookup
    stop_modes       - STOP_AND_CLOSE / STOP_TRADING_ONLY constants
    lot_size         - Dynamic lot-size fetch from broker instrument master
    config_loader    - YAML strategy config loader with validation
    state_persister  - SQLite-backed ownership, recovery, stop-state persistence
    position_owner   - Position ownership tracking (assign, query, release)
    reconciler       - Broker-to-strategy reconciliation on startup
    safety_rules     - 7 hard safety rules enforcement
    recovery         - Crash/restart recovery state machine
"""

from strategies.shared.execution_modes import (
    ExecutionMode,
    get_strategy_mode,
    validate_execution_mode,
)
from strategies.shared.stop_modes import StopMode
from strategies.shared.lot_size import get_lot_size, validate_lot_size
from strategies.shared.config_loader import load_strategy_config
from strategies.shared.state_persister import StatePersister
from strategies.shared.position_owner import PositionOwner
from strategies.shared.reconciler import Reconciler
from strategies.shared.safety_rules import SafetyRules
from strategies.shared.recovery import RecoveryManager

__all__ = [
    "ExecutionMode",
    "get_strategy_mode",
    "validate_execution_mode",
    "StopMode",
    "get_lot_size",
    "validate_lot_size",
    "load_strategy_config",
    "StatePersister",
    "PositionOwner",
    "Reconciler",
    "SafetyRules",
    "RecoveryManager",
]
