"""
execution_modes.py — Strategy execution mode constants and lookup.

Every strategy is either INTRADAY (must close before session end) or SWING
(can hold overnight). Mode is fixed per strategy and read from YAML config.
"""

from enum import Enum


class ExecutionMode(str, Enum):
    """Strategy execution mode."""

    INTRADAY = "INTRADAY"
    SWING = "SWING"


# Default mode map — overridden by YAML config
_DEFAULT_MODES: dict[str, ExecutionMode] = {
    "kronos_sizing": ExecutionMode.INTRADAY,
    "kronos_vote": ExecutionMode.INTRADAY,
    "atm_premium_ml": ExecutionMode.SWING,
}


def get_strategy_mode(strategy_id: str) -> ExecutionMode:
    """Return the execution mode for a strategy_id.

    Looks up the strategy in the registered mode map.
    Falls back to INTRADAY if unknown (safe default — forces close at session end).

    Args:
        strategy_id: Unique strategy identifier (e.g. "kronos_sizing").

    Returns:
        ExecutionMode enum value.
    """
    return _DEFAULT_MODES.get(strategy_id, ExecutionMode.INTRADAY)


def validate_execution_mode(mode: str) -> ExecutionMode:
    """Validate and convert a string to ExecutionMode.

    Args:
        mode: String like "INTRADAY" or "SWING".

    Returns:
        ExecutionMode enum value.

    Raises:
        ValueError: If mode is not a valid ExecutionMode.
    """
    try:
        return ExecutionMode(mode.upper())
    except ValueError:
        valid = ", ".join(m.value for m in ExecutionMode)
        raise ValueError(f"Invalid execution mode '{mode}'. Must be one of: {valid}")


def register_strategy_mode(strategy_id: str, mode: ExecutionMode) -> None:
    """Register a strategy's execution mode at runtime.

    Called by config_loader when loading YAML configs.

    Args:
        strategy_id: Unique strategy identifier.
        mode: ExecutionMode enum value.
    """
    _DEFAULT_MODES[strategy_id] = mode
