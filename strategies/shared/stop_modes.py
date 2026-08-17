"""
stop_modes.py — Manual stop mode constants.

When a user manually stops a strategy, they choose between:
  - STOP_AND_CLOSE: close all positions AND stop trading (default)
  - STOP_TRADING_ONLY: stop placing new orders but keep existing positions open
"""

from enum import Enum


class StopMode(str, Enum):
    """How a strategy is stopped."""

    STOP_AND_CLOSE = "STOP_AND_CLOSE"
    STOP_TRADING_ONLY = "STOP_TRADING_ONLY"


def validate_stop_mode(mode: str) -> StopMode:
    """Validate and convert a string to StopMode.

    Args:
        mode: String like "STOP_AND_CLOSE" or "STOP_TRADING_ONLY".

    Returns:
        StopMode enum value.

    Raises:
        ValueError: If mode is not a valid StopMode.
    """
    try:
        return StopMode(mode.upper())
    except ValueError:
        valid = ", ".join(m.value for m in StopMode)
        raise ValueError(f"Invalid stop mode '{mode}'. Must be one of: {valid}")
