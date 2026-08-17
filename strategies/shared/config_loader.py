"""
config_loader.py — YAML strategy configuration loader.

Loads strategy configs from a YAML file, validates required fields,
and registers execution modes. Adding a new strategy only requires
a YAML entry — no core engine changes.
"""

import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

from strategies.shared.execution_modes import (
    ExecutionMode,
    register_strategy_mode,
    validate_execution_mode,
)
from strategies.shared.stop_modes import StopMode, validate_stop_mode

# Default config path (relative to strategies/)
_DEFAULT_CONFIG = Path(__file__).parent.parent / "strategy_configs.yaml"


class StrategyConfig:
    """Configuration for a single strategy."""

    def __init__(self, strategy_id: str, data: dict) -> None:
        self.strategy_id = strategy_id
        self.name: str = data.get("name", strategy_id)
        self.execution_mode: ExecutionMode = validate_execution_mode(
            data.get("execution_mode", "INTRADAY")
        )
        self.stop_mode: StopMode = validate_stop_mode(
            data.get("stop_mode", "STOP_AND_CLOSE")
        )
        self.underlying: str = data.get("underlying", "NIFTY")
        self.exchange: str = data.get("exchange", "NFO")
        self.product: str = data.get("product", "MIS")
        self.quantity: int = int(data.get("quantity", 75))
        self.symbols: list[dict] = data.get("symbols", [])
        self.risk: dict = data.get("risk", {})
        self.params: dict = data.get("params", {})

    def __repr__(self) -> str:
        return (
            f"StrategyConfig(id={self.strategy_id}, mode={self.execution_mode.value}, "
            f"underlying={self.underlying}, qty={self.quantity})"
        )


def load_strategy_config(config_path: str | Path | None = None) -> dict[str, StrategyConfig]:
    """Load and validate strategy configurations from YAML.

    Args:
        config_path: Path to YAML config file. Defaults to strategies/strategy_configs.yaml.

    Returns:
        Dict mapping strategy_id -> StrategyConfig.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If config file not found.
        ValueError: If config is invalid.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is required for strategy config loading. "
            "Install with: pip install pyyaml"
        )

    path = Path(config_path) if config_path else _DEFAULT_CONFIG
    if not path.exists():
        raise FileNotFoundError(f"Strategy config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not raw or "strategies" not in raw:
        raise ValueError(f"Config file {path} must have a top-level 'strategies' key")

    configs: dict[str, StrategyConfig] = {}
    for strategy_id, strategy_data in raw["strategies"].items():
        if not isinstance(strategy_data, dict):
            raise ValueError(f"Strategy '{strategy_id}' config must be a mapping")

        cfg = StrategyConfig(strategy_id, strategy_data)
        configs[strategy_id] = cfg

        # Register execution mode globally
        register_strategy_mode(strategy_id, cfg.execution_mode)

        print(f"[config] Loaded strategy: {cfg}")

    return configs


def get_config_for_env(configs: dict[str, StrategyConfig]) -> StrategyConfig | None:
    """Get the config for the current strategy based on STRATEGY_ID env var.

    Args:
        configs: Dict of loaded configs.

    Returns:
        StrategyConfig if found, None if STRATEGY_ID not set or not found.
    """
    strategy_id = os.getenv("STRATEGY_ID", "")
    if not strategy_id:
        return None
    return configs.get(strategy_id)
