"""Kronos configuration -- reads from .env at the project root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class KronosConfig:
    """Global configuration for the kronos module.

    Reads from environment variables / ``.env``.  All values have
    sensible defaults so the module is usable out of the box.
    """

    # ── Kronos inference server ──────────────────────────────────────
    kronos_server_host: str = field(
        default_factory=lambda: _env_str("KRONOS_SERVER_HOST", "http://127.0.0.1:8000")
    )
    """Base URL of the Kronos inference server (no trailing slash)."""

    # ── OpenAlgo connection ──────────────────────────────────────────
    openalgo_host: str = field(
        default_factory=lambda: _env_str("OPENALGO_HOST", "http://127.0.0.1:5000")
    )
    """Base URL of the OpenAlgo instance (no trailing slash)."""

    openalgo_api_key: str = field(
        default_factory=lambda: _env_str("OPENALGO_API_KEY", "")
    )
    """API key for OpenAlgo authentication."""

    # ── Kronos model ─────────────────────────────────────────────────
    model_name: str = field(
        default_factory=lambda: _env_str("KRONOS_MODEL_NAME", "shiyu-coder/Kronos")
    )
    """Hugging Face Hub model identifier."""

    max_context: int = field(default_factory=lambda: _env_int("KRONOS_MAX_CONTEXT", 512))
    """Maximum context length (tokens) the model can ingest."""

    pred_len: int = field(default_factory=lambda: _env_int("KRONOS_PRED_LEN", 24))
    """Default prediction horizon (number of future steps)."""

    temperature: float = field(
        default_factory=lambda: _env_float("KRONOS_TEMPERATURE", 1.0)
    )
    """Sampling temperature for generation."""

    top_p: float = field(default_factory=lambda: _env_float("KRONOS_TOP_P", 0.95))
    """Nucleus sampling threshold."""

    sample_count: int = field(
        default_factory=lambda: _env_int("KRONOS_SAMPLE_COUNT", 20)
    )
    """Number of prediction samples to draw."""

    # ── Device ───────────────────────────────────────────────────────
    device: str = field(default_factory=lambda: _env_str("KRONOS_DEVICE", "auto"))
    """One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``."""

    # ── Backtest defaults ────────────────────────────────────────────
    commission_pct: float = field(
        default_factory=lambda: _env_float("KRONOS_COMMISSION_PCT", 0.02)
    )
    """Brokerage per trade as a percentage of trade value."""

    initial_capital: float = field(
        default_factory=lambda: _env_float("KRONOS_INITIAL_CAPITAL", 100_000.0)
    )
    """Starting capital for backtests."""

    # ── Live trading ─────────────────────────────────────────────────
    poll_interval_seconds: int = field(
        default_factory=lambda: _env_int("KRONOS_POLL_INTERVAL", 60)
    )
    """Seconds between live prediction cycles."""

    # ── Paths ────────────────────────────────────────────────────────
    project_root: Path = field(
        default_factory=lambda: Path(
            _env_str("KRONOS_PROJECT_ROOT", str(Path.cwd()))
        )
    )
    """Root directory used for log / output file locations."""

    @classmethod
    def from_env(cls) -> KronosConfig:
        """Build a config from the current environment (the normal entry
        point)."""
        return cls()

    @property
    def resolved_device(self) -> str:
        """Return the resolved compute device string."""
        if self.device != "auto":
            return self.device
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
