"""Shared helpers for the kronos module."""

from __future__ import annotations

import logging
import sys
from typing import Any

import numpy as np
import pandas as pd
from typing import cast

# Re-exported type alias for readability
KronosSignal = int  # 1 = BUY, -1 = SELL, 0 = HOLD

# Module-level logger so every submodule can ``from kronos.utils.helpers import logger``
logger = logging.getLogger("kronos")


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``kronos`` namespace.

    Usage
    -----
        logger = get_logger(__name__)
    """
    return logger.getChild(name)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure the ``kronos`` logger with a simple console handler.

    Call once at application start.  If the logger already has handlers
    attached (e.g. because OpenAlgo's own ``setup_logging`` ran first)
    this is a no-op.
    """
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)


# ── OHLCV ↔ Kronos helpers ──────────────────────────────────────────


def ohlcv_to_kronos_df(
    records: list[dict[str, Any]],
    *,
    timestamp_col: str = "timestamp",
    open_col: str = "open",
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    volume_col: str = "volume",
    amount_col: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Convert a list of OHLCV records (as returned by the OpenAlgo
    History API) into a DataFrame + timestamp Series suitable for
    ``KronosPredictor.predict()``.

    Parameters
    ----------
    records : list[dict[str, Any]]
        Raw API response ``data`` list.
    timestamp_col : str
        Name of the timestamp field in each record.
    open_col, high_col, low_col, close_col, volume_col : str
        OHLCV field names.
    amount_col : str | None
        Optional amount (quote volume) field.

    Returns
    -------
    df : pd.DataFrame
        Columns ``[open, high, low, close]`` plus ``volume`` and,
        if available, ``amount``.  Index is a ``pd.DatetimeIndex``.
    timestamps : pd.Series
        Parsed timestamps aligned to ``df``.
    """
    df = pd.DataFrame(records)

    # Parse timestamps (Unix seconds → datetime)
    timestamps = pd.to_datetime(df[timestamp_col], unit="s")
    df.set_index(timestamps, inplace=True)

    # Keep only the columns Kronos needs
    keep = [open_col, high_col, low_col, close_col, volume_col]
    if amount_col and amount_col in df.columns:
        keep.append(amount_col)

    df = df[keep].astype(np.float64)
    df.columns = ["open", "high", "low", "close", "volume"] + (
        ["amount"] if amount_col and amount_col in records[0] else []
    )

    return cast(pd.DataFrame, df), cast(pd.Series, timestamps)


def ohlcv_df_to_kronos_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert a standard OHLCV DataFrame (as returned by
    :class:`OpenAlgoDataProvider`) to Kronos model input format.

    The provider returns columns ``['open', 'high', 'low', 'close',
    'volume']`` with a ``DatetimeIndex``.  This function ensures the
    data is cast to ``float64`` — exactly what the Kronos model expects.

    Parameters
    ----------
    df:
        OHLCV DataFrame (at minimum columns ``open``, ``high``,
        ``low``, ``close``; ``volume`` is optional).

    Returns
    -------
    pd.DataFrame
        Same columns, all ``float64``.
    """
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame missing required columns: {missing}")

    result = df[required].copy()
    if "volume" in df.columns:
        result["volume"] = df["volume"].astype(np.float64)

    return cast(pd.DataFrame, result.astype(np.float64))


def kronos_prediction_to_signal(
    prediction: np.ndarray,
    *,
    lookahead: int = 0,
    threshold: float = 0.0,
) -> tuple[KronosSignal, float]:
    """Convert a Kronos multi-sample price prediction into a directional
    signal.

    The function takes the median of all sampled predictions at index
    ``lookahead`` and compares it to the last known price.

    Parameters
    ----------
    prediction : np.ndarray
        Shape ``(sample_count, pred_len, 4)`` — raw output of
        ``KronosPredictor.predict()`` (channels: open, high, low, close).
    lookahead : int
        Which future step to evaluate (0-based).  Default ``0`` (first
        predicted candle).
    threshold : float
        Minimum absolute return (fraction) to produce a non-neutral
        signal.  Default ``0.0`` — every non-zero move triggers a signal.

    Returns
    -------
    signal : int
        ``1`` (BUY), ``-1`` (SELL), or ``0`` (HOLD / no conviction).
    confidence : float
        Fraction of samples agreeing on direction (``[0, 1]``).
    """
    # Median predicted close across samples at the target step
    pred_closes = prediction[:, lookahead, 3]  # index 3 = close
    median_pred = float(np.median(pred_closes))
    # The "last known" close is the close of the final input candle.
    # We do *not* have access to that here, so the caller must pass a
    # pre-computed ``last_close`` or we use a simple proxy: the mean of
    # all samples' open at step 0 (this is approximate).
    last_close = float(np.mean(prediction[:, 0, 0]))  # open of step 0 ≈ last close

    returns = (median_pred - last_close) / last_close

    if abs(returns) <= threshold:
        return 0, 0.0

    # Fraction of samples that move in the majority direction
    direction = np.sign(median_pred - last_close)
    agreement = float(np.mean(np.sign(pred_closes - last_close) == direction))

    return (1 if returns > 0 else -1), agreement
