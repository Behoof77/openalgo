"""Data adapter transforming raw OHLCV data into Kronos-ready format.

Sits between data providers (OpenAlgo, yfinance, CSV) and the
:class:`~kronos.client.KronosClient`.  Handles validation, NaN/inf
cleaning, context-length enforcement, and frequency resolution so
strategies don't have to.

Usage
-----
    provider = OpenAlgoDataProvider(config)
    adapter = KronosDataAdapter(max_context=config.max_context)

    raw_df = provider.fetch_history("NIFTY", "NSE", "3m", start, end)
    clean_df, freq = adapter.prepare(raw_df, interval="3m")

    pred = kronos_client.predict(clean_df, freq=freq, ...)
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd

from kronos.utils.helpers import logger as _kronos_logger

logger = _kronos_logger.getChild("adapter")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")
"""Minimum columns the Kronos model needs."""

OPTIONAL_COLUMNS: tuple[str, ...] = ("volume",)
"""Columns that are useful but not mandatory."""

# OpenAlgo interval string → pandas frequency alias
# See https://pandas.pydata.org/docs/user_guide/timeseries.html#dateoffset-objects
INTERVAL_TO_FREQ: dict[str, str] = {
    "1s": "1s",
    "5s": "5s",
    "10s": "10s",
    "15s": "15s",
    "30s": "30s",
    "1m": "1min",
    "2m": "2min",
    "3m": "3min",
    "5m": "5min",
    "10m": "10min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "3h": "3h",
    "4h": "4h",
    "D": "D",
    "W": "W",
    "M": "ME",
    "Q": "QE",
    "Y": "YE",
}
"""Maps OpenAlgo/DataProvider interval labels to pandas frequency strings."""

# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class KronosDataError(ValueError):
    """Raised when the input data fails validation."""


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class KronosDataAdapter:
    """Transforms raw OHLCV data into a clean DataFrame ready for the
    Kronos inference server.

    Parameters
    ----------
    max_context:
        Maximum number of rows the Kronos model can ingest.  Data is
        truncated from the **left** (oldest rows dropped) to at most this
        many rows.  Must be >= 64.
    strict:
        When ``True`` (default), raise :class:`KronosDataError` on missing
        columns or empty data.  When ``False``, log a warning and return
        an empty DataFrame.
    """

    def __init__(self, max_context: int = 512, strict: bool = True) -> None:
        if max_context < 64:
            raise ValueError(f"max_context must be >= 64, got {max_context}")
        self.max_context = max_context
        self.strict = strict

    # -- public entry point -------------------------------------------------

    def prepare(
        self,
        df: pd.DataFrame,
        interval: str | None = None,
    ) -> tuple[pd.DataFrame, str]:
        """Validate, clean, and truncate *df* for Kronos inference.

        Parameters
        ----------
        df:
            Raw OHLCV DataFrame from any source.  Must contain columns
            ``open``, ``high``, ``low``, ``close`` (lower-case).  May
            optionally include ``volume``.  Should have a ``DatetimeIndex``
            or a ``timestamp`` column.

        interval:
            Candle interval (e.g. ``"3m"``, ``"1h"``, ``"D"``).  When
            provided, the pandas frequency string is derived from this.
            Pass ``None`` if the caller will supply *freq* directly to
            :meth:`KronosClient.predict`.

        Returns
        -------
        clean_df:
            DataFrame with ``max_context`` rows (or fewer if the source
            has fewer rows).  All numeric columns are ``float64``.
            Guaranteed free of ``NaN`` and ``inf`` values.
        freq:
            Pandas frequency string inferred from *interval*, or ``""``
            if un-available.

        Raises
        ------
        KronosDataError
            On validation failure when ``strict=True``.
        """
        logger.debug("Preparing DataFrame: shape=%s, interval=%s", df.shape, interval)

        # 1. Validate
        self.validate(df)

        # Early exit on missing required columns (non-strict)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            logger.warning("Cannot prepare data: missing %s. Returning empty.", missing)
            return pd.DataFrame(), ""

        # 2. Cast all known columns to float64
        df = self._cast_float64(df)

        # 3. Clean NaN / inf
        df = self.clean(df)

        # 4. Truncate to max_context
        df = self.truncate(df)

        # 5. Resolve frequency
        freq = self._resolve_freq(df, interval)

        logger.debug(
            "Prepare done: shape=%s, freq=%s, NaN=%s",
            df.shape, freq, df.isna().any().any(),
        )
        return df, freq

    # -- validation ---------------------------------------------------------

    def validate(self, df: pd.DataFrame) -> None:
        """Check that *df* has the columns Kronos requires.

        Raises :class:`KronosDataError` if validation fails and
        ``self.strict`` is ``True``; otherwise logs a warning.
        """
        if df is None or df.empty:
            msg = "DataFrame is None or empty"
            if self.strict:
                raise KronosDataError(msg)
            logger.warning(msg)
            return

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            msg = f"Missing required columns: {missing}. Got: {list(df.columns)}"
            if self.strict:
                raise KronosDataError(msg)
            logger.warning(msg)

        # Warn about unexpected types
        for col in REQUIRED_COLUMNS:
            if col in df.columns and not np.issubdtype(df[col].dtype, np.number):
                logger.warning(
                    "Column '%s' has non-numeric dtype %s — will coerce",
                    col, df[col].dtype,
                )

    # -- cleaning -----------------------------------------------------------

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove rows with NaN or infinite values in required columns.

        Operates on a copy — the original DataFrame is not mutated.
        """
        result = df.copy()

        # Drop rows where any required column is NaN
        before = len(result)
        result = result.dropna(subset=list(REQUIRED_COLUMNS))
        dropped_nan = before - len(result)
        if dropped_nan:
            logger.warning("Dropped %d rows with NaN values", dropped_nan)

        # Replace inf with NaN, then drop
        for col in REQUIRED_COLUMNS:
            if col in result.columns:
                mask = np.isinf(result[col].values)
                n_inf = int(mask.sum())
                if n_inf:
                    logger.warning("Found %d inf values in '%s' — dropping rows", n_inf, col)
                    result.loc[mask, col] = np.nan

        result = result.dropna(subset=list(REQUIRED_COLUMNS))

        return result

    # -- truncation ---------------------------------------------------------

    def truncate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep only the last ``max_context`` rows (drop oldest).

        If the DataFrame already fits within ``max_context`` rows it is
        returned unchanged.
        """
        if len(df) > self.max_context:
            logger.info(
                "Truncating %d → %d rows (dropping %d oldest)",
                len(df), self.max_context, len(df) - self.max_context,
            )
            return df.iloc[-self.max_context :].copy()
        return df

    # -- frequency resolution -----------------------------------------------

    @staticmethod
    def interval_to_freq(interval: str) -> str:
        """Convert an OpenAlgo interval string to a pandas frequency.

        Examples
        --------
        >>> KronosDataAdapter.interval_to_freq("3m")
        '3min'
        >>> KronosDataAdapter.interval_to_freq("D")
        'D'
        >>> KronosDataAdapter.interval_to_freq("1h")
        '1h'

        Returns ``""`` if the interval is not recognised.
        """
        return INTERVAL_TO_FREQ.get(interval, "")

    @staticmethod
    def infer_freq_from_index(df: pd.DataFrame) -> str:
        """Infer the pandas frequency string from a DataFrame's DatetimeIndex.

        Returns ``""`` if the index is not a DatetimeIndex or inference
        fails.
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return ""
        try:
            freq = pd.infer_freq(df.index)
            return freq if freq else ""
        except Exception:
            return ""

    # -- helpers ------------------------------------------------------------

    @staticmethod
    def _cast_float64(df: pd.DataFrame) -> pd.DataFrame:
        """Cast known OHLCV columns to ``float64``."""
        result = df.copy()
        all_known = list(REQUIRED_COLUMNS) + list(OPTIONAL_COLUMNS)
        for col in all_known:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce")
                result[col] = result[col].astype(np.float64)
        return result

    def _resolve_freq(self, df: pd.DataFrame, interval: str | None) -> str:
        """Best-effort frequency resolution."""
        if interval:
            freq = self.interval_to_freq(interval)
            if freq:
                return freq
        return self.infer_freq_from_index(df)
