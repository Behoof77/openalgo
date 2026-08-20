"""Data provider that fetches OHLCV history from the OpenAlgo History API."""

from __future__ import annotations

from typing import Any

import pandas as pd

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import logger

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


class OpenAlgoDataProvider:
    """Fetches historical OHLCV data from OpenAlgo's ``/api/v1/history/``
    endpoint and returns it as a ``pd.DataFrame`` ready for Kronos."""

    def __init__(self, config: KronosConfig | None = None) -> None:
        self.config = config or KronosConfig.from_env()
        self._client: httpx.Client | None = None

    # ── public helpers ───────────────────────────────────────────────

    VALID_INTERVALS: tuple[str, ...] = (
        "1s", "5s", "10s", "15s", "30s", "45s",
        "1m", "2m", "3m", "5m", "10m", "15m", "20m", "30m",
        "1h", "2h", "3h", "4h",
        "D", "W", "M", "Q", "Y",
    )

    @staticmethod
    def validate_interval(interval: str) -> None:
        """Raise ``ValueError`` if *interval* is not recognised."""
        if interval not in OpenAlgoDataProvider.VALID_INTERVALS:
            raise ValueError(
                f"Unknown interval {interval!r}.  Valid choices: "
                f"{', '.join(OpenAlgoDataProvider.VALID_INTERVALS)}"
            )

    # ── HTTP lifecycle ───────────────────────────────────────────────

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx is not installed — run `pip install httpx`")
            self._client = httpx.Client(timeout=httpx.Timeout(30.0))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAlgoDataProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── main fetch method ────────────────────────────────────────────

    def fetch_history(
        self,
        symbol: str,
        exchange: str,
        interval: str,
        start_date: str,
        end_date: str,
        *,
        source: str = "api",
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from the OpenAlgo History API.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g. ``"NIFTY"``, ``"SBIN"``).
        exchange : str
            Exchange (e.g. ``"NSE"``, ``"NFO"``).
        interval : str
            Candle interval — see ``VALID_INTERVALS``.
        start_date : str
            Start date in ``"YYYY-MM-DD"`` format.
        end_date : str
            End date in ``"YYYY-MM-DD"`` format.
        source : str
            ``"api"`` (broker API, default) or ``"db"`` (DuckDB/Historify).

        Returns
        -------
        pd.DataFrame
            Columns ``[open, high, low, close, volume]``, index is a
            ``DatetimeIndex`` (parsed from Unix-second timestamps).

        Raises
        ------
        RuntimeError
            If the API returns a non-success status.
        httpx.HTTPError
            On network / HTTP errors.
        """
        self.validate_interval(interval)

        if not self.config.openalgo_api_key:
            raise RuntimeError(
                "OPENALGO_API_KEY is not set.  Add it to your .env file."
            )

        url = f"{self.config.openalgo_host}/api/v1/history/"
        payload: dict[str, Any] = {
            "apikey": self.config.openalgo_api_key,
            "symbol": symbol,
            "exchange": exchange,
            "interval": interval,
            "start_date": start_date,
            "end_date": end_date,
            "source": source,
        }

        logger.info(
            "Fetching history: %s %s %s [%s … %s] (%s)",
            symbol, exchange, interval, start_date, end_date, source,
        )
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        body = response.json()

        if body.get("status") != "success":
            msg = body.get("message", "unknown error")
            raise RuntimeError(f"OpenAlgo history API error: {msg}")

        records: list[dict[str, Any]] = body.get("data", [])
        if not records:
            logger.warning("No data returned for %s %s %s", symbol, exchange, interval)
            return pd.DataFrame()

        df = pd.DataFrame(records)

        # Parse timestamps (Unix seconds → datetime)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df.set_index("timestamp", inplace=True)

        # Ensure required columns and cast to float64
        for col in ("open", "high", "low", "close", "volume"):
            if col not in df.columns:
                df[col] = 0.0
        df = df.astype({c: "float64" for c in ("open", "high", "low", "close", "volume")})

        logger.info("Fetched %d candles for %s", len(df), symbol)
        return df[["open", "high", "low", "close", "volume"]]
