"""HTTP client for the Kronos Inference Server.

The strategy uses this client to request predictions instead of loading the
Kronos model directly.  This keeps the strategy lightweight and the model
loaded persistently in the server process.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from kronos.utils.helpers import get_logger

logger = get_logger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class KronosPrediction:
    """Prediction result returned by the inference server."""

    signal: int = 0
    """1 (up), -1 (down), 0 (neutral/hold)."""

    confidence: float = 0.0
    """Confidence between 0 and 1."""

    raw_predictions: list[float] = field(default_factory=list)
    """Per-sample predictions from the server."""

    inference_ms: float = 0.0
    """Server-side inference latency."""

    rows_sent: int = 0
    """Number of OHLCV rows sent in the request."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class KronosClient:
    """HTTP client for the Kronos inference server.

    Usage
    -----
        client = KronosClient("http://127.0.0.1:8000")
        pred = client.predict(df)
        print(pred.signal, pred.confidence)
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 120.0):
        if httpx is None:
            raise RuntimeError("httpx is required — pip install httpx")
        self.base_url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(timeout))

    # -- health -------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        """Check server status.  Raises on unreachable server."""
        resp = self._http.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()

    # -- predict -------------------------------------------------------------

    def predict(
        self,
        df: pd.DataFrame,
        pred_len: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        sample_count: int | None = None,
    ) -> KronosPrediction:
        """Send OHLCV data to the Kronos server and return a prediction.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV data with columns ``open``, ``high``, ``low``, ``close``
            (and optionally ``volume``).  Should have at least 512 rows
            (``max_context``).
        pred_len : int, optional
            Forecast horizon override.
        temperature : float, optional
            Sampling temperature override.
        top_p : float, optional
            Nucleus sampling threshold override.
        sample_count : int, optional
            Number of samples override.

        Returns
        -------
        KronosPrediction
        """
        # Prepare data payload: transpose DataFrame to list of dicts
        # Drop index / timestamp — the server doesn't need it for inference
        cols = ["open", "high", "low", "close"]
        if "volume" in df.columns:
            cols.append("volume")

        records: list[dict[str, Any]] = df[cols].to_dict(orient="records")

        body: dict[str, Any] = {"data": records}
        if pred_len is not None:
            body["pred_len"] = pred_len
        if temperature is not None:
            body["temperature"] = temperature
        if top_p is not None:
            body["top_p"] = top_p
        if sample_count is not None:
            body["sample_count"] = sample_count

        logger.debug(
            "Predict request: %d rows, cols=%s",
            len(records), list(records[0].keys()) if records else [],
        )

        resp = self._http.post(f"{self.base_url}/predict", json=body)
        resp.raise_for_status()
        data = resp.json()

        if data.get("prediction") is None:
            raise RuntimeError(f"Server returned no prediction: {data}")

        return KronosPrediction(
            signal=data["prediction"],
            confidence=data.get("confidence", 0.0),
            raw_predictions=data.get("raw_predictions", []),
            inference_ms=data.get("inference_ms", 0.0),
            rows_sent=len(records),
        )
