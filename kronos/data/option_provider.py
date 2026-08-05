"""Option-market data provider.

Fetches the NIFTY option chain, Greeks, and expiry information from the
OpenAlgo REST API.  Used in live mode to compute sentiment factors
(PCR, IV skew, OI momentum) for the multi-factor signal fusion pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import get_logger

logger = get_logger(__name__)

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


@dataclass
class OptionSentiment:
    """Aggregated option-market sentiment snapshot."""

    pcr: float = 1.0
    iv_skew: float = 0.0
    oi_change_pct: float = 0.0
    atm_strike: float = 0.0
    atm_iv: float = 0.0
    near_expiry: str = ""


class OptionDataProvider:
    """Fetches option chain, expiry, and Greeks from OpenAlgo.

    Every call hits the REST API with no caching.
    Rate-limit callers to once per minute (option data changes slowly).
    """

    def __init__(self, config: KronosConfig | None = None) -> None:
        self.config = config or KronosConfig.from_env()
        self._client: httpx.Client | None = None

    # -- HTTP lifecycle -------------------------------------------------

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx required -- pip install httpx")
            self._client = httpx.Client(timeout=httpx.Timeout(20.0))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OptionDataProvider:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # -- internal -------------------------------------------------------

    @staticmethod
    def _normalize_expiry(expiry: str) -> str:
        """Normalize expiry date from API format to symbol format.

        The `/api/v1/expiry/` endpoint returns ``"10-JUL-25"``.
        For symbol construction we need ``"10JUL25"`` (no dashes).
        """
        return expiry.replace("-", "")

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.config.openalgo_host}/api/v1/{endpoint}/"
        payload.setdefault("apikey", self.config.openalgo_api_key)
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    # -- public API -----------------------------------------------------

    def get_expiry(self, symbol: str = "NIFTY", exchange: str = "NSE") -> list[str]:
        """Return available expiry date strings (chronological, raw format)."""
        body = self._post("expiry", {"symbol": symbol, "exchange": exchange})
        if body.get("status") != "success":
            logger.warning("Expiry API error: %s", body.get("message"))
            return []
        raw = body.get("data", [])
        return raw if isinstance(raw, list) else []

    def get_normalized_expiry(self, symbol: str = "NIFTY", exchange: str = "NSE") -> str:
        """Return the nearest expiry in symbol format (e.g. ``\"10JUL25\"``)."""
        exps = self.get_expiry(symbol, exchange)
        if not exps:
            return ""
        return self._normalize_expiry(exps[0])

    def resolve_option_symbol(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
        expiry_date: str = "",
        offset: str = "ATM",
        option_type: str = "PE",
    ) -> dict[str, Any]:
        """Resolve an option symbol via the ``optionsymbol`` API endpoint.

        Returns the full API response dict with keys ``status``, ``symbol``,
        ``exchange``, ``lotsize``, ``tick_size``, ``underlying_ltp``.

        Parameters
        ----------
        expiry_date : str
            Symbol-format expiry (e.g. ``\"10JUL25\"``).  If empty the
            nearest expiry is fetched automatically.
        offset : str
            ``\"ATM\"``, ``\"ITM1\"`` … ``\"ITM20\"``, ``\"OTM1\"`` … ``\"OTM20\"``.
        option_type : str
            ``\"CE\"`` or ``\"PE\"``.
        """
        if not expiry_date:
            exps = self.get_expiry(underlying, exchange)
            if not exps:
                return {"status": "error", "message": "no expiry available"}
            expiry_date = self._normalize_expiry(exps[0])

        payload = {
            "underlying": underlying,
            "exchange": exchange,
            "expiry_date": expiry_date,
            "offset": offset,
            "option_type": option_type,
        }
        return self._post("optionsymbol", payload)

    def resolve_futures_symbol(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
    ) -> dict[str, Any]:
        """Resolve the current futures symbol via the ``symbol`` API endpoint.

        Builds a futures symbol in the format ``NIFTY10JUL25FUT`` and
        calls the symbol API to validate it.

        Returns the API response with ``status``, ``symbol``, ``lotsize``, etc.
        """
        expiry_date = self.get_normalized_expiry(underlying, exchange)
        if not expiry_date:
            return {"status": "error", "message": "no expiry available"}
        fut_symbol = f"{underlying}{expiry_date}FUT"
        return self._post("symbol", {"symbol": fut_symbol, "exchange": exchange})

    def get_option_chain(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
        expiry_date: str = "",
        strike_count: int = 10,
    ) -> list[dict[str, Any]]:
        """Return the option chain for *underlying* at *expiry_date*.

        Parameters
        ----------
        underlying : str
            Underlying index or symbol (e.g. ``\"NIFTY\"``).
        expiry_date : str
            Symbol-format expiry (e.g. ``\"10JUL25\"``).  Empty = nearest.
        strike_count : int
            Number of strikes above and below ATM (default 10).
        """
        if not expiry_date:
            exps = self.get_expiry(underlying, exchange)
            if not exps:
                logger.warning("No expiry available for %s", underlying)
                return []
            expiry_date = self._normalize_expiry(exps[0])

        payload: dict[str, Any] = {
            "underlying": underlying,
            "exchange": exchange,
            "expiry_date": expiry_date,
            "strike_count": strike_count,
        }
        body = self._post("optionchain", payload)
        if body.get("status") != "success":
            logger.warning("Option chain API error: %s", body.get("message"))
            return []
        return body.get("data", [])

    def compute_sentiment(
        self,
        underlying: str = "NIFTY",
        exchange: str = "NFO",
        spot_price: float = 0.0,
        expiry_date: str = "",
    ) -> OptionSentiment:
        """Fetch option data and return aggregated sentiment.

        Parameters
        ----------
        underlying : str
            Underlying index (e.g. ``\"NIFTY\"``).
        exchange : str
            Exchange for F&O trading (default ``\"NFO\"``).
        spot_price : float
            Current underlying price (for ATM strike detection).
        expiry_date : str
            Symbol-format expiry (e.g. ``\"10JUL25\"``).
            If empty, uses the nearest available expiry.
        """
        # Resolve expiry
        if not expiry_date:
            exps = self.get_expiry(underlying, exchange)
            if not exps:
                logger.warning("No expiry available for %s", underlying)
                return OptionSentiment()
            expiry_date = self._normalize_expiry(exps[0])

        chain = self.get_option_chain(underlying, exchange, expiry_date)
        if not chain:
            logger.warning("Empty chain for %s %s", underlying, expiry_date)
            return OptionSentiment()

        # Separate CE / PE
        ce_data: list[dict[str, Any]] = []
        pe_data: list[dict[str, Any]] = []
        for item in chain:
            ot = (item.get("option_type") or "").upper()
            if ot == "CE":
                ce_data.append(item)
            elif ot == "PE":
                pe_data.append(item)

        total_ce_oi = sum(float(c.get("open_interest", 0)) for c in ce_data)
        total_pe_oi = sum(float(p.get("open_interest", 0)) for p in pe_data)
        pcr = (total_pe_oi / max(total_ce_oi, 1)) if total_ce_oi > 0 else 1.0

        # ATM strike + IV skew
        atm_strike = 0.0
        atm_iv = 0.0
        iv_skew = 0.0
        combined = ce_data + pe_data
        if spot_price > 0 and combined:
            combined.sort(key=lambda x: abs(float(x.get("strike", 0)) - spot_price))
            atm = combined[0]
            atm_strike = float(atm.get("strike", 0))
            if atm.get("iv"):
                atm_iv = float(atm["iv"])

            pe_sorted = sorted(pe_data, key=lambda x: float(x.get("strike", 0)))
            otm_puts = [p for p in pe_sorted if float(p.get("strike", 0)) < spot_price]
            if otm_puts and atm_iv > 0:
                otm_put_iv = float(otm_puts[-1].get("iv", 0))
                iv_skew = otm_put_iv - atm_iv

        # OI change
        oi_changes = [float(c.get("change", 0)) for c in chain if c.get("change") is not None]
        oi_chg_pct = 0.0
        if oi_changes:
            avg_chg = sum(oi_changes) / len(oi_changes)
            total_oi = total_ce_oi + total_pe_oi
            avg_oi = total_oi / max(len(chain), 1)
            oi_chg_pct = (avg_chg / max(avg_oi, 1)) * 100 if avg_oi > 0 else 0.0

        return OptionSentiment(
            pcr=round(pcr, 3),
            iv_skew=round(iv_skew, 2),
            oi_change_pct=round(oi_chg_pct, 2),
            atm_strike=atm_strike,
            atm_iv=round(atm_iv, 2),
            near_expiry=expiry_date,
        )
