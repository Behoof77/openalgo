"""Broker bridge — places, modifies, and cancels orders via the OpenAlgo
Order API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from kronos.utils.config import KronosConfig
from kronos.utils.helpers import logger

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


@dataclass
class OrderResult:
    """Result returned by every order operation."""

    success: bool
    order_id: str | None = None
    message: str = ""
    raw: dict[str, Any] | None = None


@dataclass
class PlaceOrderRequest:
    """Required and optional fields for placing an order."""

    strategy: str = "kronos"
    exchange: str = "NSE"
    symbol: str = ""
    action: str = "BUY"  # BUY | SELL
    quantity: int = 1
    product: str = "MIS"  # MIS | NRML | CNC
    pricetype: str = "MARKET"  # MARKET | LIMIT | SL | SL-M
    price: float = 0.0
    trigger_price: float = 0.0
    disclosed_quantity: int = 0


class OpenAlgoBrokerBridge:
    """Thin wrapper around OpenAlgo's order-related REST endpoints.

    Every public method returns an ``OrderResult`` dataclass — callers
    should check ``.success`` before inspecting ``.order_id``.
    """

    def __init__(self, config: KronosConfig | None = None) -> None:
        self.config = config or KronosConfig.from_env()
        self._client: Any | None = None  # httpx.Client

    # ── HTTP lifecycle ───────────────────────────────────────────────

    @property
    def client(self) -> Any:  # httpx.Client
        if self._client is None:
            if httpx is None:
                raise RuntimeError("httpx is not installed — run `pip install httpx`")
            self._client = httpx.Client(timeout=httpx.Timeout(15.0))
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> OpenAlgoBrokerBridge:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    # ── internal helpers ─────────────────────────────────────────────

    def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST *payload* to ``/api/v1/{endpoint}/`` and return the
        parsed JSON response body."""
        url = f"{self.config.openalgo_host}/api/v1/{endpoint}/"
        payload.setdefault("apikey", self.config.openalgo_api_key)

        logger.debug("POST %s %s", url, {k: v for k, v in payload.items() if k != "apikey"})
        resp = self.client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()

    def _make_result(self, body: dict[str, Any]) -> OrderResult:
        return OrderResult(
            success=body.get("status") == "success",
            order_id=body.get("orderid") or body.get("order_id") or body.get("data", {}).get("orderid"),
            message=body.get("message", ""),
            raw=body,
        )

    # ── public API ───────────────────────────────────────────────────

    def place_order(self, req: PlaceOrderRequest) -> OrderResult:
        """Place a new order via ``/api/v1/placeorder/``."""
        payload: dict[str, Any] = {
            "strategy": req.strategy,
            "exchange": req.exchange,
            "symbol": req.symbol,
            "action": req.action,
            "quantity": req.quantity,
            "product": req.product,
            "pricetype": req.pricetype,
            "price": req.price,
            "trigger_price": req.trigger_price,
            "disclosed_quantity": req.disclosed_quantity,
        }
        body = self._post("placeorder", payload)
        return self._make_result(body)

    def modify_order(
        self,
        order_id: str,
        req: PlaceOrderRequest,
    ) -> OrderResult:
        """Modify an existing order via ``/api/v1/modifyorder/``."""
        payload: dict[str, Any] = {
            "strategy": req.strategy,
            "exchange": req.exchange,
            "symbol": req.symbol,
            "orderid": order_id,
            "action": req.action,
            "quantity": req.quantity,
            "product": req.product,
            "pricetype": req.pricetype,
            "price": req.price,
            "trigger_price": req.trigger_price,
            "disclosed_quantity": req.disclosed_quantity,
        }
        body = self._post("modifyorder", payload)
        return self._make_result(body)

    def cancel_order(self, order_id: str, strategy: str = "kronos") -> OrderResult:
        """Cancel an order via ``/api/v1/cancelorder/``."""
        body = self._post("cancelorder", {
            "strategy": strategy,
            "orderid": order_id,
        })
        return self._make_result(body)

    def get_orderbook(self) -> list[dict[str, Any]]:
        """Return the current order book (``/api/v1/orderbook/``)."""
        body = self._post("orderbook", {"strategy": "kronos"})
        if body.get("status") == "success":
            return body.get("data", [])
        logger.warning("Failed to fetch orderbook: %s", body.get("message"))
        return []

    def get_positions(self) -> list[dict[str, Any]]:
        """Return open positions (``/api/v1/positionbook/``)."""
        body = self._post("positionbook", {"strategy": "kronos"})
        if body.get("status") == "success":
            return body.get("data", [])
        logger.warning("Failed to fetch positions: %s", body.get("message"))
        return []

    def get_funds(self) -> dict[str, Any]:
        """Return account funds / margin (``/api/v1/funds/``)."""
        body = self._post("funds", {"strategy": "kronos"})
        if body.get("status") == "success":
            return body.get("data", {})
        logger.warning("Failed to fetch funds: %s", body.get("message"))
        return {}

    def get_holdings(self) -> list[dict[str, Any]]:
        """Return delivery holdings (``/api/v1/holdings/``)."""
        body = self._post("holdings", {"strategy": "kronos"})
        if body.get("status") == "success":
            return body.get("data", [])
        logger.warning("Failed to fetch holdings: %s", body.get("message"))
        return []
