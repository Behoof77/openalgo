"""
position_owner.py — Position ownership tracking.

Every position/order must carry ownership metadata:
  - strategy_id: which strategy owns this position
  - position_id: unique identifier for this position
  - entry_order_id: the order that opened this position

This module wraps StatePersister to provide a clean API for:
  - Registering new positions on order fill
  - Querying positions by strategy
  - Verifying ownership before any close/modify
  - Releasing positions on exit

SAFETY RULE: Never close a position without verifying ownership first.
"""

import os
import time
import uuid
from typing import Any

from strategies.shared.state_persister import StatePersister


def _generate_position_id(strategy_id: str, symbol: str) -> str:
    """Generate a unique position ID.

    Format: {strategy_id}_{symbol}_{timestamp_short}_{random}
    """
    ts = int(time.time())
    short_rand = uuid.uuid4().hex[:6]
    return f"{strategy_id}_{symbol}_{ts}_{short_rand}"


class PositionOwner:
    """Position ownership manager."""

    def __init__(self, persister: StatePersister | None = None) -> None:
        """Initialize the position owner.

        Args:
            persister: StatePersister instance. Creates default if None.
        """
        self.persister = persister or StatePersister()

    def register_position(
        self,
        strategy_id: str,
        symbol: str,
        exchange: str,
        quantity: int,
        product: str,
        entry_order_id: str = "",
        position_id: str | None = None,
    ) -> str:
        """Register a new position as owned by a strategy.

        Called when an order fill is detected.

        Args:
            strategy_id: Owning strategy's ID.
            symbol: Trading symbol.
            exchange: Exchange code.
            quantity: Position quantity (positive for long).
            product: MIS/CNC/NRML.
            entry_order_id: Order ID that opened this position.
            position_id: Optional pre-generated position ID.

        Returns:
            The position_id for tracking.
        """
        if position_id is None:
            position_id = _generate_position_id(strategy_id, symbol)

        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self.persister.save_position(
            {
                "position_id": position_id,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "exchange": exchange,
                "quantity": quantity,
                "product": product,
                "entry_order_id": entry_order_id,
                "entry_time": now,
                "status": "OPEN",
            }
        )
        print(f"[owner] Registered position: {position_id} owned by {strategy_id}")
        return position_id

    def verify_ownership(self, position_id: str, strategy_id: str) -> bool:
        """Verify that a strategy owns a position.

        SAFETY RULE: Always call this before closing/modifying any position.

        Args:
            position_id: Position to verify.
            strategy_id: Strategy claiming ownership.

        Returns:
            True if the strategy owns this position.
        """
        position = self.persister.get_position(position_id)
        if position is None:
            return False
        return (
            position["strategy_id"] == strategy_id
            and position["status"] == "OPEN"
        )

    def get_positions_by_strategy(self, strategy_id: str) -> list[dict[str, Any]]:
        """Get all open positions for a strategy.

        Args:
            strategy_id: Strategy identifier.

        Returns:
            List of position dicts.
        """
        return self.persister.get_positions_by_strategy(strategy_id, status="OPEN")

    def release_position(self, position_id: str, strategy_id: str) -> bool:
        """Release a position (mark as closed) after verifying ownership.

        SAFETY RULE: Only the owning strategy can release its positions.

        Args:
            position_id: Position to release.
            strategy_id: Strategy requesting release.

        Returns:
            True if released, False if not owned by this strategy.

        Raises:
            PermissionError: If strategy does not own the position.
        """
        if not self.verify_ownership(position_id, strategy_id):
            raise PermissionError(
                f"Strategy '{strategy_id}' does not own position '{position_id}'. "
                f"Cannot release a position you don't own."
            )
        self.persister.release_position(position_id)
        print(f"[owner] Released position: {position_id} by {strategy_id}")
        return True

    def release_all_for_strategy(self, strategy_id: str) -> int:
        """Release all open positions for a strategy.

        Called during graceful shutdown (STOP_AND_CLOSE mode).

        Args:
            strategy_id: Strategy to release all positions for.

        Returns:
            Number of positions released.
        """
        positions = self.get_positions_by_strategy(strategy_id)
        count = 0
        for pos in positions:
            self.persister.release_position(pos["position_id"])
            count += 1
        if count > 0:
            print(f"[owner] Released {count} positions for {strategy_id}")
        return count

    def get_all_open_positions(self) -> list[dict[str, Any]]:
        """Get all open positions across all strategies."""
        return self.persister.get_all_open_positions()

    def get_unowned_positions(
        self, broker_positions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Find positions from the broker that have no owner in the persister.

        Used during reconciliation to identify orphan positions.

        Args:
            broker_positions: List of positions from the broker API.

        Returns:
            List of broker positions with no matching owner record.
        """
        owned = self.persister.get_all_open_positions()
        owned_keys = {
            (p["symbol"], p["exchange"], p["product"]) for p in owned
        }

        orphans = []
        for bp in broker_positions:
            key = (
                bp.get("symbol", ""),
                bp.get("exchange", ""),
                bp.get("product", ""),
            )
            if key not in owned_keys and int(bp.get("quantity", 0)) != 0:
                orphans.append(bp)

        return orphans
