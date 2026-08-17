"""
reconciler.py — Broker-to-strategy reconciliation on startup.

SAFETY RULE: Never resume trading before reconciliation is complete.

On every strategy startup (or restart after crash), the reconciler:
1. Fetches current positions from the broker
2. Compares with owned positions in the persister
3. Classifies each position:
   - OWNED: broker position matches a persister record → safe to manage
   - ORPHAN: broker position has no owner → RECOVERY_REQUIRED
   - STALE: persister has a position the broker doesn't → marked CLOSED
4. Blocks trading if any RECOVERY_REQUIRED state exists

This prevents the deadly scenario of multiple strategies fighting over
the same positions, or a restarted strategy closing another strategy's positions.
"""

import os
import time
from typing import Any

from strategies.shared.position_owner import PositionOwner
from strategies.shared.state_persister import StatePersister


class ReconciliationResult:
    """Result of a reconciliation pass."""

    def __init__(self) -> None:
        self.owned: list[dict[str, Any]] = []
        self.orphan: list[dict[str, Any]] = []
        self.stale: list[dict[str, Any]] = []
        self.is_clean: bool = True
        self.blocked_reason: str = ""

    def __repr__(self) -> str:
        return (
            f"ReconcileResult(owned={len(self.owned)}, orphan={len(self.orphan)}, "
            f"stale={len(self.stale)}, clean={self.is_clean})"
        )


class Reconciler:
    """Broker-to-strategy position reconciler."""

    def __init__(
        self,
        persister: StatePersister | None = None,
        owner: PositionOwner | None = None,
    ) -> None:
        self.persister = persister or StatePersister()
        self.owner = owner or PositionOwner(self.persister)

    def reconcile(
        self,
        strategy_id: str,
        broker_positions: list[dict[str, Any]],
    ) -> ReconciliationResult:
        """Reconcile broker positions against owned positions for a strategy.

        Args:
            strategy_id: The strategy performing reconciliation.
            broker_positions: Current positions from broker API
                              (list of dicts with symbol, exchange, quantity, product).

        Returns:
            ReconciliationResult with classified positions.
        """
        result = ReconciliationResult()

        # Get owned positions from persister
        owned = self.persister.get_positions_by_strategy(strategy_id, status="OPEN")
        owned_map = {(p["symbol"], p["exchange"], p["product"]): p for p in owned}

        broker_map = {}
        for bp in broker_positions:
            sym = bp.get("symbol", "")
            exch = bp.get("exchange", "")
            prod = bp.get("product", "")
            qty = int(bp.get("quantity", 0))
            if qty != 0:
                broker_map[(sym, exch, prod)] = bp

        # Classify broker positions
        for key, bp in broker_map.items():
            if key in owned_map:
                # Position is owned by this strategy
                result.owned.append(bp)
            else:
                # Position has no owner in persister
                result.orphan.append(bp)

        # Find stale positions (in persister but not at broker)
        for key, op in owned_map.items():
            if key not in broker_map:
                result.stale.append(op)
                # Mark stale position as closed
                self.persister.release_position(op["position_id"])

        # Block trading if orphans exist
        if result.orphan:
            result.is_clean = False
            result.blocked_reason = (
                f"Found {len(result.orphan)} orphan position(s) at broker "
                f"with no owner. RECOVERY REQUIRED."
            )
            # Save recovery state
            self.persister.save_recovery_state(
                strategy_id,
                "RECOVERY_REQUIRED",
                details=f"Orphans: {[p.get('symbol', '?') for p in result.orphan]}",
            )
            # Mark orphans in persister
            for bp in result.orphan:
                self.persister.mark_orphan(
                    f"orphan_{bp.get('symbol', 'unknown')}_{int(time.time())}"
                )

        # Mark as reconciled if clean
        if result.is_clean:
            self.persister.save_recovery_state(strategy_id, "RECONCILED")

        print(f"[reconciler] {strategy_id}: {result}")
        return result

    def can_trade(self, strategy_id: str) -> bool:
        """Check if a strategy is allowed to trade.

        Returns False if the strategy is in RECOVERY_REQUIRED state.

        Args:
            strategy_id: Strategy to check.

        Returns:
            True if safe to trade.
        """
        state = self.persister.get_recovery_state(strategy_id)
        if state is None:
            return True  # No recovery state = clean
        return state.get("state") == "RECONCILED"

    def force_clear_recovery(self, strategy_id: str) -> None:
        """Force-clear recovery state (manual intervention).

        Call this only after the user has manually resolved orphan positions.

        Args:
            strategy_id: Strategy to clear.
        """
        self.persister.save_recovery_state(strategy_id, "RECONCILED", "Force cleared by user")
        print(f"[reconciler] {strategy_id}: Force-cleared recovery state")
