"""
recovery.py — Crash/restart recovery state machine.

States:
  STARTING      → initial state on process start
  RECONCILING   → fetching broker positions, comparing with persister
  RECOVERY_REQUIRED → orphans found, trading BLOCKED
  RECONCILED    → all positions accounted for, trading allowed
  TRADING       → actively trading
  STOPPING      → graceful shutdown in progress
  STOPPED       → fully stopped

On startup, every strategy MUST pass through RECONCILING before reaching
TRADING. If RECOVERY_REQUIRED is reached, the strategy logs the orphan
details and waits for manual intervention.

The strategy script calls RecoveryManager.startup() which:
1. Sets state to RECONCILING
2. Fetches broker positions via a provided callback
3. Runs reconciliation
4. If clean → RECONCILED → TRADING
5. If orphans → RECOVERY_REQUIRED → blocks
"""

import os
import time
from enum import Enum
from typing import Any, Callable

from strategies.shared.position_owner import PositionOwner
from strategies.shared.reconciler import Reconciler, ReconciliationResult
from strategies.shared.state_persister import StatePersister


class RecoveryState(str, Enum):
    """Recovery state machine states."""

    STARTING = "STARTING"
    RECONCILING = "RECONCILING"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"
    RECONCILED = "RECONCILED"
    TRADING = "TRADING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"


class RecoveryManager:
    """Crash/restart recovery state machine."""

    def __init__(
        self,
        strategy_id: str,
        persister: StatePersister | None = None,
        owner: PositionOwner | None = None,
        reconciler: Reconciler | None = None,
    ) -> None:
        self.strategy_id = strategy_id
        self.persister = persister or StatePersister()
        self.owner = owner or PositionOwner(self.persister)
        self.reconciler = reconciler or Reconciler(self.persister, self.owner)
        self.state = RecoveryState.STARTING

    def startup(
        self,
        fetch_broker_positions: Callable[[], list[dict[str, Any]]],
    ) -> ReconciliationResult:
        """Run the startup reconciliation flow.

        Args:
            fetch_broker_positions: Callable that returns current positions
                                    from the broker API (list of dicts with
                                    symbol, exchange, quantity, product).

        Returns:
            ReconciliationResult with classified positions.

        Raises:
            RuntimeError: If recovery is required (orphans found).
        """
        print(f"[recovery] {self.strategy_id}: Starting reconciliation...")
        self.state = RecoveryState.RECONCILING
        self.persister.save_recovery_state(self.strategy_id, "RECONCILING")

        # Fetch broker positions
        try:
            broker_positions = fetch_broker_positions()
        except Exception as e:
            print(f"[recovery] {self.strategy_id}: Failed to fetch broker positions: {e}")
            self.persister.save_recovery_state(
                self.strategy_id, "RECOVERY_REQUIRED", f"API error: {e}"
            )
            self.state = RecoveryState.RECOVERY_REQUIRED
            raise RuntimeError(
                f"Cannot fetch broker positions for reconciliation: {e}"
            ) from e

        # Reconcile
        result = self.reconciler.reconcile(self.strategy_id, broker_positions)

        if result.is_clean:
            self.state = RecoveryState.RECONCILED
            print(f"[recovery] {self.strategy_id}: Reconciliation clean. Trading allowed.")
        else:
            self.state = RecoveryState.RECOVERY_REQUIRED
            print(
                f"[recovery] {self.strategy_id}: RECOVERY REQUIRED. "
                f"Orphans: {[p.get('symbol', '?') for p in result.orphan]}. "
                f"Trading BLOCKED until orphans are resolved."
            )

        return result

    def begin_trading(self) -> None:
        """Transition to TRADING state.

        Must be called after successful reconciliation.
        """
        if self.state != RecoveryState.RECONCILED:
            raise RuntimeError(
                f"Cannot begin trading from state {self.state.value}. "
                f"Must be RECONCILED first."
            )
        self.state = RecoveryState.TRADING
        print(f"[recovery] {self.strategy_id}: Trading started.")

    def begin_stopping(self) -> None:
        """Transition to STOPPING state (graceful shutdown)."""
        self.state = RecoveryState.STOPPING
        print(f"[recovery] {self.strategy_id}: Stopping...")

    def complete_stop(self) -> None:
        """Transition to STOPPED state."""
        self.state = RecoveryState.STOPPED
        print(f"[recovery] {self.strategy_id}: Stopped.")

    def can_trade(self) -> bool:
        """Check if the strategy is in a tradeable state."""
        return self.state == RecoveryState.TRADING

    def is_recovered(self) -> bool:
        """Check if recovery is complete."""
        return self.state in (RecoveryState.RECONCILED, RecoveryState.TRADING)
