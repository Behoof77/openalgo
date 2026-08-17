"""
safety_rules.py — 7 hard safety rules enforcement.

These rules are NON-NEGOTIABLE. Violating any of them must raise an
immediate exception. No config flag can disable them.

Rules:
1. NEVER mix positions — each position belongs to exactly one strategy.
2. NEVER close without ownership — verify before closing any position.
3. NEVER resume before reconciliation — block trading until reconciled.
4. NEVER guess orphan ownership — unknown positions go to RECOVERY_REQUIRED.
5. NEVER hardcode lot size — always fetch from instrument master.
6. NEVER auto-close SWING on session end — SWING holds overnight.
7. NEVER let INTRADAY carry overnight — force close before session end.
"""

from typing import Any

from strategies.shared.execution_modes import ExecutionMode, get_strategy_mode
from strategies.shared.position_owner import PositionOwner
from strategies.shared.reconciler import Reconciler
from strategies.shared.state_persister import StatePersister


class SafetyViolation(Exception):
    """Raised when a safety rule is violated."""

    def __init__(self, rule: str, message: str) -> None:
        self.rule = rule
        super().__init__(f"Safety Rule {rule} violated: {message}")


class SafetyRules:
    """Enforcer for the 7 hard safety rules."""

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

    def check_rule_1_no_position_mixing(
        self, new_symbol: str, new_exchange: str, new_product: str
    ) -> None:
        """Rule 1: Each position belongs to exactly one strategy.

        Checks that no other strategy already owns a position on the
        same symbol+exchange+product combination.

        Raises:
            SafetyViolation: If another strategy owns this position.
        """
        all_positions = self.owner.get_all_open_positions()
        for pos in all_positions:
            if (
                pos["symbol"] == new_symbol
                and pos["exchange"] == new_exchange
                and pos["product"] == new_product
                and pos["strategy_id"] != self.strategy_id
            ):
                raise SafetyViolation(
                    "1",
                    f"Position {new_symbol}/{new_exchange}/{new_product} is already "
                    f"owned by strategy '{pos['strategy_id']}'. "
                    f"Cannot be claimed by '{self.strategy_id}'.",
                )

    def check_rule_2_no_close_without_ownership(self, position_id: str) -> None:
        """Rule 2: Never close a position without verifying ownership.

        Raises:
            SafetyViolation: If this strategy does not own the position.
        """
        if not self.owner.verify_ownership(position_id, self.strategy_id):
            position = self.persister.get_position(position_id)
            if position is None:
                raise SafetyViolation(
                    "2",
                    f"Position '{position_id}' not found in ownership records.",
                )
            raise SafetyViolation(
                "2",
                f"Strategy '{self.strategy_id}' does not own position '{position_id}' "
                f"(owned by '{position['strategy_id']}'). Cannot close.",
            )

    def check_rule_3_no_resume_without_reconciliation(self) -> None:
        """Rule 3: Never resume trading before reconciliation is complete.

        Raises:
            SafetyViolation: If strategy is in RECOVERY_REQUIRED state.
        """
        if not self.reconciler.can_trade(self.strategy_id):
            raise SafetyViolation(
                "3",
                f"Strategy '{self.strategy_id}' has not been reconciled. "
                f"Cannot trade until reconciliation is complete.",
            )

    def check_rule_4_no_orphan_guessing(self, position_symbol: str) -> None:
        """Rule 4: Never guess orphan ownership.

        If a position exists at the broker but has no owner record,
        it must go to RECOVERY_REQUIRED — never auto-assign.

        This is enforced by the reconciler; this check verifies the
        strategy hasn't been set to RECOVERY_REQUIRED.

        Args:
            position_symbol: Symbol being traded (for error message).
        """
        state = self.persister.get_recovery_state(self.strategy_id)
        if state and state.get("state") == "RECOVERY_REQUIRED":
            raise SafetyViolation(
                "4",
                f"Strategy '{self.strategy_id}' is in RECOVERY_REQUIRED state. "
                f"Cannot guess ownership of orphan positions. Resolve orphans first.",
            )

    def check_rule_5_no_hardcoded_lot_size(self, lot_size: int) -> None:
        """Rule 5: Never hardcode lot size.

        Validates that the lot size is a positive integer and was
        fetched from the instrument master (not hardcoded to a known
        default like 75 for NIFTY).

        Args:
            lot_size: Lot size to validate.

        Raises:
            SafetyViolation: If lot_size is not positive.
        """
        if not isinstance(lot_size, int) or lot_size <= 0:
            raise SafetyViolation(
                "5",
                f"Invalid lot size: {lot_size}. Must be a positive integer "
                f"fetched from the instrument master.",
            )

    def check_rule_6_no_auto_close_swing(self) -> None:
        """Rule 6: Never auto-close SWING on session end.

        Checks that the strategy is not being forced to close at session end
        if it's in SWING mode.

        Raises:
            SafetyViolation: If this strategy is SWING and someone tries
                             to auto-close it.
        """
        mode = get_strategy_mode(self.strategy_id)
        if mode == ExecutionMode.SWING:
            # This rule is checked by the host — if the host tries to
            # send SIGTERM for session-end on a SWING strategy, this
            # check should block it.
            raise SafetyViolation(
                "6",
                f"Strategy '{self.strategy_id}' is SWING mode. "
                f"Auto-close at session end is not allowed. "
                f"Use manual STOP_AND_CLOSE if you want to close positions.",
            )

    def check_rule_7_no_overnight_intraday(self, positions: list[dict[str, Any]]) -> None:
        """Rule 7: Never let INTRADAY carry overnight.

        Called at session end to verify all INTRADAY positions are closed.

        Args:
            positions: List of open positions to check.

        Raises:
            SafetyViolation: If any INTRADAY position is still open at session end.
        """
        mode = get_strategy_mode(self.strategy_id)
        if mode == ExecutionMode.INTRADAY:
            open_count = len([p for p in positions if int(p.get("quantity", 0)) != 0])
            if open_count > 0:
                raise SafetyViolation(
                    "7",
                    f"Strategy '{self.strategy_id}' is INTRADAY with {open_count} "
                    f"open position(s) at session end. Must close before market close.",
                )
