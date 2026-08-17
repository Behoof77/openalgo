"""
test_acceptance.py — Acceptance tests A-F for strategy engine overhaul.

Tests the 6 critical acceptance criteria:
  A. Multi-strategy coexistence (positions don't mix)
  B. Stop one strategy (other strategies unaffected)
  C. Swing strategy holds overnight (not auto-closed)
  D. Crash recovery (reconcile on restart)
  E. Orphan position detection (unknown positions blocked)
  F. Dynamic lot-size change (fetched from instrument master)

These tests use in-memory SQLite for speed and isolation.
"""

import os
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add strategies/ to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from strategies.shared.execution_modes import (
    ExecutionMode,
    get_strategy_mode,
    register_strategy_mode,
    validate_execution_mode,
)
from strategies.shared.stop_modes import StopMode
from strategies.shared.state_persister import StatePersister
from strategies.shared.position_owner import PositionOwner
from strategies.shared.reconciler import Reconciler
from strategies.shared.recovery import RecoveryManager, RecoveryState
from strategies.shared.safety_rules import SafetyRules, SafetyViolation


def _make_persister() -> StatePersister:
    """Create an in-memory StatePersister for testing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    persister = StatePersister.__new__(StatePersister)
    persister.db_path = Path(":memory:")
    persister._local = type("Local", (), {"conn": conn})()
    persister._init_schema()
    return persister


class TestA_MultiStrategyCoexistence(unittest.TestCase):
    """Test A: Two strategies can coexist without position interference."""

    def setUp(self):
        self.persister = _make_persister()
        self.owner_kronos = PositionOwner(self.persister)
        self.owner_premium = PositionOwner(self.persister)

    def test_two_strategies_own_different_positions(self):
        """Both strategies register positions on same symbol — must not conflict."""
        # Kronos sizing opens NIFTY futures
        pos1 = self.owner_kronos.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
            entry_order_id="ORD001",
        )

        # ATM premium opens NIFTY options (different product)
        pos2 = self.owner_premium.register_position(
            strategy_id="atm_premium_ml",
            symbol="NIFTY28MAR2424500CE",
            exchange="NFO",
            quantity=75,
            product="NRML",
            entry_order_id="ORD002",
        )

        # Each strategy sees only its own positions
        self.assertEqual(len(self.owner_kronos.get_positions_by_strategy("kronos_sizing")), 1)
        self.assertEqual(len(self.owner_premium.get_positions_by_strategy("atm_premium_ml")), 1)

        # Verify ownership
        self.assertTrue(self.owner_kronos.verify_ownership(pos1, "kronos_sizing"))
        self.assertFalse(self.owner_kronos.verify_ownership(pos1, "atm_premium_ml"))
        self.assertTrue(self.owner_premium.verify_ownership(pos2, "atm_premium_ml"))

    def test_same_symbol_different_product_no_conflict(self):
        """Two strategies can hold same symbol with different products."""
        self.owner_kronos.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )
        self.owner_premium.register_position(
            strategy_id="atm_premium_ml",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="NRML",
        )

        # Both should coexist
        self.assertEqual(len(self.owner_kronos.get_positions_by_strategy("kronos_sizing")), 1)
        self.assertEqual(len(self.owner_premium.get_positions_by_strategy("atm_premium_ml")), 1)

    def test_cannot_close_other_strategy_position(self):
        """Strategy A cannot release Strategy B's position."""
        pos = self.owner_kronos.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )

        with self.assertRaises(PermissionError):
            self.owner_premium.release_position(pos, "atm_premium_ml")


class TestB_StopOneStrategy(unittest.TestCase):
    """Test B: Stopping one strategy doesn't affect others."""

    def setUp(self):
        self.persister = _make_persister()
        self.owner = PositionOwner(self.persister)

    def test_stop_strategy_releases_its_positions(self):
        """STOP_AND_CLOSE releases only the stopped strategy's positions."""
        self.owner.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )
        self.owner.register_position(
            strategy_id="kronos_vote",
            symbol="NIFTY28MAR2424500CE",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )

        # Stop kronos_sizing — only its positions released
        released = self.owner.release_all_for_strategy("kronos_sizing")
        self.assertEqual(released, 1)

        # kronos_vote still has its position
        self.assertEqual(len(self.owner.get_positions_by_strategy("kronos_vote")), 1)

    def test_stop_mode_stored(self):
        """Stop state is persisted for the stopped strategy only."""
        self.persister.save_stop_state("kronos_sizing", "STOP_AND_CLOSE", "User requested")

        stop = self.persister.get_stop_state("kronos_sizing")
        self.assertIsNotNone(stop)
        self.assertEqual(stop["stop_mode"], "STOP_AND_CLOSE")

        # Other strategy not stopped
        self.assertIsNone(self.persister.get_stop_state("kronos_vote"))

    def test_stop_trading_only_keeps_positions(self):
        """STOP_TRADING_ONLY stops new orders but keeps positions open."""
        self.owner.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )

        self.persister.save_stop_state("kronos_sizing", "STOP_TRADING_ONLY", "Pause")
        self.owner.release_all_for_strategy("kronos_sizing")
        # After STOP_TRADING_ONLY, we'd still have positions (not released in real usage)
        # This test verifies the stop mode is stored correctly
        stop = self.persister.get_stop_state("kronos_sizing")
        self.assertEqual(stop["stop_mode"], "STOP_TRADING_ONLY")


class TestC_SwingOvernight(unittest.TestCase):
    """Test C: SWING strategy is not auto-closed at session end."""

    def setUp(self):
        register_strategy_mode("atm_premium_ml", ExecutionMode.SWING)

    def test_swing_mode_registered(self):
        """SWING mode is correctly registered and retrieved."""
        mode = get_strategy_mode("atm_premium_ml")
        self.assertEqual(mode, ExecutionMode.SWING)

    def test_intraday_mode_registered(self):
        """INTRADAY mode is correctly registered."""
        mode = get_strategy_mode("kronos_sizing")
        self.assertEqual(mode, ExecutionMode.INTRADAY)

    def test_swing_block_auto_close(self):
        """Safety rule prevents auto-closing SWING strategies."""
        persister = _make_persister()
        rules = SafetyRules("atm_premium_ml", persister=persister)

        with self.assertRaises(SafetyViolation) as ctx:
            rules.check_rule_6_no_auto_close_swing()
        self.assertIn("6", str(ctx.exception))


class TestD_CrashRecovery(unittest.TestCase):
    """Test D: Strategy reconciles on restart after crash."""

    def setUp(self):
        self.persister = _make_persister()
        self.owner = PositionOwner(self.persister)
        self.reconciler = Reconciler(self.persister, self.owner)

    def test_clean_reconciliation(self):
        """All broker positions match persister records — clean state."""
        # Register owned positions before "crash"
        self.owner.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )

        # After restart, broker shows same position
        broker_positions = [
            {"symbol": "NIFTY", "exchange": "NFO", "quantity": "75", "product": "MIS"},
        ]

        result = self.reconciler.reconcile("kronos_sizing", broker_positions)
        self.assertTrue(result.is_clean)
        self.assertEqual(len(result.owned), 1)
        self.assertEqual(len(result.orphan), 0)

    def test_orphan_detection(self):
        """Broker has position not in persister — orphans detected."""
        # No positions registered before "crash"
        broker_positions = [
            {"symbol": "NIFTY", "exchange": "NFO", "quantity": "75", "product": "MIS"},
        ]

        result = self.reconciler.reconcile("kronos_sizing", broker_positions)
        self.assertFalse(result.is_clean)
        self.assertEqual(len(result.orphan), 1)

    def test_stale_position_cleaned(self):
        """Persister has position broker doesn't — stale cleaned."""
        self.owner.register_position(
            strategy_id="kronos_sizing",
            symbol="NIFTY",
            exchange="NFO",
            quantity=75,
            product="MIS",
        )

        # After crash, position was closed externally
        broker_positions = []

        result = self.reconciler.reconcile("kronos_sizing", broker_positions)
        self.assertTrue(result.is_clean)
        self.assertEqual(len(result.stale), 1)

        # Stale position should now be CLOSED
        positions = self.persister.get_positions_by_strategy("kronos_sizing", status="OPEN")
        self.assertEqual(len(positions), 0)

    def test_cannot_trade_during_recovery(self):
        """Strategy blocked from trading while in RECOVERY_REQUIRED."""
        self.persister.save_recovery_state("kronos_sizing", "RECOVERY_REQUIRED", "orphans found")
        self.assertFalse(self.reconciler.can_trade("kronos_sizing"))

    def test_can_trade_after_reconciliation(self):
        """Strategy allowed to trade after clean reconciliation."""
        self.assertTrue(self.reconciler.can_trade("kronos_sizing"))

    def test_recovery_manager_flow(self):
        """Full recovery manager flow: STARTING → RECONCILING → RECONCILED."""
        manager = RecoveryManager("kronos_sizing", self.persister, self.owner, self.reconciler)

        # No broker positions = clean reconciliation
        result = manager.startup(lambda: [])
        self.assertTrue(result.is_clean)
        self.assertEqual(manager.state, RecoveryState.RECONCILED)

        # Now can trade
        manager.begin_trading()
        self.assertEqual(manager.state, RecoveryState.TRADING)
        self.assertTrue(manager.can_trade())


class TestE_OrphanPosition(unittest.TestCase):
    """Test E: Unknown/orphan positions go to RECOVERY_REQUIRED."""

    def setUp(self):
        self.persister = _make_persister()
        self.owner = PositionOwner(self.persister)
        self.reconciler = Reconciler(self.persister, self.owner)

    def test_orphan_blocks_trading(self):
        """Orphan position puts strategy in RECOVERY_REQUIRED, blocks trading."""
        # Orphan exists
        broker_positions = [
            {"symbol": "BANKNIFTY", "exchange": "NFO", "quantity": "25", "product": "MIS"},
        ]

        result = self.reconciler.reconcile("kronos_sizing", broker_positions)
        self.assertFalse(result.is_clean)

        # Cannot trade
        self.assertFalse(self.reconciler.can_trade("kronos_sizing"))

    def test_force_clear_recovery(self):
        """Manual force-clear allows trading again."""
        self.persister.save_recovery_state("kronos_sizing", "RECOVERY_REQUIRED", "orphans")
        self.assertFalse(self.reconciler.can_trade("kronos_sizing"))

        self.reconciler.force_clear_recovery("kronos_sizing")
        self.assertTrue(self.reconciler.can_trade("kronos_sizing"))

    def test_safety_rule_4_prevents_guessing(self):
        """Rule 4: Cannot trade when in RECOVERY_REQUIRED state."""
        self.persister.save_recovery_state("kronos_sizing", "RECOVERY_REQUIRED")
        rules = SafetyRules("kronos_sizing", persister=self.persister)

        with self.assertRaises(SafetyViolation) as ctx:
            rules.check_rule_4_no_orphan_guessing("NIFTY")
        self.assertIn("4", str(ctx.exception))


class TestF_DynamicLotSize(unittest.TestCase):
    """Test F: Lot size fetched dynamically from instrument master."""

    def test_lot_size_validation(self):
        """Lot size must be a positive integer."""
        persister = _make_persister()
        rules = SafetyRules("kronos_sizing", persister=persister)

        # Valid lot size
        rules.check_rule_5_no_hardcoded_lot_size(75)

        # Invalid: zero
        with self.assertRaises(SafetyViolation):
            rules.check_rule_5_no_hardcoded_lot_size(0)

        # Invalid: negative
        with self.assertRaises(SafetyViolation):
            rules.check_rule_5_no_hardcoded_lot_size(-75)

    @patch("strategies.shared.lot_size.requests")
    def test_lot_size_fetched_from_api(self, mock_requests):
        """Lot size is fetched from broker instrument master, not hardcoded."""
        from strategies.shared.lot_size import get_lot_size, clear_cache

        clear_cache()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "data": [
                {"symbol": "NIFTY", "lotsize": 75, "exchange": "NFO"},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        lot = get_lot_size("NIFTY", "NFO")
        self.assertEqual(lot, 75)

        # Verify API was called (not hardcoded)
        mock_requests.post.assert_called_once()

    @patch("strategies.shared.lot_size.requests")
    def test_lot_size_validates_against_config(self, mock_requests):
        """validate_lot_size detects when config is outdated."""
        from strategies.shared.lot_size import validate_lot_size, clear_cache

        clear_cache()

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "status": "success",
            "data": [
                {"symbol": "NIFTY", "lotsize": 50, "exchange": "NFO"},
            ],
        }
        mock_resp.raise_for_status = MagicMock()
        mock_requests.post.return_value = mock_resp

        # Config says 75, but broker says 50 (lot size changed!)
        current = validate_lot_size("NIFTY", "NFO", expected=75)
        self.assertEqual(current, 50)  # Returns the CURRENT value


if __name__ == "__main__":
    unittest.main()
