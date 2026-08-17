"""
state_persister.py — SQLite-backed state persistence for strategy engine.

Stores:
  - Position ownership records (which strategy owns which position)
  - Recovery states (which strategies need reconciliation)
  - Stop states (which strategies are stopped, in which mode)

Uses a simple SQLite database with WAL mode for concurrent read safety.
Each strategy's state is isolated by strategy_id.
"""

import json
import os
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

# Default DB path — in strategies/state/ directory
_DEFAULT_DB_DIR = Path(__file__).parent.parent / "state"


class StatePersister:
    """SQLite-backed state store for strategy engine."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Initialize the state persister.

        Args:
            db_path: Path to SQLite database file. Defaults to strategies/state/engine.db.
        """
        if db_path is None:
            db_path = os.getenv(
                "STRATEGY_STATE_DB",
                str(_DEFAULT_DB_DIR / "engine.db"),
            )
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            conn = sqlite3.connect(str(self.db_path), timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return self._local.conn

    @contextmanager
    def _cursor(self) -> Generator[sqlite3.Cursor, None, None]:
        """Context manager for database cursor with auto-commit."""
        conn = self._get_conn()
        cur = conn.cursor()
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        with self._cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS position_ownership (
                    position_id TEXT PRIMARY KEY,
                    strategy_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    exchange TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    product TEXT NOT NULL,
                    entry_order_id TEXT,
                    entry_time TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'OPEN'
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS recovery_states (
                    strategy_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'UNKNOWN',
                    details TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS stop_states (
                    strategy_id TEXT PRIMARY KEY,
                    stop_mode TEXT NOT NULL,
                    stopped_at TEXT NOT NULL,
                    reason TEXT
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ownership_strategy
                ON position_ownership(strategy_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_ownership_status
                ON position_ownership(status)
            """)

    # --- Position Ownership ---

    def save_position(self, position: dict[str, Any]) -> None:
        """Save or update a position ownership record.

        Args:
            position: Dict with keys: position_id, strategy_id, symbol,
                      exchange, quantity, product, entry_order_id,
                      entry_time, status.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO position_ownership
                    (position_id, strategy_id, symbol, exchange, quantity,
                     product, entry_order_id, entry_time, last_updated, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    position["position_id"],
                    position["strategy_id"],
                    position["symbol"],
                    position["exchange"],
                    position["quantity"],
                    position["product"],
                    position.get("entry_order_id", ""),
                    position.get("entry_time", now),
                    now,
                    position.get("status", "OPEN"),
                ),
            )

    def get_positions_by_strategy(
        self, strategy_id: str, status: str = "OPEN"
    ) -> list[dict[str, Any]]:
        """Get all positions for a strategy.

        Args:
            strategy_id: Strategy identifier.
            status: Filter by status (default: "OPEN").

        Returns:
            List of position dicts.
        """
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM position_ownership WHERE strategy_id = ? AND status = ?",
                (strategy_id, status),
            )
            return [dict(row) for row in cur.fetchall()]

    def get_all_open_positions(self) -> list[dict[str, Any]]:
        """Get all open positions across all strategies."""
        return self.get_positions_by_strategy("%", status="OPEN") if False else self._get_all_open()

    def _get_all_open(self) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM position_ownership WHERE status = 'OPEN'"
            )
            return [dict(row) for row in cur.fetchall()]

    def release_position(self, position_id: str) -> None:
        """Mark a position as closed.

        Args:
            position_id: Position identifier.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._cursor() as cur:
            cur.execute(
                "UPDATE position_ownership SET status = 'CLOSED', last_updated = ? WHERE position_id = ?",
                (now, position_id),
            )

    def get_position(self, position_id: str) -> dict[str, Any] | None:
        """Get a specific position by ID."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM position_ownership WHERE position_id = ?",
                (position_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def get_orphan_positions(self) -> list[dict[str, Any]]:
        """Get positions with status ORPHAN (no strategy claims them)."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM position_ownership WHERE status = 'ORPHAN'"
            )
            return [dict(row) for row in cur.fetchall()]

    def mark_orphan(self, position_id: str) -> None:
        """Mark a position as orphan (no known strategy owner)."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._cursor() as cur:
            cur.execute(
                "UPDATE position_ownership SET status = 'ORPHAN', last_updated = ? WHERE position_id = ?",
                (now, position_id),
            )

    # --- Recovery States ---

    def save_recovery_state(self, strategy_id: str, state: str, details: str = "") -> None:
        """Save recovery state for a strategy.

        States: UNKNOWN, RECOVERY_REQUIRED, RECONCILED, FAILED.
        """
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO recovery_states (strategy_id, state, details, timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (strategy_id, state, details, now),
            )

    def get_recovery_state(self, strategy_id: str) -> dict[str, Any] | None:
        """Get recovery state for a strategy."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM recovery_states WHERE strategy_id = ?",
                (strategy_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    # --- Stop States ---

    def save_stop_state(self, strategy_id: str, stop_mode: str, reason: str = "") -> None:
        """Save stop state for a strategy."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        with self._cursor() as cur:
            cur.execute(
                """
                INSERT OR REPLACE INTO stop_states (strategy_id, stop_mode, stopped_at, reason)
                VALUES (?, ?, ?, ?)
                """,
                (strategy_id, stop_mode, now, reason),
            )

    def get_stop_state(self, strategy_id: str) -> dict[str, Any] | None:
        """Get stop state for a strategy. Returns None if not stopped."""
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM stop_states WHERE strategy_id = ?",
                (strategy_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None

    def clear_stop_state(self, strategy_id: str) -> None:
        """Clear stop state for a strategy (resume trading)."""
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM stop_states WHERE strategy_id = ?",
                (strategy_id,),
            )

    # --- Cleanup ---

    def close(self) -> None:
        """Close the thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None
