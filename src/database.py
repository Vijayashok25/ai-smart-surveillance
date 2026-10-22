"""SQLite database manager for persisting surveillance alerts and events."""

import csv
import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.utils import setup_logger


class DatabaseManager:
    """Thread-safe SQLite database manager for surveillance alert records."""

    def __init__(
        self,
        db_path: Union[str, Path] = "surveillance.db",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the database manager and runs table migrations.

        Args:
            db_path: Path to the SQLite database file.
            logger: Custom logger instance.
        """
        self.db_path = Path(db_path)
        self.logger = logger or setup_logger("DatabaseManager")
        self._lock = threading.Lock()

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a SQLite connection with row factory enabled."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        """Initializes database schema and indexes."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL NOT NULL,
                        datetime_str TEXT NOT NULL,
                        rule_name TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        object_class TEXT,
                        track_id INTEGER,
                        confidence REAL,
                        message TEXT,
                        snapshot_path TEXT,
                        clip_path TEXT,
                        metadata TEXT
                    )
                    """
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts (timestamp)"
                )
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS idx_alerts_rule ON alerts (rule_name)"
                )
                conn.commit()
                self.logger.info(f"Database initialized at: {self.db_path.resolve()}")

    def insert_alert(
        self,
        timestamp: float,
        rule_name: str,
        severity: str = "MEDIUM",
        object_class: Optional[str] = None,
        track_id: Optional[int] = None,
        confidence: float = 0.0,
        message: Optional[str] = None,
        snapshot_path: Optional[str] = None,
        clip_path: Optional[str] = None,
        metadata_str: Optional[str] = None,
    ) -> int:
        """Inserts a new alert record into the database.

        Returns:
            The primary key ID of the inserted alert.
        """
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    INSERT INTO alerts (
                        timestamp, datetime_str, rule_name, severity, object_class,
                        track_id, confidence, message, snapshot_path, clip_path, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        dt_str,
                        rule_name,
                        severity,
                        object_class,
                        track_id,
                        round(float(confidence), 4),
                        message,
                        snapshot_path,
                        clip_path,
                        metadata_str,
                    ),
                )
                conn.commit()
                alert_id = cursor.lastrowid
                self.logger.debug(f"Saved alert #{alert_id} for rule '{rule_name}'")
                return alert_id

    def get_alerts(
        self,
        rule_name: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        """Queries alert records with optional filters.

        Args:
            rule_name: Filter by rule name if provided.
            start_time: Filter by minimum timestamp.
            end_time: Filter by maximum timestamp.
            limit: Maximum records to return.

        Returns:
            List of dictionaries representing alert rows.
        """
        query = "SELECT * FROM alerts WHERE 1=1"
        params: List[Any] = []

        if rule_name:
            query += " AND rule_name = ?"
            params.append(rule_name)
        if start_time is not None:
            query += " AND timestamp >= ?"
            params.append(start_time)
        if end_time is not None:
            query += " AND timestamp <= ?"
            params.append(end_time)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]

    def get_total_count(self) -> int:
        """Returns the total number of alerts stored."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM alerts")
                return cursor.fetchone()[0]

    def export_to_csv(
        self,
        output_path: Union[str, Path] = "alerts_export.csv",
        rule_name: Optional[str] = None,
    ) -> Path:
        """Exports alerts to a CSV file.

        Args:
            output_path: Target CSV file path.
            rule_name: Optional filter for a specific rule.

        Returns:
            Path to the saved CSV file.
        """
        alerts = self.get_alerts(rule_name=rule_name, limit=10000)
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if not alerts:
            # Write header only
            fieldnames = [
                "id",
                "timestamp",
                "datetime_str",
                "rule_name",
                "severity",
                "object_class",
                "track_id",
                "confidence",
                "message",
                "snapshot_path",
                "clip_path",
            ]
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
            return out_file

        fieldnames = list(alerts[0].keys())
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(alerts)

        self.logger.info(f"Exported {len(alerts)} alerts to {out_file.resolve()}")
        return out_file
