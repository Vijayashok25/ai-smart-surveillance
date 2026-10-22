"""Unit tests for SQLite database manager and alert persistence."""

from pathlib import Path
import pytest

from src.database import DatabaseManager


def test_database_insert_and_query(tmp_path: Path):
    """Test inserting alert records and querying them back."""
    db_file = tmp_path / "test_surveillance.db"
    db = DatabaseManager(db_path=db_file)

    # Insert 2 alerts
    id1 = db.insert_alert(
        timestamp=1000.0,
        rule_name="Restricted Zone Intrusion",
        severity="HIGH",
        object_class="person",
        track_id=1,
        confidence=0.92,
        message="Intrusion in Zone A",
    )
    id2 = db.insert_alert(
        timestamp=1005.0,
        rule_name="Loitering Detected",
        severity="MEDIUM",
        object_class="person",
        track_id=2,
        confidence=0.88,
        message="Person loitering",
    )

    assert id1 == 1
    assert id2 == 2
    assert db.get_total_count() == 2

    # Query all
    records = db.get_alerts()
    assert len(records) == 2

    # Filter by rule
    intrusion_alerts = db.get_alerts(rule_name="Restricted Zone Intrusion")
    assert len(intrusion_alerts) == 1
    assert intrusion_alerts[0]["track_id"] == 1


def test_database_csv_export(tmp_path: Path):
    """Test exporting database alerts to CSV."""
    db_file = tmp_path / "test_surveillance.db"
    csv_file = tmp_path / "exported_alerts.csv"
    db = DatabaseManager(db_path=db_file)

    db.insert_alert(
        timestamp=2000.0,
        rule_name="Crowd Surge / Overcrowding",
        severity="MEDIUM",
        object_class="person",
        track_id=None,
        confidence=1.0,
        message="Capacity exceeded",
    )

    exported_path = db.export_to_csv(output_path=csv_file)
    assert exported_path.is_file()

    with open(exported_path, "r", encoding="utf-8") as f:
        content = f.read()

    assert "Crowd Surge / Overcrowding" in content
    assert "Capacity exceeded" in content
