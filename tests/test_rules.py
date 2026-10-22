"""Unit tests for all suspicious activity detection rules."""

import datetime
from collections import deque
import numpy as np
import pytest

from src.rules.abandoned_object import AbandonedObjectRule
from src.rules.after_hours import AfterHoursRule
from src.rules.crowding import CrowdingRule
from src.rules.loitering import LoiteringRule
from src.rules.zone_intrusion import ZoneIntrusionRule
from src.tracker import TrackedObject


def make_track(
    track_id: int,
    box: list,
    class_name: str = "person",
    class_id: int = 0,
    first_seen: float = 100.0,
    last_seen: float = 100.0,
    last_move: float = 100.0,
) -> TrackedObject:
    """Helper creating a TrackedObject with predefined state."""
    return TrackedObject(
        track_id=track_id,
        box=tuple(box),
        class_id=class_id,
        class_name=class_name,
        confidence=0.90,
        trajectory=deque([( (box[0] + box[2]) / 2, box[3] )], maxlen=30),
        first_seen_time=first_seen,
        last_seen_time=last_seen,
        last_movement_time=last_move,
        anchor_position=((box[0] + box[2]) / 2, (box[1] + box[3]) / 2),
    )


# -----------------------------------------------------------------------------
# 1. Zone Intrusion Tests
# -----------------------------------------------------------------------------
def test_zone_intrusion_trigger():
    """Person whose foot position is inside polygon triggers zone intrusion alert."""
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    rule = ZoneIntrusionRule(polygon=polygon, cooldown_sec=5.0)

    # Foot position: ((150 + 250) / 2, 200) = (200, 200) -> inside
    track_inside = make_track(1, [150, 50, 250, 200], class_name="person")
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    alerts = rule.evaluate([track_inside], dummy_frame, timestamp=10.0)
    assert len(alerts) == 1
    assert alerts[0].rule_name == "Restricted Zone Intrusion"
    assert alerts[0].track_id == 1


def test_zone_intrusion_cooldown():
    """Zone intrusion respects cooldown window for the same track."""
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    rule = ZoneIntrusionRule(polygon=polygon, cooldown_sec=5.0)
    track_inside = make_track(1, [150, 50, 250, 200], class_name="person")
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # First alert triggers
    alerts1 = rule.evaluate([track_inside], dummy_frame, timestamp=10.0)
    assert len(alerts1) == 1

    # Second frame within cooldown window (12s < 10s + 5s) does NOT trigger
    alerts2 = rule.evaluate([track_inside], dummy_frame, timestamp=12.0)
    assert len(alerts2) == 0

    # Third frame after cooldown elapsed (16s > 10s + 5s) triggers again
    alerts3 = rule.evaluate([track_inside], dummy_frame, timestamp=16.0)
    assert len(alerts3) == 1


def test_zone_intrusion_outside():
    """Person outside polygon must not trigger alert."""
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    rule = ZoneIntrusionRule(polygon=polygon)
    # Foot position: (500, 500) -> outside
    track_outside = make_track(2, [450, 400, 550, 500], class_name="person")
    dummy_frame = np.zeros((600, 600, 3), dtype=np.uint8)

    alerts = rule.evaluate([track_outside], dummy_frame, timestamp=10.0)
    assert len(alerts) == 0


# -----------------------------------------------------------------------------
# 2. Loitering Tests
# -----------------------------------------------------------------------------
def test_loitering_triggers_after_threshold():
    """Person loitering in designated area triggers after dwell_threshold_sec."""
    polygon = [[100, 100], [300, 100], [300, 300], [100, 300]]
    rule = LoiteringRule(polygon=polygon, dwell_threshold_sec=10.0, cooldown_sec=10.0)
    track = make_track(1, [150, 50, 250, 200], class_name="person")
    dummy_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Entry at t=10.0s
    alerts0 = rule.evaluate([track], dummy_frame, timestamp=10.0)
    assert len(alerts0) == 0

    # Still inside at t=15.0s (5s elapsed < 10s)
    alerts1 = rule.evaluate([track], dummy_frame, timestamp=15.0)
    assert len(alerts1) == 0

    # Inside at t=21.0s (11s elapsed >= 10s) -> TRIGGER
    alerts2 = rule.evaluate([track], dummy_frame, timestamp=21.0)
    assert len(alerts2) == 1
    assert "loitering" in alerts2[0].message.lower()


# -----------------------------------------------------------------------------
# 3. Abandoned Object Tests
# -----------------------------------------------------------------------------
def test_abandoned_object_trigger():
    """Stationary bag left without nearby person triggers abandoned object alert."""
    rule = AbandonedObjectRule(
        stationary_threshold_sec=10.0,
        proximity_radius_px=100.0,
    )
    dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)

    # Bag at (100, 100) stationary for 12 seconds
    bag_track = make_track(
        track_id=1,
        box=[80, 80, 120, 120],
        class_name="backpack",
        first_seen=10.0,
        last_seen=22.0,
        last_move=10.0,  # stationary_duration = 12s >= 10s
    )

    # Person located far away at (400, 400), dist ~424px > 100px
    person_track = make_track(
        track_id=2,
        box=[380, 350, 420, 450],
        class_name="person",
        first_seen=10.0,
        last_seen=22.0,
        last_move=20.0,
    )

    alerts = rule.evaluate([bag_track, person_track], dummy_frame, timestamp=22.0)
    assert len(alerts) == 1
    assert alerts[0].rule_name == "Abandoned Object"
    assert alerts[0].track_id == 1


def test_abandoned_object_attended():
    """Stationary bag with person standing nearby (< proximity_radius) must NOT trigger."""
    rule = AbandonedObjectRule(
        stationary_threshold_sec=10.0,
        proximity_radius_px=100.0,
    )
    dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)

    bag_track = make_track(
        track_id=1,
        box=[100, 100, 140, 140],
        class_name="suitcase",
        first_seen=10.0,
        last_seen=25.0,
        last_move=10.0,
    )

    # Person standing right beside the bag (at 130, 130), dist ~20px < 100px
    person_track = make_track(
        track_id=2,
        box=[110, 80, 150, 180],
        class_name="person",
        first_seen=10.0,
        last_seen=25.0,
    )

    alerts = rule.evaluate([bag_track, person_track], dummy_frame, timestamp=25.0)
    assert len(alerts) == 0


# -----------------------------------------------------------------------------
# 4. Crowding Tests
# -----------------------------------------------------------------------------
def test_crowding_trigger():
    """Crowd exceeding threshold triggers alert."""
    rule = CrowdingRule(max_person_count=3, cooldown_sec=10.0)
    dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)

    # 4 persons present
    persons = [
        make_track(i, [i * 30, 100, i * 30 + 20, 150], class_name="person")
        for i in range(1, 5)
    ]

    alerts = rule.evaluate(persons, dummy_frame, timestamp=10.0)
    assert len(alerts) == 1
    assert "Overcrowding detected" in alerts[0].message
    assert alerts[0].metadata["count"] == 4


def test_crowding_under_limit():
    """Crowd below or equal to threshold does NOT trigger alert."""
    rule = CrowdingRule(max_person_count=3)
    dummy_frame = np.zeros((500, 500, 3), dtype=np.uint8)

    # 2 persons present
    persons = [
        make_track(1, [10, 100, 30, 150], class_name="person"),
        make_track(2, [50, 100, 70, 150], class_name="person"),
    ]

    alerts = rule.evaluate(persons, dummy_frame, timestamp=10.0)
    assert len(alerts) == 0


# -----------------------------------------------------------------------------
# 5. After-Hours Tests
# -----------------------------------------------------------------------------
def test_after_hours_evaluation():
    """Checks midnight crossing after-hours detection."""
    rule = AfterHoursRule(start_time="22:00", end_time="06:00")

    # 23:30 is after hours
    t1 = datetime.datetime(2026, 10, 21, 23, 30)
    assert rule.is_current_time_after_hours(t1) is True

    # 02:15 is after hours
    t2 = datetime.datetime(2026, 10, 22, 2, 15)
    assert rule.is_current_time_after_hours(t2) is True

    # 14:00 is daytime (authorized)
    t3 = datetime.datetime(2026, 10, 22, 14, 0)
    assert rule.is_current_time_after_hours(t3) is False

    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    person = make_track(1, [20, 20, 60, 100], class_name="person")

    # Evaluation during night
    alerts_night = rule.evaluate([person], dummy_frame, override_now=t1)
    assert len(alerts_night) == 1
    assert alerts_night[0].severity == "CRITICAL"

    # Evaluation during day
    rule.reset_cooldowns()
    alerts_day = rule.evaluate([person], dummy_frame, override_now=t3)
    assert len(alerts_day) == 0
