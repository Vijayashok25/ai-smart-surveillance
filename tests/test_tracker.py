"""Unit tests for CentroidTracker and TrackedObject state management."""

import pytest

from src.detector import Detection
from src.tracker import CentroidTracker, TrackedObject


def test_centroid_tracker_lifecycle():
    """Tracks should be initialized, updated with trajectory, and purged after disappearing."""
    tracker = CentroidTracker(
        max_disappeared_frames=2,
        max_distance_px=50.0,
        max_trail_points=10,
    )

    # Frame 1: Detection at (100, 100)
    det1 = Detection(box=(90.0, 90.0, 110.0, 110.0), confidence=0.9, class_id=0, class_name="person")
    tracks = tracker.update([det1], timestamp=1.0)

    assert len(tracks) == 1
    track_id = tracks[0].track_id
    assert track_id == 1
    assert tracks[0].dwell_time == 0.0

    # Frame 2: Object moves slightly to (105, 105)
    det2 = Detection(box=(95.0, 95.0, 115.0, 115.0), confidence=0.92, class_id=0, class_name="person")
    tracks = tracker.update([det2], timestamp=2.0)

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].dwell_time == 1.0
    assert len(tracks[0].trajectory) >= 2

    # Frame 3: Disappeared (no detections)
    tracks = tracker.update([], timestamp=3.0)
    assert len(tracks) == 1
    assert tracks[0].disappeared_count == 1

    # Frame 4: Disappeared frame 2
    tracks = tracker.update([], timestamp=4.0)
    assert len(tracks) == 1
    assert tracks[0].disappeared_count == 2

    # Frame 5: Disappeared frame 3 (exceeds max_disappeared_frames=2) -> removed
    tracks = tracker.update([], timestamp=5.0)
    assert len(tracks) == 0
