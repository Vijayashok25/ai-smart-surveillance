"""Unit tests for utility functions: IoU, NMS, geometry, and FPS counter."""

import pytest
import numpy as np

from src.utils import (
    FPSCounter,
    bbox_bottom_center,
    bbox_center,
    compute_iou,
    compute_iou_matrix,
    euclidean_distance,
    numpy_nms,
    point_in_polygon,
)


def test_compute_iou_identical_boxes():
    """IoU between identical boxes must be exactly 1.0."""
    box = [10.0, 10.0, 50.0, 50.0]
    assert pytest.approx(compute_iou(box, box), 1e-5) == 1.0


def test_compute_iou_disjoint_boxes():
    """IoU between completely non-overlapping boxes must be 0.0."""
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    assert compute_iou(box1, box2) == 0.0


def test_compute_iou_partial_overlap():
    """IoU for 50% intersecting boxes."""
    box1 = [0.0, 0.0, 10.0, 10.0]  # area = 100
    box2 = [5.0, 0.0, 15.0, 10.0]  # area = 100, inter = 5x10 = 50, union = 150
    expected = 50.0 / 150.0
    assert pytest.approx(compute_iou(box1, box2), 1e-4) == expected


def test_compute_iou_matrix():
    """Vectorized pairwise IoU calculation."""
    boxes1 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
    boxes2 = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    iou_mat = compute_iou_matrix(boxes1, boxes2)

    assert iou_mat.shape == (2, 2)
    assert pytest.approx(iou_mat[0, 0], 1e-4) == 1.0
    assert iou_mat[0, 1] == 0.0
    assert iou_mat[1, 1] == 0.0


def test_numpy_nms_suppression():
    """NMS suppresses highly overlapping lower-confidence duplicate boxes."""
    boxes = [
        [10.0, 10.0, 50.0, 50.0],  # Box 0: highest score
        [12.0, 11.0, 51.0, 49.0],  # Box 1: high overlap with 0, lower score
        [100.0, 100.0, 150.0, 150.0],  # Box 2: distinct object
    ]
    scores = [0.95, 0.85, 0.90]

    kept = numpy_nms(boxes, scores, iou_threshold=0.45)
    assert 0 in kept
    assert 1 not in kept  # suppressed
    assert 2 in kept


def test_numpy_nms_empty():
    """NMS safely handles empty input lists."""
    assert numpy_nms([], []) == []


def test_point_in_polygon():
    """Ray-casting test for point inside, outside, and boundary."""
    polygon = [[0, 0], [100, 0], [100, 100], [0, 100]]

    # Center is inside
    assert point_in_polygon([50, 50], polygon) is True
    # Distant point is outside
    assert point_in_polygon([150, 50], polygon) is False
    assert point_in_polygon([-10, 50], polygon) is False

    # Triangle polygon
    triangle = [[0, 0], [100, 0], [50, 100]]
    assert point_in_polygon([50, 30], triangle) is True
    assert point_in_polygon([10, 80], triangle) is False


def test_geometry_helpers():
    """Center, bottom center, and Euclidean distance tests."""
    box = [0.0, 0.0, 100.0, 200.0]
    assert bbox_center(box) == (50.0, 100.0)
    assert bbox_bottom_center(box) == (50.0, 200.0)

    p1 = (0.0, 0.0)
    p2 = (3.0, 4.0)
    assert pytest.approx(euclidean_distance(p1, p2), 1e-4) == 5.0


def test_fps_counter():
    """FPS counter rolling metrics calculation."""
    counter = FPSCounter(window_size=5)
    assert counter.fps == 0.0

    # Simulate ticks with dummy latencies
    counter.update(inference_ms=15.0)
    counter.update(inference_ms=25.0)

    assert counter.avg_inference_ms == 20.0
