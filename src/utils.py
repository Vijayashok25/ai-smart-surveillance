"""Core utility functions: NMS, IoU, geometry, FPS calculation, configuration, and logging."""

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import yaml


def setup_logger(
    name: str = "SurveillanceSystem",
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configures and returns a structured logger.

    Args:
        name: Name of the logger.
        level: Logging level (e.g. 'DEBUG', 'INFO', 'WARNING', 'ERROR').
        log_file: Optional path to append log file output.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)

    # Avoid duplicate handlers if already configured
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(numeric_level)
        logger.addHandler(console_handler)

        if log_file:
            file_path = Path(log_file)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(file_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            file_handler.setLevel(numeric_level)
            logger.addHandler(file_handler)

    return logger


def load_config(config_path: Union[str, Path] = "config.yaml") -> Dict[str, Any]:
    """Loads and validates the YAML configuration file.

    Args:
        config_path: Path to configuration YAML file.

    Returns:
        Dictionary of configuration parameters.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If essential configuration sections are missing.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML content in {path}; expected a mapping dictionary.")

    # Validate mandatory top-level sections
    required_sections = ["video", "detection", "tracking", "rules", "alerts", "database"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing mandatory configuration section '{section}' in {path}.")

    return config


def compute_iou(
    box1: Sequence[float],
    box2: Sequence[float],
) -> float:
    """Computes the Intersection over Union (IoU) of two bounding boxes.

    Boxes are in [x1, y1, x2, y2] format.

    Args:
        box1: [x1, y1, x2, y2] coordinates.
        box2: [x1, y1, x2, y2] coordinates.

    Returns:
        IoU value in the range [0.0, 1.0].
    """
    x1 = max(float(box1[0]), float(box2[0]))
    y1 = max(float(box1[1]), float(box2[1]))
    x2 = min(float(box1[2]), float(box2[2]))
    y2 = min(float(box1[3]), float(box2[3]))

    intersection_w = max(0.0, x2 - x1)
    intersection_h = max(0.0, y2 - y1)
    intersection_area = intersection_w * intersection_h

    area1 = max(0.0, float(box1[2]) - float(box1[0])) * max(0.0, float(box1[3]) - float(box1[1]))
    area2 = max(0.0, float(box2[2]) - float(box2[0])) * max(0.0, float(box2[3]) - float(box2[1]))

    union_area = area1 + area2 - intersection_area
    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def compute_iou_matrix(
    boxes1: np.ndarray,
    boxes2: np.ndarray,
) -> np.ndarray:
    """Vectorized calculation of IoU between two sets of bounding boxes.

    Args:
        boxes1: Array of shape (N, 4) in [x1, y1, x2, y2].
        boxes2: Array of shape (M, 4) in [x1, y1, x2, y2].

    Returns:
        Array of shape (N, M) containing pairwise IoU scores.
    """
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    boxes1 = np.asarray(boxes1, dtype=np.float32)
    boxes2 = np.asarray(boxes2, dtype=np.float32)

    # Intersections
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # Areas
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = area1[:, None] + area2[None, :] - inter_area
    union_area = np.maximum(union_area, 1e-8)

    return inter_area / union_area


def numpy_nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float = 0.45,
) -> List[int]:
    """Pure NumPy Non-Maximum Suppression (NMS) for bounding box filtering.

    Args:
        boxes: List or 2D array of shape (N, 4) with [x1, y1, x2, y2].
        scores: 1D array-like of shape (N,) containing confidence scores.
        iou_threshold: Threshold above which overlapping boxes are suppressed.

    Returns:
        List of integer indices of kept boxes, sorted in descending order of score.
    """
    if len(boxes) == 0 or len(scores) == 0:
        return []

    boxes_arr = np.asarray(boxes, dtype=np.float32)
    scores_arr = np.asarray(scores, dtype=np.float32)

    if boxes_arr.ndim != 2 or boxes_arr.shape[1] != 4:
        raise ValueError(f"boxes must have shape (N, 4), got {boxes_arr.shape}")
    if scores_arr.ndim != 1 or len(scores_arr) != len(boxes_arr):
        raise ValueError(f"scores length ({len(scores_arr)}) must match boxes length ({len(boxes_arr)})")

    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = boxes_arr[:, 2]
    y2 = boxes_arr[:, 3]

    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores_arr.argsort()[::-1]

    keep: List[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))

        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        union = areas[i] + areas[order[1:]] - inter
        iou = np.where(union > 0, inter / union, 0.0)

        # Indices of boxes with IoU <= threshold
        remaining_indices = np.where(iou <= iou_threshold)[0]
        order = order[remaining_indices + 1]

    return keep


def bbox_center(box: Sequence[float]) -> Tuple[float, float]:
    """Calculates the geometric center (cx, cy) of a bounding box [x1, y1, x2, y2]."""
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def bbox_bottom_center(box: Sequence[float]) -> Tuple[float, float]:
    """Calculates the bottom center (cx, y2) of a bounding box [x1, y1, x2, y2].

    This represents the ground contact / foot position of an object or pedestrian.
    """
    return ((float(box[0]) + float(box[2])) / 2.0, float(box[3]))


def euclidean_distance(
    p1: Sequence[float],
    p2: Sequence[float],
) -> float:
    """Calculates Euclidean distance between two 2D points."""
    return float(np.hypot(float(p1[0]) - float(p2[0]), float(p1[1]) - float(p2[1])))


def point_in_polygon(
    point: Sequence[float],
    polygon: Sequence[Sequence[float]],
) -> bool:
    """Determines if a 2D point [x, y] lies inside a polygon using ray casting.

    Args:
        point: Coordinates [x, y].
        polygon: Ordered list of vertices [[x0, y0], [x1, y1], ...].

    Returns:
        True if the point is strictly or tangentially inside the polygon, False otherwise.
    """
    if len(polygon) < 3:
        return False

    px, py = float(point[0]), float(point[1])
    num_vertices = len(polygon)
    inside = False

    p1x, p1y = float(polygon[0][0]), float(polygon[0][1])
    for i in range(num_vertices + 1):
        p2x, p2y = float(polygon[i % num_vertices][0]), float(polygon[i % num_vertices][1])
        if py > min(p1y, p2y):
            if py <= max(p1y, p2y):
                if px <= max(p1x, p2x):
                    if p1y != p2y:
                        x_inters = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    else:
                        x_inters = p1x
                    if p1x == p2x or px <= x_inters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside


class FPSCounter:
    """Thread-safe, sliding-window FPS and latency calculator."""

    def __init__(self, window_size: int = 30) -> None:
        """Initializes the FPS counter.

        Args:
            window_size: Number of recent frame intervals to smooth over.
        """
        self.window_size = window_size
        self._timestamps: deque = deque(maxlen=window_size)
        self._inference_latencies: deque = deque(maxlen=window_size)
        self._last_time = time.perf_counter()

    def update(self, inference_ms: Optional[float] = None) -> float:
        """Records a new frame tick and optional model inference latency.

        Args:
            inference_ms: Latency of the model inference in milliseconds.

        Returns:
            Current smoothed FPS.
        """
        now = time.perf_counter()
        delta = now - self._last_time
        self._last_time = now

        if delta > 0:
            self._timestamps.append(delta)

        if inference_ms is not None:
            self._inference_latencies.append(inference_ms)

        return self.fps

    @property
    def fps(self) -> float:
        """Returns the current rolling average frames per second."""
        if not self._timestamps:
            return 0.0
        avg_delta = sum(self._timestamps) / len(self._timestamps)
        return 1.0 / avg_delta if avg_delta > 0 else 0.0

    @property
    def avg_inference_ms(self) -> float:
        """Returns rolling average inference latency in milliseconds."""
        if not self._inference_latencies:
            return 0.0
        return float(sum(self._inference_latencies) / len(self._inference_latencies))
