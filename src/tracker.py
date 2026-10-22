"""Object tracking module with trajectory history, dwell time, and stationary analysis."""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.detector import Detection
from src.utils import compute_iou, euclidean_distance, setup_logger

try:
    from ultralytics.trackers.byte_tracker import BYTETracker
    HAS_BYTETRACK = True
except (ImportError, Exception):
    BYTETracker = None
    HAS_BYTETRACK = False


@dataclass
class TrackedObject:
    """Represents an active tracked entity across frames."""

    track_id: int
    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    class_id: int
    class_name: str
    confidence: float
    trajectory: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=30))
    first_seen_time: float = field(default_factory=time.time)
    last_seen_time: float = field(default_factory=time.time)
    last_movement_time: float = field(default_factory=time.time)
    disappeared_count: int = 0
    anchor_position: Tuple[float, float] = (0.0, 0.0)

    def __post_init__(self) -> None:
        """Initializes anchor point and initial trajectory."""
        center = ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)
        bottom_center = ((self.box[0] + self.box[2]) / 2.0, self.box[3])
        if len(self.trajectory) == 0:
            self.trajectory.append(bottom_center)
        if self.anchor_position == (0.0, 0.0):
            self.anchor_position = center

    @property
    def dwell_time(self) -> float:
        """Returns total duration (seconds) this object has been tracked."""
        return max(0.0, self.last_seen_time - self.first_seen_time)

    @property
    def stationary_duration(self) -> float:
        """Returns time (seconds) this object has been stationary around its anchor."""
        return max(0.0, self.last_seen_time - self.last_movement_time)

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates (cx, cy) bounding box center."""
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates (cx, y2) bottom center."""
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])

    def update_position(
        self,
        new_box: Tuple[float, float, float, float],
        confidence: float,
        timestamp: float,
        movement_threshold_px: float = 15.0,
    ) -> None:
        """Updates track coordinates, trajectory, and stationary state."""
        self.box = new_box
        self.confidence = confidence
        self.last_seen_time = timestamp
        self.disappeared_count = 0

        current_center = self.center
        current_bottom = self.bottom_center
        self.trajectory.append(current_bottom)

        # Check if object moved beyond movement threshold from anchor
        dist_from_anchor = euclidean_distance(current_center, self.anchor_position)
        if dist_from_anchor > movement_threshold_px:
            self.anchor_position = current_center
            self.last_movement_time = timestamp


class CentroidTracker:
    """Distance- and IoU-based multi-object tracker."""

    def __init__(
        self,
        max_disappeared_frames: int = 30,
        max_distance_px: float = 80.0,
        max_trail_points: int = 30,
        movement_threshold_px: float = 15.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the CentroidTracker."""
        self.max_disappeared_frames = max_disappeared_frames
        self.max_distance_px = max_distance_px
        self.max_trail_points = max_trail_points
        self.movement_threshold_px = movement_threshold_px
        self.logger = logger or setup_logger("CentroidTracker")

        self.next_track_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def update(
        self,
        detections: List[Detection],
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """Associates incoming detections with existing tracks."""
        if timestamp is None:
            timestamp = time.time()

        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id].disappeared_count += 1
                if self.tracks[track_id].disappeared_count > self.max_disappeared_frames:
                    del self.tracks[track_id]
            return list(self.tracks.values())

        if len(self.tracks) == 0:
            for det in detections:
                self._register_new_track(det, timestamp)
            return list(self.tracks.values())

        # Match existing tracks with detections
        track_ids = list(self.tracks.keys())
        track_centers = [self.tracks[tid].center for tid in track_ids]
        det_centers = [d.center for d in detections]

        D = np.zeros((len(track_ids), len(detections)), dtype=np.float32)
        for i, t_center in enumerate(track_centers):
            for j, d_center in enumerate(det_centers):
                # Penalize class mismatch
                class_penalty = 0.0 if self.tracks[track_ids[i]].class_id == detections[j].class_id else 1000.0
                dist = euclidean_distance(t_center, d_center)
                iou = compute_iou(self.tracks[track_ids[i]].box, detections[j].box)
                D[i, j] = dist + class_penalty - (iou * 50.0)

        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        assigned_rows = set()
        assigned_cols = set()

        for row, col in zip(rows, cols):
            if row in assigned_rows or col in assigned_cols:
                continue

            track_id = track_ids[row]
            det = detections[col]

            # If distance exceeds threshold, do not associate
            if euclidean_distance(track_centers[row], det_centers[col]) > self.max_distance_px:
                continue

            self.tracks[track_id].update_position(
                new_box=det.box,
                confidence=det.confidence,
                timestamp=timestamp,
                movement_threshold_px=self.movement_threshold_px,
            )
            det.track_id = track_id
            assigned_rows.add(row)
            assigned_cols.add(col)

        # Unassigned existing tracks
        for row in range(len(track_ids)):
            if row not in assigned_rows:
                track_id = track_ids[row]
                self.tracks[track_id].disappeared_count += 1
                if self.tracks[track_id].disappeared_count > self.max_disappeared_frames:
                    del self.tracks[track_id]

        # Unassigned detections -> new tracks
        for col in range(len(detections)):
            if col not in assigned_cols:
                self._register_new_track(detections[col], timestamp)

        return list(self.tracks.values())

    def _register_new_track(self, detection: Detection, timestamp: float) -> None:
        """Creates a new TrackedObject instance."""
        track = TrackedObject(
            track_id=self.next_track_id,
            box=detection.box,
            class_id=detection.class_id,
            class_name=detection.class_name,
            confidence=detection.confidence,
            trajectory=deque(maxlen=self.max_trail_points),
            first_seen_time=timestamp,
            last_seen_time=timestamp,
            last_movement_time=timestamp,
            anchor_position=detection.center,
        )
        detection.track_id = self.next_track_id
        self.tracks[self.next_track_id] = track
        self.next_track_id += 1

    def reset(self) -> None:
        """Clears all tracked objects and resets ID counter."""
        self.tracks.clear()
        self.next_track_id = 1


class ByteTrackTracker:
    """Wrapper around Ultralytics ByteTrack with persistent trajectory and dwell-time states."""

    def __init__(
        self,
        max_trail_points: int = 30,
        movement_threshold_px: float = 15.0,
        frame_rate: int = 30,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the ByteTrack tracker."""
        self.logger = logger or setup_logger("ByteTracker")
        self.max_trail_points = max_trail_points
        self.movement_threshold_px = movement_threshold_px
        self.tracks: Dict[int, TrackedObject] = {}

        if not HAS_BYTETRACK:
            self.logger.warning("BYTETracker unavailable; falling back to CentroidTracker.")
            self._fallback = CentroidTracker(
                max_trail_points=max_trail_points,
                movement_threshold_px=movement_threshold_px,
                logger=self.logger,
            )
            self._using_fallback = True
        else:
            args = SimpleNamespace(
                track_high_thresh=0.45,
                track_low_thresh=0.1,
                new_track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            )
            self.tracker = BYTETracker(args, frame_rate=frame_rate)
            self._using_fallback = False

    def update(
        self,
        detections: List[Detection],
        timestamp: Optional[float] = None,
    ) -> List[TrackedObject]:
        """Updates ByteTrack with detections and syncs trajectory history."""
        if timestamp is None:
            timestamp = time.time()

        if self._using_fallback:
            return self._fallback.update(detections, timestamp)

        if len(detections) == 0:
            return list(self.tracks.values())

        # Build tensor/array for ByteTrack
        # ByteTrack update expects an object with .xywh, .xyxy, .conf, .cls, and sliceable via __getitem__
        xyxy_list = [d.box for d in detections]
        xywh_list = [
            [
                (b[0] + b[2]) / 2.0,
                (b[1] + b[3]) / 2.0,
                b[2] - b[0],
                b[3] - b[1],
            ]
            for b in xyxy_list
        ]
        conf_list = [d.confidence for d in detections]
        cls_list = [d.class_id for d in detections]

        class DetectionResults:
            def __init__(self, xywh_arr, xyxy_arr, confs_arr, clss_arr):
                import torch
                self.xywh = torch.as_tensor(xywh_arr, dtype=torch.float32).reshape(-1, 4)
                self.xyxy = torch.as_tensor(xyxy_arr, dtype=torch.float32).reshape(-1, 4)
                self.conf = torch.as_tensor(confs_arr, dtype=torch.float32).reshape(-1)
                self.cls = torch.as_tensor(clss_arr, dtype=torch.float32).reshape(-1)

            def __len__(self) -> int:
                return len(self.conf)

            def __getitem__(self, idx: Any) -> "DetectionResults":
                return DetectionResults(
                    self.xywh[idx].cpu().numpy(),
                    self.xyxy[idx].cpu().numpy(),
                    self.conf[idx].cpu().numpy(),
                    self.cls[idx].cpu().numpy(),
                )

        try:
            results_container = DetectionResults(xywh_list, xyxy_list, conf_list, cls_list)
            online_targets = self.tracker.update(results_container)

            active_ids = set()
            for row in online_targets:
                tlbr = row[:4]  # [x1, y1, x2, y2]
                tid = int(row[4])
                score = float(row[5])
                cls_id = int(row[6])
                active_ids.add(tid)

                box_tuple = (float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3]))

                if tid in self.tracks:
                    self.tracks[tid].update_position(
                        new_box=box_tuple,
                        confidence=score,
                        timestamp=timestamp,
                        movement_threshold_px=self.movement_threshold_px,
                    )
                else:
                    cname = detections[0].class_name if detections else "object"
                    for d in detections:
                        if d.class_id == cls_id:
                            cname = d.class_name
                            break

                    self.tracks[tid] = TrackedObject(
                        track_id=tid,
                        box=box_tuple,
                        class_id=cls_id,
                        class_name=cname,
                        confidence=score,
                        trajectory=deque(maxlen=self.max_trail_points),
                        first_seen_time=timestamp,
                        last_seen_time=timestamp,
                        last_movement_time=timestamp,
                        anchor_position=((box_tuple[0] + box_tuple[2]) / 2.0, (box_tuple[1] + box_tuple[3]) / 2.0),
                    )

                # Associate back to Detection
                for det in detections:
                    if compute_iou(det.box, box_tuple) > 0.6:
                        det.track_id = tid

            # Remove inactive tracks
            for tid in list(self.tracks.keys()):
                if tid not in active_ids:
                    self.tracks[tid].disappeared_count += 1
                    if self.tracks[tid].disappeared_count > 30:
                        del self.tracks[tid]

        except Exception as e:
            self.logger.warning(f"ByteTrack update failed ({e}); falling back to CentroidTracker.")
            self._using_fallback = True
            self._fallback = CentroidTracker(
                max_trail_points=self.max_trail_points,
                movement_threshold_px=self.movement_threshold_px,
                logger=self.logger,
            )
            return self._fallback.update(detections, timestamp)

        return list(self.tracks.values())

    def reset(self) -> None:
        """Clears all tracking states."""
        self.tracks.clear()
        if not self._using_fallback and hasattr(self, "tracker"):
            args = SimpleNamespace(
                track_high_thresh=0.45,
                track_low_thresh=0.1,
                new_track_thresh=0.5,
                track_buffer=30,
                match_thresh=0.8,
                fuse_score=True,
            )
            self.tracker = BYTETracker(args)


def create_tracker(
    tracker_type: str = "bytetrack",
    max_trail_points: int = 30,
    max_disappeared_frames: int = 30,
    max_distance_px: float = 80.0,
    movement_threshold_px: float = 15.0,
    logger: Optional[logging.Logger] = None,
) -> Any:
    """Factory creating configured tracker instance."""
    t_type = tracker_type.strip().lower()
    if t_type == "bytetrack":
        return ByteTrackTracker(
            max_trail_points=max_trail_points,
            movement_threshold_px=movement_threshold_px,
            logger=logger,
        )
    return CentroidTracker(
        max_disappeared_frames=max_disappeared_frames,
        max_distance_px=max_distance_px,
        max_trail_points=max_trail_points,
        movement_threshold_px=movement_threshold_px,
        logger=logger,
    )
