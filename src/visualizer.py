"""Drawing and visualization module: bounding boxes, track IDs, trails, zones, and alert banners."""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject

# Distinct vibrant color palette for visual clarity
COLOR_PALETTE = [
    (0, 255, 128),    # Spring Green
    (255, 140, 0),    # Deep Sky Orange
    (238, 130, 238),  # Violet
    (0, 215, 255),    # Gold
    (255, 105, 180),  # Hot Pink
    (50, 205, 50),    # Lime Green
    (30, 144, 255),   # Dodger Blue
    (255, 20, 147),   # Deep Pink
    (0, 255, 255),    # Yellow
    (147, 112, 219),  # Medium Purple
]


class SurveillanceVisualizer:
    """Renders overlays, bounding boxes, trajectory trails, monitored zones, and alerts on frames."""

    def __init__(
        self,
        show_boxes: bool = True,
        show_labels: bool = True,
        show_trajectories: bool = True,
        show_zones: bool = True,
        show_fps: bool = True,
        show_active_tracks: bool = True,
    ) -> None:
        """Initializes the visualizer."""
        self.show_boxes = show_boxes
        self.show_labels = show_labels
        self.show_trajectories = show_trajectories
        self.show_zones = show_zones
        self.show_fps = show_fps
        self.show_active_tracks = show_active_tracks

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SurveillanceVisualizer":
        """Instantiates visualizer from configuration dictionary."""
        vis_cfg = config.get("visualization", {})
        return cls(
            show_boxes=vis_cfg.get("show_boxes", True),
            show_labels=vis_cfg.get("show_labels", True),
            show_trajectories=vis_cfg.get("show_trajectories", True),
            show_zones=vis_cfg.get("show_zones", True),
            show_fps=vis_cfg.get("show_fps", True),
            show_active_tracks=vis_cfg.get("show_active_tracks", True),
        )

    def _get_color_for_id(self, track_id: int) -> Tuple[int, int, int]:
        """Maps an integer ID deterministically to a distinct BGR color."""
        return COLOR_PALETTE[track_id % len(COLOR_PALETTE)]

    def draw_zones(
        self,
        frame: np.ndarray,
        rules: Sequence[BaseRule],
    ) -> np.ndarray:
        """Draws configured polygon perimeters with translucent overlay."""
        if not self.show_zones:
            return frame

        overlay = frame.copy()
        alpha = 0.25

        for rule in rules:
            if not getattr(rule, "enabled", True):
                continue
            polygon = getattr(rule, "polygon", None)
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon, np.int32).reshape((-1, 1, 2))
                # Fill semi-transparent zone
                cv2.fillPoly(overlay, [pts], (0, 0, 200))
                # Draw sharp border
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

                # Add label near first vertex
                p0 = tuple(map(int, polygon[0]))
                cv2.putText(
                    frame,
                    f"ZONE: {rule.name}",
                    (p0[0] + 5, p0[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        return frame

    def draw_trajectories(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
    ) -> np.ndarray:
        """Draws smooth fading motion trajectory trails for tracked entities."""
        if not self.show_trajectories:
            return frame

        for track in tracks:
            color = self._get_color_for_id(track.track_id)
            pts = list(track.trajectory)
            num_pts = len(pts)
            if num_pts < 2:
                continue

            for i in range(1, num_pts):
                # Fade thickness towards older points
                thickness = int(np.sqrt(16 * (float(i) / num_pts))) + 1
                pt1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
                pt2 = (int(pts[i][0]), int(pts[i][1]))
                cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)

        return frame

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
    ) -> np.ndarray:
        """Draws bounding boxes, IDs, labels, and dwell time tags."""
        if not self.show_boxes:
            return frame

        for track in tracks:
            color = self._get_color_for_id(track.track_id)
            x1, y1, x2, y2 = map(int, track.box)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            if self.show_labels:
                dwell_sec = int(track.dwell_time)
                label = f"#{track.track_id} {track.class_name} {track.confidence:.2f} ({dwell_sec}s)"

                # Label background rectangle
                (text_w, text_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                bg_y1 = max(0, y1 - text_h - 6)
                bg_y2 = y1
                cv2.rectangle(
                    frame,
                    (x1, bg_y1),
                    (x1 + text_w + 6, bg_y2),
                    color,
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return frame

    def draw_banner(
        self,
        frame: np.ndarray,
        banner_info: Optional[Tuple[str, float]],
        current_time: float,
    ) -> np.ndarray:
        """Draws top red warning banner if an alert is currently active."""
        if banner_info is None:
            return frame

        message, expiry = banner_info
        if current_time >= expiry:
            return frame

        h, w = frame.shape[:2]
        banner_h = 44

        # Red banner rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 220), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Warning icon and text
        full_text = f"WARNING / SECURITY ALERT: {message}"
        (text_w, text_h), _ = cv2.getTextSize(full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        tx = max(10, (w - text_w) // 2)
        ty = (banner_h + text_h) // 2 - 2

        cv2.putText(
            frame,
            full_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Pulsing bottom line
        cv2.line(frame, (0, banner_h), (w, banner_h), (0, 255, 255), 2)
        return frame

    def draw_hud(
        self,
        frame: np.ndarray,
        fps: float,
        inference_ms: float,
        active_track_count: int,
        alert_count: int,
    ) -> np.ndarray:
        """Draws top-left status heads-up display (HUD)."""
        if not self.show_fps and not self.show_active_tracks:
            return frame

        lines = []
        if self.show_fps:
            lines.append(f"FPS: {fps:.1f} | Latency: {inference_ms:.1f}ms")
        if self.show_active_tracks:
            lines.append(f"Active Tracks: {active_track_count} | Total Alerts: {alert_count}")

        # Semi-transparent HUD background
        hud_w = 340
        hud_h = 24 + len(lines) * 22
        x0, y0 = 10, 50

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + hud_w, y0 + hud_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (x0, y0), (x0 + hud_w, y0 + hud_h), (60, 60, 60), 1)

        for idx, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (x0 + 8, y0 + 20 + idx * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return frame

    def render(
        self,
        frame: np.ndarray,
        tracks: Sequence[TrackedObject],
        rules: Sequence[BaseRule],
        banner_info: Optional[Tuple[str, float]],
        current_time: float,
        fps: float = 0.0,
        inference_ms: float = 0.0,
        total_alerts: int = 0,
    ) -> np.ndarray:
        """Full rendering pass returning annotated BGR frame."""
        annotated = frame.copy()

        annotated = self.draw_zones(annotated, rules)
        annotated = self.draw_trajectories(annotated, tracks)
        annotated = self.draw_tracks(annotated, tracks)
        annotated = self.draw_banner(annotated, banner_info, current_time)
        annotated = self.draw_hud(
            annotated,
            fps=fps,
            inference_ms=inference_ms,
            active_track_count=len(tracks),
            alert_count=total_alerts,
        )

        return annotated
