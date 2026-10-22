"""Loitering Detection Rule."""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject
from src.utils import point_in_polygon


class LoiteringRule(BaseRule):
    """Detects when an object or person stays within a designated zone longer than a threshold."""

    def __init__(
        self,
        polygon: Sequence[Sequence[float]],
        dwell_threshold_sec: float = 10.0,
        name: str = "Loitering Detected",
        enabled: bool = True,
        cooldown_sec: float = 10.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the Loitering rule.

        Args:
            polygon: Vertices [[x, y], ...] defining the monitored area.
            dwell_threshold_sec: Seconds an entity may remain before alerting.
            name: Rule display name.
            enabled: Active state flag.
            cooldown_sec: Minimum seconds between repeated loitering alerts for the same ID.
            target_classes: Classes to check (defaults to ['person']).
        """
        super().__init__(
            name=name,
            enabled=enabled,
            cooldown_sec=cooldown_sec,
            target_classes=target_classes or ["person"],
        )
        self.polygon = [list(pt) for pt in polygon]
        self.dwell_threshold_sec = float(dwell_threshold_sec)
        # track_id -> entry_timestamp
        self._zone_entry_times: Dict[int, float] = {}

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "LoiteringRule":
        """Instantiates rule from configuration dictionary."""
        cfg = config.get("rules", {}).get("loitering", {})
        return cls(
            polygon=cfg.get("polygon", []),
            dwell_threshold_sec=cfg.get("dwell_threshold_sec", 10.0),
            name=cfg.get("name", "Loitering Detected"),
            enabled=cfg.get("enabled", True),
            cooldown_sec=cfg.get("cooldown_sec", 10.0),
            target_classes=cfg.get("target_classes", ["person"]),
        )

    def evaluate(
        self,
        tracks: List[TrackedObject],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[RuleAlert]:
        """Evaluates whether any tracked entity has exceeded the loitering duration."""
        if not self.enabled or len(self.polygon) < 3:
            return []

        now = timestamp if timestamp is not None else time.time()
        alerts: List[RuleAlert] = []

        active_track_ids = {t.track_id for t in tracks}
        # Purge deleted tracks
        for tid in list(self._zone_entry_times.keys()):
            if tid not in active_track_ids:
                del self._zone_entry_times[tid]

        relevant_tracks = self.filter_target_tracks(tracks)
        for track in relevant_tracks:
            tid = track.track_id
            point = track.bottom_center
            is_inside = point_in_polygon(point, self.polygon)

            if is_inside:
                if tid not in self._zone_entry_times:
                    self._zone_entry_times[tid] = now

                dwell_duration = now - self._zone_entry_times[tid]
                if dwell_duration >= self.dwell_threshold_sec:
                    if not self.is_on_cooldown(track_id=tid, timestamp=now):
                        self.mark_alerted(track_id=tid, timestamp=now)
                        alerts.append(
                            RuleAlert(
                                rule_name=self.name,
                                message=f"{track.class_name.capitalize()} (ID {tid}) loitering for {int(dwell_duration)}s (limit: {int(self.dwell_threshold_sec)}s)",
                                severity="MEDIUM",
                                timestamp=now,
                                track_id=tid,
                                class_name=track.class_name,
                                confidence=track.confidence,
                                box=track.box,
                                metadata={
                                    "dwell_duration": dwell_duration,
                                    "threshold": self.dwell_threshold_sec,
                                    "point": point,
                                },
                            )
                        )
            else:
                # Reset entry time if entity leaves zone
                self._zone_entry_times.pop(tid, None)

        return alerts
