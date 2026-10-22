"""Restricted Zone Intrusion Rule."""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject
from src.utils import point_in_polygon


class ZoneIntrusionRule(BaseRule):
    """Detects when unauthorized objects or persons enter a restricted polygonal zone."""

    def __init__(
        self,
        polygon: Sequence[Sequence[float]],
        name: str = "Restricted Zone Intrusion",
        enabled: bool = True,
        cooldown_sec: float = 5.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the Zone Intrusion rule.

        Args:
            polygon: List of [x, y] coordinates defining the forbidden perimeter.
            name: Rule display name.
            enabled: Active state flag.
            cooldown_sec: Alert suppression window per track.
            target_classes: Target classes (defaults to ['person']).
        """
        super().__init__(
            name=name,
            enabled=enabled,
            cooldown_sec=cooldown_sec,
            target_classes=target_classes or ["person"],
        )
        self.polygon = [list(pt) for pt in polygon]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ZoneIntrusionRule":
        """Instantiates rule from configuration dictionary."""
        cfg = config.get("rules", {}).get("zone_intrusion", {})
        return cls(
            polygon=cfg.get("polygon", []),
            name=cfg.get("name", "Restricted Zone Intrusion"),
            enabled=cfg.get("enabled", True),
            cooldown_sec=cfg.get("cooldown_sec", 5.0),
            target_classes=cfg.get("target_classes", ["person"]),
        )

    def evaluate(
        self,
        tracks: List[TrackedObject],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[RuleAlert]:
        """Evaluates whether any target object entered the polygon."""
        if not self.enabled or len(self.polygon) < 3:
            return []

        now = timestamp if timestamp is not None else time.time()
        alerts: List[RuleAlert] = []

        relevant_tracks = self.filter_target_tracks(tracks)
        for track in relevant_tracks:
            # Check bottom-center (ground contact foot point)
            point = track.bottom_center
            if point_in_polygon(point, self.polygon):
                if not self.is_on_cooldown(track_id=track.track_id, timestamp=now):
                    self.mark_alerted(track_id=track.track_id, timestamp=now)
                    alerts.append(
                        RuleAlert(
                            rule_name=self.name,
                            message=f"{track.class_name.capitalize()} (ID {track.track_id}) breached restricted zone at ({int(point[0])}, {int(point[1])})",
                            severity="HIGH",
                            timestamp=now,
                            track_id=track.track_id,
                            class_name=track.class_name,
                            confidence=track.confidence,
                            box=track.box,
                            metadata={
                                "point": point,
                                "polygon": self.polygon,
                            },
                        )
                    )

        return alerts
