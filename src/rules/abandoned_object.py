"""Abandoned Object Detection Rule."""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject
from src.utils import euclidean_distance


class AbandonedObjectRule(BaseRule):
    """Detects unattended luggage, backpacks, or bags left stationary without a person nearby."""

    def __init__(
        self,
        stationary_threshold_sec: float = 10.0,
        proximity_radius_px: float = 150.0,
        name: str = "Abandoned Object",
        enabled: bool = True,
        cooldown_sec: float = 15.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the Abandoned Object rule.

        Args:
            stationary_threshold_sec: Seconds an item must remain stationary before triggering.
            proximity_radius_px: Radius in pixels within which an owner (person) must be present.
            name: Rule display name.
            enabled: Active state flag.
            cooldown_sec: Minimum cooldown interval between alerts for the same object.
            target_classes: Luggage classes (defaults to ['backpack', 'handbag', 'suitcase']).
        """
        super().__init__(
            name=name,
            enabled=enabled,
            cooldown_sec=cooldown_sec,
            target_classes=target_classes or ["backpack", "handbag", "suitcase"],
        )
        self.stationary_threshold_sec = float(stationary_threshold_sec)
        self.proximity_radius_px = float(proximity_radius_px)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AbandonedObjectRule":
        """Instantiates rule from configuration dictionary."""
        cfg = config.get("rules", {}).get("abandoned_object", {})
        return cls(
            stationary_threshold_sec=cfg.get("stationary_threshold_sec", 10.0),
            proximity_radius_px=cfg.get("proximity_radius_px", 150.0),
            name=cfg.get("name", "Abandoned Object"),
            enabled=cfg.get("enabled", True),
            cooldown_sec=cfg.get("cooldown_sec", 15.0),
            target_classes=cfg.get("target_classes", ["backpack", "handbag", "suitcase"]),
        )

    def evaluate(
        self,
        tracks: List[TrackedObject],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[RuleAlert]:
        """Evaluates whether any stationary luggage is without nearby persons."""
        if not self.enabled:
            return []

        now = timestamp if timestamp is not None else time.time()
        alerts: List[RuleAlert] = []

        # Find all persons in current frame
        person_centers = [
            t.bottom_center for t in tracks if t.class_name.lower() == "person"
        ]

        luggage_tracks = self.filter_target_tracks(tracks)
        for lug in luggage_tracks:
            # Check if luggage has been stationary long enough
            if lug.stationary_duration < self.stationary_threshold_sec:
                continue

            lug_center = lug.center
            # Calculate distance to closest person
            closest_dist = float("inf")
            if person_centers:
                closest_dist = min(
                    euclidean_distance(lug_center, p_center) for p_center in person_centers
                )

            # If no person is within the designated radius
            if closest_dist > self.proximity_radius_px:
                if not self.is_on_cooldown(track_id=lug.track_id, timestamp=now):
                    self.mark_alerted(track_id=lug.track_id, timestamp=now)
                    alerts.append(
                        RuleAlert(
                            rule_name=self.name,
                            message=(
                                f"Unattended {lug.class_name} (ID {lug.track_id}) stationary for "
                                f"{int(lug.stationary_duration)}s without owner nearby ({int(closest_dist)}px away)"
                            ),
                            severity="HIGH",
                            timestamp=now,
                            track_id=lug.track_id,
                            class_name=lug.class_name,
                            confidence=lug.confidence,
                            box=lug.box,
                            metadata={
                                "stationary_duration": lug.stationary_duration,
                                "closest_person_dist": closest_dist,
                                "radius": self.proximity_radius_px,
                            },
                        )
                    )

        return alerts
