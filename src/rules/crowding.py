"""Crowd Surge / Overcrowding Detection Rule."""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject


class CrowdingRule(BaseRule):
    """Detects crowd gathering or capacity violations exceeding a configured headcount threshold."""

    def __init__(
        self,
        max_person_count: int = 5,
        name: str = "Crowd Surge / Overcrowding",
        enabled: bool = True,
        cooldown_sec: float = 10.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the Crowding rule.

        Args:
            max_person_count: Maximum permitted concurrent persons before alerting.
            name: Rule display name.
            enabled: Active state flag.
            cooldown_sec: Alert suppression window.
            target_classes: Target classes (defaults to ['person']).
        """
        super().__init__(
            name=name,
            enabled=enabled,
            cooldown_sec=cooldown_sec,
            target_classes=target_classes or ["person"],
        )
        self.max_person_count = int(max_person_count)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CrowdingRule":
        """Instantiates rule from configuration dictionary."""
        cfg = config.get("rules", {}).get("crowding", {})
        return cls(
            max_person_count=cfg.get("max_person_count", 5),
            name=cfg.get("name", "Crowd Surge / Overcrowding"),
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
        """Evaluates whether current person count exceeds the safety threshold."""
        if not self.enabled:
            return []

        now = timestamp if timestamp is not None else time.time()
        relevant_tracks = self.filter_target_tracks(tracks)
        count = len(relevant_tracks)

        if count > self.max_person_count:
            if not self.is_on_cooldown(timestamp=now):
                self.mark_alerted(timestamp=now)
                # Group center of crowd
                avg_x = sum(t.center[0] for t in relevant_tracks) / count
                avg_y = sum(t.center[1] for t in relevant_tracks) / count
                return [
                    RuleAlert(
                        rule_name=self.name,
                        message=f"Overcrowding detected: {count} persons present (threshold: {self.max_person_count})",
                        severity="MEDIUM",
                        timestamp=now,
                        class_name="person",
                        confidence=1.0,
                        metadata={
                            "count": count,
                            "threshold": self.max_person_count,
                            "centroid": (avg_x, avg_y),
                        },
                    )
                ]

        return []
