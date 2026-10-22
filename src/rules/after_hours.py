"""After-Hours Intrusion Detection Rule."""

import datetime
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.rules.base import BaseRule, RuleAlert
from src.tracker import TrackedObject


class AfterHoursRule(BaseRule):
    """Detects human presence during unauthorized / curfew hours."""

    def __init__(
        self,
        start_time: str = "22:00",
        end_time: str = "06:00",
        name: str = "After-Hours Intrusion",
        enabled: bool = True,
        cooldown_sec: float = 15.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the After-Hours rule.

        Args:
            start_time: Start of restricted period in 'HH:MM' 24h format.
            end_time: End of restricted period in 'HH:MM' 24h format.
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
        self.start_time_str = start_time
        self.end_time_str = end_time
        self.start_time = datetime.datetime.strptime(start_time, "%H:%M").time()
        self.end_time = datetime.datetime.strptime(end_time, "%H:%M").time()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AfterHoursRule":
        """Instantiates rule from configuration dictionary."""
        cfg = config.get("rules", {}).get("after_hours", {})
        return cls(
            start_time=cfg.get("start_time", "22:00"),
            end_time=cfg.get("end_time", "06:00"),
            name=cfg.get("name", "After-Hours Intrusion"),
            enabled=cfg.get("enabled", True),
            cooldown_sec=cfg.get("cooldown_sec", 15.0),
            target_classes=cfg.get("target_classes", ["person"]),
        )

    def is_current_time_after_hours(self, dt: Optional[datetime.datetime] = None) -> bool:
        """Checks if given or current time falls within restricted hours (including midnight crossing).

        Args:
            dt: Optional datetime instance (defaults to datetime.datetime.now()).

        Returns:
            True if within restricted hours, False otherwise.
        """
        now_dt = dt or datetime.datetime.now()
        current_time = now_dt.time()

        if self.start_time <= self.end_time:
            # Simple window (e.g., 01:00 to 05:00)
            return self.start_time <= current_time < self.end_time
        else:
            # Wraps over midnight (e.g., 22:00 to 06:00)
            return current_time >= self.start_time or current_time < self.end_time

    def evaluate(
        self,
        tracks: List[TrackedObject],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
        override_now: Optional[datetime.datetime] = None,
    ) -> List[RuleAlert]:
        """Evaluates whether any unauthorized persons are present after hours."""
        if not self.enabled:
            return []

        now_dt = override_now or (
            datetime.datetime.fromtimestamp(timestamp)
            if timestamp is not None
            else datetime.datetime.now()
        )

        if not self.is_current_time_after_hours(now_dt):
            return []

        now = timestamp if timestamp is not None else time.time()
        alerts: List[RuleAlert] = []

        relevant_tracks = self.filter_target_tracks(tracks)
        for track in relevant_tracks:
            if not self.is_on_cooldown(track_id=track.track_id, timestamp=now):
                self.mark_alerted(track_id=track.track_id, timestamp=now)
                time_str = now_dt.strftime("%H:%M:%S")
                alerts.append(
                    RuleAlert(
                        rule_name=self.name,
                        message=(
                            f"{track.class_name.capitalize()} (ID {track.track_id}) detected during "
                            f"after-hours ({time_str} inside restricted window {self.start_time_str}-{self.end_time_str})"
                        ),
                        severity="CRITICAL",
                        timestamp=now,
                        track_id=track.track_id,
                        class_name=track.class_name,
                        confidence=track.confidence,
                        box=track.box,
                        metadata={
                            "detected_time": time_str,
                            "window": f"{self.start_time_str}-{self.end_time_str}",
                        },
                    )
                )

        return alerts
