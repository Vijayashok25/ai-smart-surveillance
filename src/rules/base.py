"""Base class and common data structures for suspicious activity rules."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from src.detector import Detection
from src.tracker import TrackedObject

# Alias for backwards compatibility / rule interfaces
DetectionItem = Detection


@dataclass
class RuleAlert:
    """Represents a suspicious activity alert triggered by a rule."""

    rule_name: str
    message: str
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    timestamp: float = field(default_factory=time.time)
    track_id: Optional[int] = None
    class_name: Optional[str] = None
    confidence: float = 0.0
    box: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseRule(ABC):
    """Abstract base class that all suspicious activity rules must implement."""

    def __init__(
        self,
        name: str,
        enabled: bool = True,
        cooldown_sec: float = 10.0,
        target_classes: Optional[Sequence[str]] = None,
    ) -> None:
        """Initializes the base rule.

        Args:
            name: Human-readable identifier for the rule.
            enabled: Whether this rule actively evaluates frames.
            cooldown_sec: Minimum seconds between consecutive triggers to prevent spam.
            target_classes: Object class names this rule evaluates (e.g. ['person']).
        """
        self.name = name
        self.enabled = bool(enabled)
        self.cooldown_sec = float(cooldown_sec)
        self.target_classes = [c.lower() for c in target_classes] if target_classes else []
        self._last_alert_time: float = 0.0
        self._track_alert_times: Dict[int, float] = {}

    def is_on_cooldown(self, track_id: Optional[int] = None, timestamp: Optional[float] = None) -> bool:
        """Checks if the rule or a specific track is currently within its cooldown window.

        Args:
            track_id: Optional ID of the object being evaluated.
            timestamp: Current timestamp (defaults to time.time()).

        Returns:
            True if within cooldown interval, False if eligible to alert.
        """
        now = timestamp if timestamp is not None else time.time()

        if track_id is not None:
            last = self._track_alert_times.get(track_id, 0.0)
            return (now - last) < self.cooldown_sec

        return (now - self._last_alert_time) < self.cooldown_sec

    def mark_alerted(self, track_id: Optional[int] = None, timestamp: Optional[float] = None) -> None:
        """Updates cooldown timestamp for the rule and optional track."""
        now = timestamp if timestamp is not None else time.time()
        self._last_alert_time = now
        if track_id is not None:
            self._track_alert_times[track_id] = now

    def reset_cooldowns(self) -> None:
        """Resets all cooldown timers."""
        self._last_alert_time = 0.0
        self._track_alert_times.clear()

    def filter_target_tracks(self, tracks: Sequence[TrackedObject]) -> List[TrackedObject]:
        """Filters active tracks matching configured target classes."""
        if not self.target_classes:
            return list(tracks)
        return [t for t in tracks if t.class_name.lower() in self.target_classes]

    @abstractmethod
    def evaluate(
        self,
        tracks: List[TrackedObject],
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> List[RuleAlert]:
        """Evaluates tracked objects against rule conditions.

        Args:
            tracks: List of active TrackedObject instances.
            frame: Current video frame (BGR).
            timestamp: Timestamp in seconds.

        Returns:
            List of triggered RuleAlert instances.
        """
        pass
