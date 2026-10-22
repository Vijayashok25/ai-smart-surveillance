"""End-to-end surveillance pipeline orchestrating capture, preprocessing, inference, tracking, rules, and visualization."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.alerts import AlertManager
from src.capture import VideoCaptureThread
from src.database import DatabaseManager
from src.detector import Detection, ObjectDetector
from src.preprocess import Preprocessor
from src.rules.abandoned_object import AbandonedObjectRule
from src.rules.after_hours import AfterHoursRule
from src.rules.base import BaseRule, RuleAlert
from src.rules.crowding import CrowdingRule
from src.rules.loitering import LoiteringRule
from src.rules.zone_intrusion import ZoneIntrusionRule
from src.tracker import TrackedObject, create_tracker
from src.utils import FPSCounter, setup_logger
from src.visualizer import SurveillanceVisualizer


class SurveillancePipeline:
    """Master pipeline integrating all computer vision and alerting subsystems."""

    def __init__(
        self,
        config: Dict[str, Any],
        override_source: Optional[Union[int, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes all pipeline components from configuration dictionary.

        Args:
            config: Full dictionary from config.yaml.
            override_source: Optional video source to override config setting.
            logger: Custom logger instance.
        """
        self.config = config
        self.logger = logger or setup_logger("Pipeline", level=config.get("system", {}).get("log_level", "INFO"))

        # 1. Database
        db_path = config.get("database", {}).get("db_path", "surveillance.db")
        self.db = DatabaseManager(db_path=db_path, logger=self.logger)

        # 2. Alert Manager
        self.alert_manager = AlertManager.from_config(config, db_manager=self.db, logger=self.logger)

        # 3. Preprocessor
        self.preprocessor = Preprocessor.from_config(config)

        # 4. Detector
        self.detector = ObjectDetector.from_config(config, logger=self.logger)

        # 5. Tracker
        track_cfg = config.get("tracking", {})
        self.tracker = create_tracker(
            tracker_type=track_cfg.get("tracker_type", "bytetrack"),
            max_trail_points=track_cfg.get("max_trail_points", 30),
            max_disappeared_frames=track_cfg.get("max_disappeared_frames", 30),
            max_distance_px=track_cfg.get("max_distance_px", 80.0),
            movement_threshold_px=config.get("rules", {}).get("abandoned_object", {}).get("movement_threshold_px", 15.0),
            logger=self.logger,
        )

        # 6. Rules Engine
        self.rules: List[BaseRule] = self._init_rules(config)

        # 7. Visualizer
        self.visualizer = SurveillanceVisualizer.from_config(config)

        # 8. FPS Counter
        self.fps_counter = FPSCounter(window_size=30)

        # 9. Video Capture
        self.capture_source = override_source if override_source is not None else config.get("video", {}).get("source", 0)
        self.capture: Optional[VideoCaptureThread] = None

        self._running = False
        self._total_alerts_count = self.db.get_total_count()

    def _init_rules(self, config: Dict[str, Any]) -> List[BaseRule]:
        """Initializes and returns configured suspicious activity rule objects."""
        rules: List[BaseRule] = []
        rules_cfg = config.get("rules", {})

        if "zone_intrusion" in rules_cfg:
            rules.append(ZoneIntrusionRule.from_config(config))
        if "loitering" in rules_cfg:
            rules.append(LoiteringRule.from_config(config))
        if "abandoned_object" in rules_cfg:
            rules.append(AbandonedObjectRule.from_config(config))
        if "crowding" in rules_cfg:
            rules.append(CrowdingRule.from_config(config))
        if "after_hours" in rules_cfg:
            rules.append(AfterHoursRule.from_config(config))

        self.logger.info(f"Loaded {len(rules)} surveillance rules: {[r.name for r in rules]}")
        return rules

    def start_capture(self) -> None:
        """Starts video capture background thread."""
        if self.capture is None:
            self.capture = VideoCaptureThread.from_config(
                self.config,
                override_source=self.capture_source,
                logger=self.logger,
            )
            self.capture.start()

    def stop_capture(self) -> None:
        """Stops video capture and releases resources."""
        if self.capture is not None:
            self.capture.stop()
            self.capture = None

    def process_frame(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[TrackedObject], List[RuleAlert], float, float]:
        """Executes full pipeline processing on a single frame.

        Args:
            frame: Raw BGR frame.
            timestamp: Frame timestamp in seconds.

        Returns:
            Tuple of:
                - annotated_frame (np.ndarray)
                - active_tracks (List[TrackedObject])
                - triggered_alerts (List[RuleAlert])
                - current_fps (float)
                - inference_time_ms (float)
        """
        now = timestamp if timestamp is not None else time.time()
        orig_shape = (frame.shape[0], frame.shape[1])

        # Add to alert ring buffer
        self.alert_manager.add_frame(frame, timestamp=now)

        # Check frame skip
        if not self.preprocessor.should_process_frame():
            # Skip inference for this tick
            fps = self.fps_counter.update()
            annotated = self.visualizer.render(
                frame=frame,
                tracks=list(getattr(self.tracker, "tracks", {}).values()),
                rules=self.rules,
                banner_info=self.alert_manager.active_banner,
                current_time=now,
                fps=fps,
                inference_ms=self.fps_counter.avg_inference_ms,
                total_alerts=self._total_alerts_count,
            )
            return annotated, [], [], fps, 0.0

        # 1. Preprocess (Letterbox + enhancements)
        preprocessed_img, ratio, pad = self.preprocessor.preprocess(frame)

        # 2. Detection (YOLO inference)
        detections, inference_ms = self.detector.detect(
            frame=preprocessed_img,
            ratio=ratio,
            pad=pad,
            original_shape=orig_shape,
            apply_custom_nms=True,
        )

        # 3. Tracking update
        tracks = self.tracker.update(detections, timestamp=now)

        # 4. Rules evaluation
        triggered_alerts: List[RuleAlert] = []
        for rule in self.rules:
            if rule.enabled:
                rule_alerts = rule.evaluate(tracks, frame, timestamp=now)
                for alert in rule_alerts:
                    self.alert_manager.trigger(alert, current_frame=frame)
                    triggered_alerts.append(alert)
                    self._total_alerts_count += 1

        # 5. FPS update
        fps = self.fps_counter.update(inference_ms=inference_ms)

        # 6. Render overlays
        annotated = self.visualizer.render(
            frame=frame,
            tracks=tracks,
            rules=self.rules,
            banner_info=self.alert_manager.active_banner,
            current_time=now,
            fps=fps,
            inference_ms=inference_ms,
            total_alerts=self._total_alerts_count,
        )

        return annotated, tracks, triggered_alerts, fps, inference_ms

    def stream_frames(
        self,
    ) -> Generator[Tuple[np.ndarray, List[TrackedObject], List[RuleAlert], float, float], None, None]:
        """Generator yielding processed frames from the active capture stream."""
        self.start_capture()
        self._running = True

        try:
            while self._running:
                success, frame, ts = self.capture.read(timeout=1.0)
                if not success or frame is None:
                    # Stream ended or queue timeout
                    if self.capture and self.capture._is_file:
                        break
                    continue

                annotated, tracks, alerts, fps, inf_ms = self.process_frame(frame, timestamp=ts)
                yield annotated, tracks, alerts, fps, inf_ms
        finally:
            self.stop_capture()

    def stop(self) -> None:
        """Flags the pipeline to terminate streaming loops."""
        self._running = False
        self.stop_capture()
