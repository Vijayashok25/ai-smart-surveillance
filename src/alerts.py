"""Alert system: ring-buffer evidence recording, snapshots, video clips, and notifications."""

import json
import logging
import os
import smtplib
import threading
import time
from collections import deque
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from src.database import DatabaseManager
from src.rules.base import RuleAlert
from src.utils import setup_logger

load_dotenv()


class AlertManager:
    """Orchestrates evidence capture, persistence, and external notification dispatch."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        evidence_dir: Union[str, Path] = "evidence",
        save_snapshots: bool = True,
        save_clips: bool = True,
        clip_pre_seconds: float = 5.0,
        clip_post_seconds: float = 5.0,
        audible_beep: bool = False,
        telegram_enabled: bool = False,
        email_enabled: bool = False,
        banner_display_sec: float = 3.0,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes AlertManager.

        Args:
            db_manager: SQLite database manager.
            evidence_dir: Folder to store snapshots and MP4 clips.
            save_snapshots: Toggle saving JPEG snapshots.
            save_clips: Toggle recording MP4 clips.
            clip_pre_seconds: Seconds of video buffered prior to alert.
            clip_post_seconds: Seconds of video recorded following alert.
            audible_beep: Toggle sound prompt on alert.
            telegram_enabled: Dispatch Telegram message if token/chat_id present.
            email_enabled: Dispatch email via SMTP if configured.
            banner_display_sec: Duration to keep warning banner visible on HUD.
            logger: Custom logger instance.
        """
        self.db = db_manager
        self.evidence_dir = Path(evidence_dir)
        self.save_snapshots = save_snapshots
        self.save_clips = save_clips
        self.clip_pre_seconds = float(clip_pre_seconds)
        self.clip_post_seconds = float(clip_post_seconds)
        self.audible_beep = audible_beep
        self.telegram_enabled = telegram_enabled
        self.email_enabled = email_enabled
        self.banner_display_sec = float(banner_display_sec)
        self.logger = logger or setup_logger("AlertManager")

        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        # Ring buffer for pre-alert frames: stores (timestamp, frame_bgr)
        # Assuming ~30fps max, 5 seconds is ~150 frames; set maxlen=300 for safety margin
        self._ring_buffer: Deque[Tuple[float, np.ndarray]] = deque(maxlen=300)
        self._buffer_lock = threading.Lock()

        # In-progress clip recordings
        self._clip_tasks: List[Dict[str, Any]] = []
        self._clip_lock = threading.Lock()

        # Banner state: (message, expiry_timestamp)
        self.active_banner: Optional[Tuple[str, float]] = None

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        db_manager: DatabaseManager,
        logger: Optional[logging.Logger] = None,
    ) -> "AlertManager":
        """Instantiates AlertManager from configuration dictionary."""
        alert_cfg = config.get("alerts", {})
        vis_cfg = config.get("visualization", {})
        telegram_cfg = alert_cfg.get("telegram", {})
        email_cfg = alert_cfg.get("email", {})

        return cls(
            db_manager=db_manager,
            evidence_dir=alert_cfg.get("evidence_dir", "evidence"),
            save_snapshots=alert_cfg.get("save_snapshots", True),
            save_clips=alert_cfg.get("save_clips", True),
            clip_pre_seconds=alert_cfg.get("clip_pre_seconds", 5.0),
            clip_post_seconds=alert_cfg.get("clip_post_seconds", 5.0),
            audible_beep=alert_cfg.get("audible_beep", False),
            telegram_enabled=telegram_cfg.get("enabled", False),
            email_enabled=email_cfg.get("enabled", False),
            banner_display_sec=vis_cfg.get("banner_display_sec", 3.0),
            logger=logger,
        )

    def add_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> None:
        """Buffers the current video frame into the ring buffer and active clip recorders.

        Args:
            frame: Raw or annotated original-resolution BGR frame.
            timestamp: Frame timestamp.
        """
        if frame is None or frame.size == 0:
            return

        ts = timestamp if timestamp is not None else time.time()
        frame_copy = frame.copy()

        with self._buffer_lock:
            self._ring_buffer.append((ts, frame_copy))

        # Feed frames into post-alert clip recording jobs
        with self._clip_lock:
            completed_tasks = []
            for task in self._clip_tasks:
                task["frames"].append(frame_copy)
                if ts >= task["end_time"]:
                    completed_tasks.append(task)

            for task in completed_tasks:
                self._clip_tasks.remove(task)
                # Finalize clip write asynchronously
                threading.Thread(
                    target=self._write_clip_file,
                    args=(task,),
                    daemon=True,
                ).start()

    def trigger(
        self,
        alert: RuleAlert,
        current_frame: np.ndarray,
    ) -> int:
        """Processes an alert trigger: saves evidence, logs, records in DB, and dispatches notices.

        Args:
            alert: The triggered RuleAlert object.
            current_frame: The original-resolution video frame at trigger moment.

        Returns:
            The created SQLite alert record ID.
        """
        now = alert.timestamp
        self.logger.warning(f"ALERT [{alert.severity}]: {alert.message}")

        # Update warning banner
        self.active_banner = (alert.message, now + self.banner_display_sec)

        # 1. Save Snapshot
        snapshot_rel_path: Optional[str] = None
        if self.save_snapshots and current_frame is not None:
            snapshot_rel_path = self._save_snapshot(alert, current_frame, now)

        # 2. Insert into SQLite DB
        alert_id = self.db.insert_alert(
            timestamp=now,
            rule_name=alert.rule_name,
            severity=alert.severity,
            object_class=alert.class_name,
            track_id=alert.track_id,
            confidence=alert.confidence,
            message=alert.message,
            snapshot_path=snapshot_rel_path,
            clip_path=None,  # Updated after clip writes
            metadata_str=json.dumps(alert.metadata) if alert.metadata else None,
        )

        # 3. Queue 5s pre- and 5s post-clip recording
        if self.save_clips:
            self._start_clip_recording(alert, alert_id, now)

        # 4. Optional Audible Beep
        if self.audible_beep:
            self._trigger_beep()

        # 5. External Notifications (async)
        if self.telegram_enabled:
            threading.Thread(
                target=self._send_telegram_notification,
                args=(alert, snapshot_rel_path),
                daemon=True,
            ).start()

        if self.email_enabled:
            threading.Thread(
                target=self._send_email_notification,
                args=(alert,),
                daemon=True,
            ).start()

        return alert_id

    def _save_snapshot(self, alert: RuleAlert, frame: np.ndarray, timestamp: float) -> str:
        """Saves JPEG snapshot to evidence directory."""
        slug = alert.rule_name.lower().replace(" ", "_").replace("/", "_")
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"snap_{dt_str}_{slug}.jpg"
        filepath = self.evidence_dir / filename

        cv2.imwrite(str(filepath), frame)
        self.logger.info(f"Saved snapshot evidence: {filepath}")
        return str(filepath)

    def _start_clip_recording(self, alert: RuleAlert, alert_id: int, timestamp: float) -> None:
        """Schedules video clip recording using pre-alert buffer + subsequent post-alert frames."""
        with self._buffer_lock:
            # Grab all frames within pre-seconds
            pre_cutoff = timestamp - self.clip_pre_seconds
            pre_frames = [f for ts, f in self._ring_buffer if ts >= pre_cutoff]

        slug = alert.rule_name.lower().replace(" ", "_").replace("/", "_")
        dt_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"clip_{dt_str}_{slug}_{alert_id}.mp4"
        filepath = self.evidence_dir / filename

        task = {
            "alert_id": alert_id,
            "filepath": filepath,
            "frames": pre_frames,
            "end_time": timestamp + self.clip_post_seconds,
            "rule_name": alert.rule_name,
        }

        with self._clip_lock:
            self._clip_tasks.append(task)

    def _write_clip_file(self, task: Dict[str, Any]) -> None:
        """Encodes gathered frames into an MP4 file and updates SQLite."""
        frames = task["frames"]
        filepath = task["filepath"]
        alert_id = task["alert_id"]

        if not frames:
            return

        h, w = frames[0].shape[:2]
        fps = 20.0  # Safe default surveillance encoding frame rate

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(filepath), fourcc, fps, (w, h))

        for f in frames:
            # Resize if dimensions differ
            if f.shape[:2] != (h, w):
                f = cv2.resize(f, (w, h))
            writer.write(f)

        writer.release()
        self.logger.info(f"Evidence clip finalized: {filepath} ({len(frames)} frames)")

        # Update SQLite with clip path
        try:
            with self.db._lock:
                with self.db._get_connection() as conn:
                    conn.cursor().execute(
                        "UPDATE alerts SET clip_path = ? WHERE id = ?",
                        (str(filepath), alert_id),
                    )
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Failed to update clip_path in DB for alert #{alert_id}: {e}")

    def _trigger_beep(self) -> None:
        """Triggers system sound prompt."""
        try:
            import winsound
            winsound.Beep(1200, 300)
        except Exception:
            # Terminal bell fallback
            print("\a", end="", flush=True)

    def _send_telegram_notification(self, alert: RuleAlert, snapshot_path: Optional[str]) -> None:
        """Sends Telegram alert message via Bot API."""
        token = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            self.logger.warning("Telegram alerts enabled but TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing.")
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        text = f"🚨 *SECURITY ALERT: {alert.rule_name}*\n\n{alert.message}\nSeverity: {alert.severity}"
        try:
            requests.post(url, json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}, timeout=5.0)
            # If snapshot exists, send photo
            if snapshot_path and os.path.exists(snapshot_path):
                photo_url = f"https://api.telegram.org/bot{token}/sendPhoto"
                with open(snapshot_path, "rb") as photo:
                    requests.post(photo_url, data={"chat_id": chat_id}, files={"photo": photo}, timeout=10.0)
        except Exception as e:
            self.logger.error(f"Failed to dispatch Telegram alert: {e}")

    def _send_email_notification(self, alert: RuleAlert) -> None:
        """Sends SMTP email alert."""
        smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", 587))
        username = os.getenv("SMTP_USERNAME")
        password = os.getenv("SMTP_PASSWORD")
        recipient = os.getenv("ALERT_RECIPIENT_EMAIL")

        if not username or not password or not recipient:
            self.logger.warning("Email alerts enabled but SMTP credentials or recipient missing in .env.")
            return

        msg = MIMEMultipart()
        msg["From"] = username
        msg["To"] = recipient
        msg["Subject"] = f"[SECURITY ALERT] {alert.rule_name} Triggered"

        body = f"AI Smart Surveillance Alert\n\nRule: {alert.rule_name}\nSeverity: {alert.severity}\nTime: {datetime.fromtimestamp(alert.timestamp)}\nDetails: {alert.message}\n"
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(smtp_server, smtp_port, timeout=10.0) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            self.logger.info(f"Sent email alert to {recipient}")
        except Exception as e:
            self.logger.error(f"Failed to dispatch email alert: {e}")
