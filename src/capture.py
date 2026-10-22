"""Threaded video capture with frame queue dropping and auto-reconnection."""

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.utils import setup_logger


class VideoCaptureThread:
    """Reads video frames in a background thread to keep processing latency strictly minimal."""

    def __init__(
        self,
        source: Union[int, str, Path] = 0,
        queue_size: int = 2,
        reconnect_delay_sec: float = 2.0,
        max_reconnect_attempts: int = 5,
        fps_limit: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes VideoCaptureThread.

        Args:
            source: Webcam device index (int or '0'), video file path, or RTSP URL.
            queue_size: Max frames queued (small size ensures freshest frames).
            reconnect_delay_sec: Base delay between reconnect attempts.
            max_reconnect_attempts: Maximum retry count before raising error.
            fps_limit: Optional FPS cap to avoid maxing out CPU on fast file streams.
            logger: Custom logger instance.
        """
        self.source = self._clean_source(source)
        self.queue_size = max(1, queue_size)
        self.reconnect_delay_sec = reconnect_delay_sec
        self.max_reconnect_attempts = max_reconnect_attempts
        self.fps_limit = fps_limit
        self.logger = logger or setup_logger("VideoCapture")

        self.queue: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self._cap: Optional[cv2.VideoCapture] = None
        self._thread: Optional[threading.Thread] = None
        self._stopped = threading.Event()
        self._is_file = isinstance(self.source, str) and Path(self.source).is_file()

        # Stream properties
        self.frame_width = 0
        self.frame_height = 0
        self.stream_fps = 30.0

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        override_source: Optional[Union[int, str]] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "VideoCaptureThread":
        """Instantiates VideoCaptureThread from configuration dictionary."""
        video_cfg = config.get("video", {})
        src = override_source if override_source is not None else video_cfg.get("source", 0)

        return cls(
            source=src,
            queue_size=video_cfg.get("queue_size", 2),
            reconnect_delay_sec=video_cfg.get("reconnect_delay_sec", 2.0),
            max_reconnect_attempts=video_cfg.get("max_reconnect_attempts", 5),
            fps_limit=video_cfg.get("fps_limit"),
            logger=logger,
        )

    @staticmethod
    def _clean_source(src: Union[int, str, Path]) -> Union[int, str]:
        """Converts digit strings into integer indices."""
        if isinstance(src, int):
            return src
        src_str = str(src).strip()
        if src_str.isdigit():
            return int(src_str)
        return src_str

    def _open_capture(self) -> bool:
        """Opens the video source with appropriate backends."""
        if self._cap is not None:
            self._cap.release()

        self.logger.info(f"Opening video source: {self.source}")
        if isinstance(self.source, int):
            # On Windows, cv2.CAP_DSHOW or default
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not self._cap.isOpened():
                self._cap = cv2.VideoCapture(self.source)
        else:
            self._cap = cv2.VideoCapture(str(self.source))

        if not self._cap.isOpened():
            self.logger.error(f"Could not open video source: {self.source}")
            return False

        self.frame_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        self.stream_fps = fps if (fps and fps > 0 and fps < 120) else 30.0

        self.logger.info(
            f"Source connected: {self.frame_width}x{self.frame_height} @ {self.stream_fps:.1f} FPS"
        )
        return True

    def start(self) -> "VideoCaptureThread":
        """Starts the background capture worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return self

        if not self._open_capture():
            raise RuntimeError(f"Failed to open video source: {self.source}")

        self._stopped.clear()
        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()
        return self

    def _capture_worker(self) -> None:
        """Continuously reads frames and maintains queue with drop-oldest policy."""
        reconnect_attempts = 0
        frame_interval = 1.0 / self.fps_limit if self.fps_limit else 0.0

        while not self._stopped.is_set():
            loop_start = time.perf_counter()

            if self._cap is None or not self._cap.isOpened():
                if not self._handle_reconnect(reconnect_attempts):
                    break
                reconnect_attempts += 1
                continue

            success, frame = self._cap.read()

            if not success or frame is None:
                if self._is_file:
                    self.logger.info("End of recorded video file reached.")
                    break

                self.logger.warning("Frame read failed / lost stream; initiating reconnect...")
                if not self._handle_reconnect(reconnect_attempts):
                    break
                reconnect_attempts += 1
                continue

            # Successful read, reset reconnect counter
            reconnect_attempts = 0

            # Drop oldest frame if queue full to guarantee lowest processing latency
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.queue.put((time.time(), frame), block=False)
            except queue.Full:
                pass

            # Optional FPS pacing for offline files
            if frame_interval > 0:
                elapsed = time.perf_counter() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        self.logger.info("Video capture thread exited.")

    def _handle_reconnect(self, attempt_count: int) -> bool:
        """Exponential backoff reconnect handler."""
        if attempt_count >= self.max_reconnect_attempts:
            self.logger.error(
                f"Exceeded max reconnection attempts ({self.max_reconnect_attempts}). Stopping."
            )
            return False

        delay = self.reconnect_delay_sec * (1.5**attempt_count)
        self.logger.info(f"Retrying connection in {delay:.1f}s (Attempt {attempt_count + 1}/{self.max_reconnect_attempts})...")
        time.sleep(delay)
        return self._open_capture()

    def read(self, timeout: float = 2.0) -> Tuple[bool, Optional[np.ndarray], float]:
        """Retrieves the latest available frame from the queue.

        Args:
            timeout: Seconds to wait for a frame.

        Returns:
            Tuple of (success_bool, frame_array_or_None, timestamp).
        """
        try:
            timestamp, frame = self.queue.get(timeout=timeout)
            return True, frame, timestamp
        except queue.Empty:
            # Check if thread is done
            if self._stopped.is_set() or (self._thread and not self._thread.is_alive()):
                return False, None, 0.0
            return False, None, 0.0

    def stop(self) -> None:
        """Stops the capture worker thread and releases OpenCV resources."""
        self._stopped.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.logger.info("Video source released.")

    def __enter__(self) -> "VideoCaptureThread":
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
