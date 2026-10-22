"""YOLOv8 Object Detection module with confidence filtering, NMS, and coordinate scaling."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # Handled gracefully in detector initialization

from src.utils import numpy_nms, setup_logger


@dataclass
class Detection:
    """Represents a single object detection instance."""

    box: Tuple[float, float, float, float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Tuple[float, float]:
        """Calculates (cx, cy) center of bounding box."""
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Calculates (cx, y2) bottom center representing ground contact."""
        return ((self.box[0] + self.box[2]) / 2.0, self.box[3])

    @property
    def width(self) -> float:
        """Returns box width."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Returns box height."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Returns box area."""
        return self.width * self.height


class ObjectDetector:
    """Wrapper around Ultralytics YOLOv8 for surveillance object detection."""

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        target_classes: Optional[Sequence[str]] = None,
        device: str = "auto",
        logger: Optional[logging.Logger] = None,
    ) -> None:
        """Initializes the YOLOv8 detector.

        Args:
            model_path: Model weights file or name (e.g. 'yolov8n.pt').
            confidence_threshold: Minimum confidence score to accept detection.
            iou_threshold: IoU threshold for NMS.
            target_classes: Optional list of class names to keep (e.g. ['person', 'car']).
            device: 'auto', 'cpu', 'cuda', 'mps'.
            logger: Custom logger instance.
        """
        self.logger = logger or setup_logger("Detector")
        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.target_classes = [c.lower() for c in target_classes] if target_classes else None
        self.device = self._resolve_device(device)

        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Please run 'pip install ultralytics'."
            )

        self.logger.info(
            f"Loading YOLO model '{model_path}' on device '{self.device}' with conf={self.confidence_threshold}, iou={self.iou_threshold}"
        )
        try:
            self.model = YOLO(model_path)
            # Move model explicitly to resolved device
            self.model.to(self.device)
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model from '{model_path}': {e}")
            raise

        # Map class names to class IDs
        self.class_names: Dict[int, str] = self.model.names
        self.target_class_ids: Optional[List[int]] = None
        if self.target_classes:
            name_to_id = {v.lower(): k for k, v in self.class_names.items()}
            self.target_class_ids = [
                name_to_id[c] for c in self.target_classes if c in name_to_id
            ]
            self.logger.info(
                f"Filtering for classes: {self.target_classes} -> IDs: {self.target_class_ids}"
            )

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        logger: Optional[logging.Logger] = None,
    ) -> "ObjectDetector":
        """Instantiates ObjectDetector from configuration dictionary."""
        det_cfg = config.get("detection", {})
        sys_cfg = config.get("system", {})
        device = sys_cfg.get("device", "auto")

        return cls(
            model_path=det_cfg.get("model_path", "yolov8n.pt"),
            confidence_threshold=det_cfg.get("confidence_threshold", 0.50),
            iou_threshold=det_cfg.get("iou_threshold", 0.45),
            target_classes=det_cfg.get("classes"),
            device=device,
            logger=logger,
        )

    def _resolve_device(self, requested: str) -> str:
        """Determines optimal PyTorch execution device."""
        req = requested.lower()
        if req == "cuda" and torch.cuda.is_available():
            return "cuda"
        if req == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if req == "auto":
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return "cpu"

    def detect(
        self,
        frame: np.ndarray,
        ratio: float = 1.0,
        pad: Tuple[float, float] = (0.0, 0.0),
        original_shape: Optional[Tuple[int, int]] = None,
        apply_custom_nms: bool = False,
    ) -> Tuple[List[Detection], float]:
        """Runs detection inference on preprocessed frame and maps coordinates back to original frame.

        Args:
            frame: Preprocessed image (H, W, C) passed to YOLO.
            ratio: Scaling ratio used during letterbox preprocessing.
            pad: (pad_w, pad_h) offsets from letterbox preprocessing.
            original_shape: (height, width) of the original frame before letterboxing.
            apply_custom_nms: Whether to run secondary standalone NumPy NMS.

        Returns:
            Tuple of:
                - List of Detection objects in original frame coordinate space.
                - Inference time in milliseconds.
        """
        if frame is None or frame.size == 0:
            return [], 0.0

        if original_shape is None:
            original_shape = (frame.shape[0], frame.shape[1])

        start_time = time.perf_counter()

        # Run inference using YOLOv8
        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                classes=self.target_class_ids,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            self.logger.error(f"Inference error: {e}")
            return [], 0.0

        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        if not results or len(results) == 0:
            return [], inference_time_ms

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return [], inference_time_ms

        # Extract coordinates, confidences, and class indices
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        # Map coordinates back to original image space
        pad_w, pad_h = pad
        orig_h, orig_w = original_shape[:2]

        mapped_boxes = xyxy.copy()
        mapped_boxes[:, [0, 2]] -= pad_w
        mapped_boxes[:, [1, 3]] -= pad_h
        if ratio > 0:
            mapped_boxes[:, :4] /= ratio

        # Clip coordinates
        mapped_boxes[:, [0, 2]] = np.clip(mapped_boxes[:, [0, 2]], 0, orig_w)
        mapped_boxes[:, [1, 3]] = np.clip(mapped_boxes[:, [1, 3]], 0, orig_h)

        # Optional secondary standalone NumPy NMS
        keep_indices = list(range(len(mapped_boxes)))
        if apply_custom_nms and len(mapped_boxes) > 1:
            keep_indices = numpy_nms(mapped_boxes, confs, iou_threshold=self.iou_threshold)

        detections: List[Detection] = []
        for idx in keep_indices:
            cid = int(cls_ids[idx])
            cname = self.class_names.get(cid, f"class_{cid}")
            box = (
                float(mapped_boxes[idx, 0]),
                float(mapped_boxes[idx, 1]),
                float(mapped_boxes[idx, 2]),
                float(mapped_boxes[idx, 3]),
            )
            detections.append(
                Detection(
                    box=box,
                    confidence=float(confs[idx]),
                    class_id=cid,
                    class_name=cname,
                )
            )

        return detections, inference_time_ms
