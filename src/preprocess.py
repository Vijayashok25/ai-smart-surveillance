"""Preprocessing pipeline for video frames: letterbox resizing, denoising, and coordinate mapping."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class Preprocessor:
    """Preprocesses video frames for YOLO inference and manages coordinate transformations."""

    def __init__(
        self,
        input_size: int = 640,
        frame_skip: int = 1,
        denoise_enabled: bool = False,
        denoise_method: str = "bilateral",
        brightness_normalize_enabled: bool = False,
    ) -> None:
        """Initializes the Preprocessor.

        Args:
            input_size: Target square dimensions for the letterboxed image.
            frame_skip: Process every Nth frame (1 means process all frames).
            denoise_enabled: Whether to apply denoising filters.
            denoise_method: 'bilateral' or 'gaussian'.
            brightness_normalize_enabled: Whether to apply CLAHE brightness normalization.
        """
        self.input_size = int(input_size)
        self.frame_skip = max(1, int(frame_skip))
        self.denoise_enabled = bool(denoise_enabled)
        self.denoise_method = str(denoise_method).lower()
        self.brightness_normalize_enabled = bool(brightness_normalize_enabled)
        self._frame_count = 0
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "Preprocessor":
        """Factory method to instantiate Preprocessor from configuration dictionary."""
        pre_cfg = config.get("preprocess", {})
        denoise_cfg = pre_cfg.get("denoise", {})
        bright_cfg = pre_cfg.get("brightness_normalize", {})

        return cls(
            input_size=pre_cfg.get("input_size", 640),
            frame_skip=pre_cfg.get("frame_skip", 1),
            denoise_enabled=denoise_cfg.get("enabled", False),
            denoise_method=denoise_cfg.get("method", "bilateral"),
            brightness_normalize_enabled=bright_cfg.get("enabled", False),
        )

    def should_process_frame(self) -> bool:
        """Determines if the current frame index should be processed according to frame_skip.

        Returns:
            True if the current frame should be processed, False if it should be skipped.
        """
        self._frame_count += 1
        return (self._frame_count % self.frame_skip) == 0

    def reset_counter(self) -> None:
        """Resets the internal frame counter."""
        self._frame_count = 0

    def letterbox(
        self,
        image: np.ndarray,
        target_size: Optional[int] = None,
        color: Tuple[int, int, int] = (114, 114, 114),
        auto: bool = False,
        scale_fill: bool = False,
        scale_up: bool = True,
        stride: int = 32,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Resizes and pads image while preserving aspect ratio to fit target_size.

        Args:
            image: Original frame array (H, W, C).
            target_size: Square target size (defaults to self.input_size).
            color: Padding border color (BGR tuple).
            auto: Minimum rectangle padding.
            scale_fill: Stretch image to fill without padding.
            scale_up: Allow scaling up if image is smaller than target_size.
            stride: Stride constraints for YOLO feature maps.

        Returns:
            Tuple of:
                - Letterboxed image (np.ndarray)
                - Scale ratio (float)
                - (pad_w, pad_h) in pixels (Tuple[float, float])
        """
        if target_size is None:
            target_size = self.input_size

        shape = image.shape[:2]  # [height, width]
        new_shape = (target_size, target_size)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scale_up:
            r = min(r, 1.0)

        # Compute padding
        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw = new_shape[1] - new_unpad[0]
        dh = new_shape[0] - new_unpad[1]

        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scale_fill:  # stretch
            dw, dh = 0, 0
            new_unpad = (new_shape[1], new_shape[0])
            r = new_shape[1] / shape[1]

        dw /= 2.0  # divide padding into 2 sides
        dh /= 2.0

        if shape[::-1] != new_unpad:  # resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        image = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return image, float(r), (float(dw), float(dh))

    def apply_enhancements(self, frame: np.ndarray) -> np.ndarray:
        """Applies configured denoising and brightness equalization to a frame.

        Args:
            frame: Input BGR image.

        Returns:
            Enhanced BGR image.
        """
        processed = frame.copy()

        if self.denoise_enabled:
            if self.denoise_method == "bilateral":
                processed = cv2.bilateralFilter(processed, d=5, sigmaColor=50, sigmaSpace=50)
            elif self.denoise_method == "gaussian":
                processed = cv2.GaussianBlur(processed, (5, 5), sigmaX=0)

        if self.brightness_normalize_enabled:
            lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l_clahe = self._clahe.apply(l)
            lab_merged = cv2.merge((l_clahe, a, b))
            processed = cv2.cvtColor(lab_merged, cv2.COLOR_LAB2BGR)

        return processed

    def preprocess(
        self,
        frame: np.ndarray,
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Complete preprocessing pipeline for an input frame.

        Args:
            frame: Original BGR frame.

        Returns:
            Tuple of:
                - Preprocessed letterboxed image ready for YOLO
                - Scale ratio
                - (pad_w, pad_h)
        """
        enhanced = self.apply_enhancements(frame)
        letterboxed, ratio, (pad_w, pad_h) = self.letterbox(enhanced, target_size=self.input_size)
        return letterboxed, ratio, (pad_w, pad_h)

    @staticmethod
    def scale_coords_to_original(
        boxes: Union[np.ndarray, Sequence[Sequence[float]]],
        original_shape: Tuple[int, int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> np.ndarray:
        """Transforms bounding box coordinates from letterbox space back to original image space.

        Args:
            boxes: Array or list of shape (N, 4) in [x1, y1, x2, y2].
            original_shape: Original image (height, width).
            ratio: Scaling ratio applied during letterboxing.
            pad: (pad_w, pad_h) offsets applied during letterboxing.

        Returns:
            NumPy array of shape (N, 4) with coordinates clipped to original image boundaries.
        """
        if len(boxes) == 0:
            return np.empty((0, 4), dtype=np.float32)

        boxes_arr = np.array(boxes, dtype=np.float32).copy()
        pad_w, pad_h = pad

        # Subtract padding
        boxes_arr[:, [0, 2]] -= pad_w
        boxes_arr[:, [1, 3]] -= pad_h

        # Rescale
        if ratio > 0:
            boxes_arr[:, :4] /= ratio

        # Clip to original image boundaries
        orig_h, orig_w = original_shape[:2]
        boxes_arr[:, [0, 2]] = np.clip(boxes_arr[:, [0, 2]], 0, orig_w)
        boxes_arr[:, [1, 3]] = np.clip(boxes_arr[:, [1, 3]], 0, orig_h)

        return boxes_arr
