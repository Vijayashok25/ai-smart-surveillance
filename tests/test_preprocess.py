"""Unit tests for frame preprocessing and coordinate transforms."""

import numpy as np
import pytest

from src.preprocess import Preprocessor


def test_letterbox_dimensions():
    """Letterbox output must match requested square target size."""
    preprocessor = Preprocessor(input_size=640)
    sample_frame = np.zeros((480, 720, 3), dtype=np.uint8)

    letterboxed, ratio, (pad_w, pad_h) = preprocessor.letterbox(sample_frame, target_size=640)

    assert letterboxed.shape == (640, 640, 3)
    assert ratio > 0.0
    assert pad_w >= 0.0
    assert pad_h >= 0.0


def test_scale_coords_roundtrip():
    """Coordinates mapped to letterbox and scaled back must match original."""
    preprocessor = Preprocessor(input_size=640)
    orig_h, orig_w = 480, 720
    sample_frame = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)

    _, ratio, pad = preprocessor.letterbox(sample_frame, target_size=640)

    # Box in original coords
    orig_box = np.array([[50.0, 60.0, 200.0, 300.0]])

    # Forward mapping to letterbox space
    letterbox_box = orig_box * ratio
    letterbox_box[:, [0, 2]] += pad[0]
    letterbox_box[:, [1, 3]] += pad[1]

    # Invert back using static method
    recovered_box = Preprocessor.scale_coords_to_original(
        letterbox_box,
        original_shape=(orig_h, orig_w),
        ratio=ratio,
        pad=pad,
    )

    np.testing.assert_allclose(orig_box, recovered_box, atol=1e-3)


def test_frame_skipping():
    """Frame skip should alternate processing according to stride."""
    preprocessor = Preprocessor(frame_skip=3)

    assert preprocessor.should_process_frame() is False  # Frame 1
    assert preprocessor.should_process_frame() is False  # Frame 2
    assert preprocessor.should_process_frame() is True   # Frame 3
    assert preprocessor.should_process_frame() is False  # Frame 4
