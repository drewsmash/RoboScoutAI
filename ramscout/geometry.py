"""Homography between a broadcast frame and the 2026 field."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH

# Default trapezoid in normalized crop coordinates (top-left, top-right, bottom-right, bottom-left).
DEFAULT_SRC_NORM = (
    (0.18, 0.08),
    (0.82, 0.08),
    (0.97, 0.92),
    (0.03, 0.92),
)

FIELD_DST = np.float32(
    [
        [0.0, 0.0],
        [FIELD_LENGTH, 0.0],
        [FIELD_LENGTH, FIELD_WIDTH],
        [0.0, FIELD_WIDTH],
    ]
)


def default_source_points(frame_w: int, frame_h: int, crop_top: float = 0.10, crop_bottom: float = 0.65) -> np.ndarray:
    """Build default source corners from the existing 10–65% broadcast crop."""
    y0 = crop_top * frame_h
    y1 = crop_bottom * frame_h
    crop_h = max(y1 - y0, 1.0)
    pts = []
    for nx, ny in DEFAULT_SRC_NORM:
        pts.append([nx * frame_w, y0 + ny * crop_h])
    return np.float32(pts)


def homography_from_corners(src_points: Sequence[Sequence[float]]) -> np.ndarray:
    import cv2

    src = np.float32(src_points)
    if src.shape != (4, 2):
        raise ValueError("Calibration needs exactly four (x, y) corners.")
    return cv2.getPerspectiveTransform(src, FIELD_DST)


def project_points(points: Sequence[Sequence[float]], homography: np.ndarray) -> np.ndarray:
    import cv2

    pts = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    mapped = cv2.perspectiveTransform(pts, homography)
    return mapped.reshape(-1, 2)


def crop_bounds(frame_h: int, top: float = 0.10, bottom: float = 0.65) -> tuple[int, int]:
    y0 = int(frame_h * top)
    y1 = int(frame_h * bottom)
    if y1 <= y0:
        y1 = min(frame_h, y0 + 1)
    return y0, y1
