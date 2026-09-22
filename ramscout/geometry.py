"""Homography between a broadcast frame and the field.

Invalid projections must never be clamped to the field edge and treated as
real robot positions — that produces the "corner spider" trails users report.
"""

from __future__ import annotations

from typing import Any, Sequence

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

# Soft margin (inches): tiny overshoot from float noise stays valid but is not
# forced onto the wall. Anything farther out is marked unavailable.
FIELD_SOFT_MARGIN_IN = 18.0
# Hard reject: beyond this, projection is garbage (wrong homography / off-carpet).
FIELD_HARD_MARGIN_IN = 48.0
# Default overview crop bottoms: prefer covering a full widescreen top pane.
DEFAULT_CROP_TOP = 0.02
DEFAULT_CROP_BOTTOM = 0.58


def default_source_points(
    frame_w: int,
    frame_h: int,
    crop_top: float = DEFAULT_CROP_TOP,
    crop_bottom: float = DEFAULT_CROP_BOTTOM,
) -> np.ndarray:
    """Build default source corners from the broadcast overview crop."""
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


def crop_bounds(frame_h: int, top: float = DEFAULT_CROP_TOP, bottom: float = DEFAULT_CROP_BOTTOM) -> tuple[int, int]:
    y0 = int(frame_h * top)
    y1 = int(frame_h * bottom)
    if y1 <= y0:
        y1 = min(frame_h, y0 + 1)
    return y0, y1


def field_point_status(
    x: float,
    y: float,
    *,
    soft_margin: float = FIELD_SOFT_MARGIN_IN,
    hard_margin: float = FIELD_HARD_MARGIN_IN,
) -> tuple[bool, str]:
    """Return ``(valid, reason)`` for a field-inch coordinate.

    Valid points may sit slightly outside the carpet (soft margin) so we do not
    invent wall contacts. Points beyond the hard margin, or non-finite values,
    are unavailable and must not be drawn or scored as field positions.
    """
    try:
        fx = float(x)
        fy = float(y)
    except (TypeError, ValueError):
        return False, "non_numeric"
    if not np.isfinite(fx) or not np.isfinite(fy):
        return False, "non_finite"
    if abs(fx) > FIELD_LENGTH + hard_margin or abs(fy) > FIELD_WIDTH + hard_margin:
        return False, "degenerate"
    if fx < -hard_margin or fy < -hard_margin:
        return False, "outside"
    if fx > FIELD_LENGTH + hard_margin or fy > FIELD_WIDTH + hard_margin:
        return False, "outside"
    # Soft band: still valid for gating/stats, but callers must not clamp to 0/L.
    if fx < -soft_margin or fy < -soft_margin or fx > FIELD_LENGTH + soft_margin or fy > FIELD_WIDTH + soft_margin:
        return False, "outside"
    return True, "ok"


def annotate_field_sample(sample: dict[str, Any], x: float, y: float) -> dict[str, Any]:
    """Write field coords onto ``sample``, marking invalid projections unavailable."""
    valid, reason = field_point_status(x, y)
    sample["x_raw"] = float(x) if np.isfinite(float(x)) else None
    sample["y_raw"] = float(y) if np.isfinite(float(y)) else None
    if valid:
        # Keep true projected inches — do NOT clamp to the wall.
        sample["x"] = float(x)
        sample["y"] = float(y)
        sample["field_valid"] = True
        sample.pop("field_invalid_reason", None)
    else:
        sample["x"] = None
        sample["y"] = None
        sample["field_valid"] = False
        sample["field_invalid_reason"] = reason
    return sample


def is_field_sample_drawable(sample: dict[str, Any] | None) -> bool:
    """True when a sample has usable field coordinates for path rendering."""
    if not sample:
        return False
    if sample.get("gap") or sample.get("camera_cut"):
        return False
    if sample.get("field_valid") is False:
        return False
    x, y = sample.get("x"), sample.get("y")
    if x is None or y is None:
        return False
    try:
        return field_point_status(float(x), float(y))[0]
    except (TypeError, ValueError):
        return False


def reproject_samples(samples: list[dict], src_points: Sequence[Sequence[float]]) -> list[dict]:
    """Map stored full-frame feet (px, py) through a new homography.

    Out-of-field / degenerate projections are marked ``field_valid=False`` with
    ``x``/``y`` cleared — never clamped to the field edge as if they were real.
    """
    feet: list[list[float]] = []
    indexes: list[int] = []
    for i, sample in enumerate(samples):
        if sample.get("px") is None or sample.get("py") is None:
            continue
        feet.append([float(sample["px"]), float(sample["py"])])
        indexes.append(i)
    if not feet:
        return samples
    mapped = project_points(feet, homography_from_corners(src_points))
    for i, (x, y) in zip(indexes, mapped):
        annotate_field_sample(samples[i], float(x), float(y))
    return samples


def sanitize_samples_field_coords(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-validate existing x/y (e.g. after smoothing) without inventing edges."""
    for sample in samples:
        if sample.get("x") is None or sample.get("y") is None:
            if sample.get("field_valid") is not False:
                sample["field_valid"] = False
                sample.setdefault("field_invalid_reason", "missing")
            continue
        annotate_field_sample(sample, float(sample["x"]), float(sample["y"]))
    return samples
