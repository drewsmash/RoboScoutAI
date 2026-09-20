"""Bird's-eye-view (BEV) helpers using depth + camera-angle cues.

Depth Anything V2 (or classical depth) estimates how steeply the broadcast
camera looks at the field. We warp that into trapezoid source corners for the
existing field homography so top-down tracking accounts for camera angle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np

from ramscout.depth import DepthResult, estimate_depth, refine_foot_point
from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.geometry import DEFAULT_SRC_NORM, homography_from_corners, project_points


@dataclass
class BevCalibration:
    """Depth-aware source corners for the broadcast → field homography."""

    src_points: list[list[float]]
    pitch_deg: float
    tilt_strength: float
    depth_source: str
    detail: str
    used_depth: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_bev(
    frame_bgr: np.ndarray,
    *,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    prefer_neural: bool = True,
    base_norm: Sequence[Sequence[float]] | None = None,
) -> tuple[BevCalibration, DepthResult]:
    """Build BEV source corners from a calibration frame + depth estimate."""
    h, w = frame_bgr.shape[:2]
    y0 = int(h * crop_top)
    y1 = int(h * crop_bottom)
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h))
    crop = frame_bgr[y0:y1, :]
    depth_result = estimate_depth(crop, prefer_neural=prefer_neural)

    norm = _tilt_adjusted_norm(base_norm or DEFAULT_SRC_NORM, depth_result.tilt_strength, depth_result.pitch_deg)
    crop_h = float(max(y1 - y0, 1))
    pts: list[list[float]] = []
    for nx, ny in norm:
        pts.append([float(nx) * w, float(y0) + float(ny) * crop_h])

    # Optional: nudge left/right bottom corners using depth symmetry so a
    # slightly skewed camera still lands blue-left / red-right cleanly.
    pts = _nudge_from_depth_edges(pts, depth_result.depth, y0)

    cal = BevCalibration(
        src_points=[[round(p[0], 2), round(p[1], 2)] for p in pts],
        pitch_deg=depth_result.pitch_deg,
        tilt_strength=depth_result.tilt_strength,
        depth_source=depth_result.source,
        detail=(
            f"BEV from {depth_result.source}: pitch≈{depth_result.pitch_deg:.0f}°, "
            f"tilt={depth_result.tilt_strength:.2f}. {depth_result.detail}"
        ),
        used_depth=True,
    )
    return cal, depth_result


def project_detections_bev(
    feet_px: Sequence[Sequence[float]],
    src_points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Map pixel foot points through the BEV/homography onto field inches."""
    H = homography_from_corners(src_points)
    mapped = project_points(feet_px, H)
    mapped[:, 0] = np.clip(mapped[:, 0], 0, FIELD_LENGTH)
    mapped[:, 1] = np.clip(mapped[:, 1], 0, FIELD_WIDTH)
    return mapped


def depth_aware_feet(
    detections: Sequence[Any],
    depth: np.ndarray | None,
    *,
    crop_y0: float = 0.0,
) -> list[tuple[float, float]]:
    """Convert detection bboxes into refined (fx, fy) full-frame feet."""
    feet: list[tuple[float, float]] = []
    for det in detections:
        bbox = getattr(det, "bbox", None) or det.get("bbox")  # type: ignore[union-attr]
        feet.append(refine_foot_point(list(bbox), depth, crop_y0=crop_y0))
    return feet


def warp_crop_to_bev(
    crop_bgr: np.ndarray,
    src_norm: Sequence[Sequence[float]] | None = None,
    *,
    out_w: int = 640,
    out_h: int = 320,
) -> tuple[np.ndarray, np.ndarray]:
    """Perspective-warp a field crop into a top-down rectangle (debug / overlay)."""
    import cv2

    h, w = crop_bgr.shape[:2]
    norm = list(src_norm or DEFAULT_SRC_NORM)
    src = np.float32([[nx * w, ny * h] for nx, ny in norm])
    dst = np.float32([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(crop_bgr, H, (out_w, out_h))
    return warped, H


def _tilt_adjusted_norm(
    base: Sequence[Sequence[float]],
    tilt: float,
    pitch_deg: float,
) -> list[tuple[float, float]]:
    """Widen/narrow the default trapezoid from estimated camera pitch."""
    pts = [(float(x), float(y)) for x, y in base]
    # Stronger overhead (higher pitch / tilt) → top edge wider, bottom less extreme.
    t = float(np.clip(tilt, 0.0, 1.0))
    pitch_t = float(np.clip((pitch_deg - 20.0) / 40.0, 0.0, 1.0))
    mix = 0.55 * t + 0.45 * pitch_t

    top_inset = 0.18 - 0.10 * mix  # default 0.18 → toward 0.08
    bot_inset = 0.03 + 0.04 * (1.0 - mix)  # default 0.03 → slightly more inset when flat
    top_y = 0.08 + 0.04 * (1.0 - mix)
    bot_y = 0.92 - 0.03 * mix

    top_inset = float(np.clip(top_inset, 0.06, 0.28))
    bot_inset = float(np.clip(bot_inset, 0.01, 0.12))
    top_y = float(np.clip(top_y, 0.04, 0.18))
    bot_y = float(np.clip(bot_y, 0.82, 0.96))

    return [
        (top_inset, top_y),
        (1.0 - top_inset, top_y),
        (1.0 - bot_inset, bot_y),
        (bot_inset, bot_y),
    ]


def _nudge_from_depth_edges(
    pts: list[list[float]],
    depth: np.ndarray,
    y0: int,
) -> list[list[float]]:
    """Slightly skew bottom corners if left/right depth means disagree."""
    if depth is None or depth.size < 16 or len(pts) != 4:
        return pts
    h, w = depth.shape[:2]
    mid = h // 2
    left = float(np.mean(depth[mid:, : w // 3]))
    right = float(np.mean(depth[mid:, 2 * w // 3 :]))
    delta = right - left
    # Positive delta ⇒ right side closer ⇒ nudge bottom-right up a touch.
    shift = float(np.clip(delta * 18.0, -12.0, 12.0))
    out = [list(p) for p in pts]
    out[2][1] = float(out[2][1] - shift)  # bottom-right
    out[3][1] = float(out[3][1] + shift)  # bottom-left
    return out
