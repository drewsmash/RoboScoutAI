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
    # Field-line refinement (ramscout.fieldlines): carpet quad + orientation.
    field_quad: dict[str, Any] | None = None
    orientation: dict[str, Any] | None = None
    method: str = "depth_trapezoid"  # depth_trapezoid | field_quad | field_quad+depth

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def calibrate_bev(
    frame_bgr: np.ndarray,
    *,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    crop_left: float = 0.0,
    crop_right: float = 1.0,
    prefer_neural: bool = True,
    base_norm: Sequence[Sequence[float]] | None = None,
    field_lines: bool = True,
    extra_frames: Sequence[np.ndarray] | None = None,
    motion_mask: np.ndarray | None = None,
    min_quad_confidence: float = 0.55,
) -> tuple[BevCalibration, DepthResult]:
    """Build BEV source corners from a calibration frame + depth estimate.

    Steps: (1) depth → camera pitch → tilt-adjusted trapezoid; (2) when
    ``field_lines`` is on, the carpet boundary quad from
    :mod:`ramscout.fieldlines` replaces the trapezoid if it is confident and
    geometrically consistent with the depth pitch, and the alliance
    orientation cue mirrors the corners so ``x = 0`` is the blue wall.
    ``extra_frames`` (same size as ``frame_bgr``) make the carpet median robust
    to robots; ``motion_mask`` is pane-sized (non-zero where robots drove).
    """
    h, w = frame_bgr.shape[:2]
    y0 = int(h * crop_top)
    y1 = int(h * crop_bottom)
    y0 = max(0, min(y0, h - 2))
    y1 = max(y0 + 1, min(y1, h))
    x0 = int(np.clip(round(w * crop_left), 0, w - 2))
    x1 = int(np.clip(round(w * crop_right), x0 + 2, w))
    crop = frame_bgr[y0:y1, x0:x1]
    depth_result = estimate_depth(crop, prefer_neural=prefer_neural)

    norm = _tilt_adjusted_norm(base_norm or DEFAULT_SRC_NORM, depth_result.tilt_strength, depth_result.pitch_deg)
    crop_h = float(max(y1 - y0, 1))
    crop_w = float(max(x1 - x0, 1))
    pts: list[list[float]] = []
    for nx, ny in norm:
        pts.append([float(x0) + float(nx) * crop_w, float(y0) + float(ny) * crop_h])

    # Optional: nudge left/right bottom corners using depth symmetry so a
    # slightly skewed camera still lands blue-left / red-right cleanly.
    pts = _nudge_from_depth_edges(pts, depth_result.depth, y0)

    detail = (
        f"BEV from {depth_result.source}: pitch≈{depth_result.pitch_deg:.0f}°, "
        f"tilt={depth_result.tilt_strength:.2f}. {depth_result.detail}"
    )
    quad_info: dict[str, Any] | None = None
    orient_info: dict[str, Any] | None = None
    method = "depth_trapezoid"
    if field_lines:
        try:
            from ramscout.fieldlines import alliance_orientation, detect_field_quad, orient_corners

            pane_frames = [crop]
            for extra in extra_frames or []:
                if extra is not None and extra.shape[:2] == frame_bgr.shape[:2]:
                    pane_frames.append(extra[y0:y1, x0:x1])
            quad = detect_field_quad(pane_frames, motion_mask=motion_mask)
            cue = alliance_orientation(pane_frames, quad)
            orient_info = cue.as_dict()
            if quad is not None:
                quad_info = quad.as_dict()
                quad_pts = [[float(x0) + c[0], float(y0) + c[1]] for c in quad.corners]
                if quad.confidence >= min_quad_confidence and _quad_agrees_with_depth(quad.corners, crop_w, crop_h, depth_result.pitch_deg):
                    pts = quad_pts
                    method = "field_quad"
                    detail += f" Homography from carpet boundary (conf {quad.confidence:.2f})."
                elif quad.confidence >= 0.4:
                    # Blend: keep the depth trapezoid's shape but centre/scale
                    # it on the carpet quad's extent.
                    pts = _blend_quads(pts, quad_pts, 0.5)
                    method = "field_quad+depth"
                    detail += f" Depth trapezoid nudged toward carpet boundary (conf {quad.confidence:.2f})."
            if cue.confidence >= 0.25:
                pts = orient_corners(pts, cue.blue_left)
                if not cue.blue_left:
                    detail += " Blue alliance wall on the right → homography mirrored."
        except Exception as exc:  # noqa: BLE001
            detail += f" Field-line refinement skipped ({exc})."

    cal = BevCalibration(
        src_points=[[round(p[0], 2), round(p[1], 2)] for p in pts],
        pitch_deg=depth_result.pitch_deg,
        tilt_strength=depth_result.tilt_strength,
        depth_source=depth_result.source,
        detail=detail,
        used_depth=True,
        field_quad=quad_info,
        orientation=orient_info,
        method=method,
    )
    return cal, depth_result


def _quad_agrees_with_depth(corners: Sequence[Sequence[float]], crop_w: float, crop_h: float, pitch_deg: float) -> bool:
    """A carpet quad is trusted only when its perspective matches the pitch."""
    tl, tr, br, bl = corners
    top_w = abs(tr[0] - tl[0])
    bot_w = abs(br[0] - bl[0])
    if top_w < 0.25 * crop_w or bot_w < 0.25 * crop_w:
        return False
    persp = bot_w / max(top_w, 1.0)
    # Steep camera → near-orthographic (ratio ~1); flat camera → strong taper.
    expected = 1.0 + max(0.0, (60.0 - float(pitch_deg)) / 60.0) * 1.2
    return abs(persp - expected) <= 0.9 and persp >= 0.8


def _blend_quads(a: list[list[float]], b: list[list[float]], t: float) -> list[list[float]]:
    return [[(1 - t) * pa[0] + t * pb[0], (1 - t) * pa[1] + t * pb[1]] for pa, pb in zip(a, b)]


def project_detections_bev(
    feet_px: Sequence[Sequence[float]],
    src_points: Sequence[Sequence[float]],
    *,
    clamp: bool = False,
) -> np.ndarray:
    """Map pixel foot points through the BEV/homography onto field inches.

    By default returns raw projected inches (may be outside the carpet). Callers
    that need drawable field coords should run ``annotate_field_sample`` —
    clamping to the wall is opt-in and discouraged for path rendering.
    """
    from ramscout.geometry import annotate_field_sample

    H = homography_from_corners(src_points)
    mapped = project_points(feet_px, H)
    if clamp:
        # Legacy path for callers that still want a wall contact estimate.
        mapped = mapped.copy()
        mapped[:, 0] = np.clip(mapped[:, 0], 0, FIELD_LENGTH)
        mapped[:, 1] = np.clip(mapped[:, 1], 0, FIELD_WIDTH)
        return mapped
    # Replace far-outside / non-finite rows with NaN so downstream code cannot
    # treat them as field-edge positions.
    out = mapped.astype(np.float64, copy=True)
    for i, (x, y) in enumerate(out):
        probe = annotate_field_sample({}, float(x), float(y))
        if not probe.get("field_valid"):
            out[i, 0] = np.nan
            out[i, 1] = np.nan
    return out


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
