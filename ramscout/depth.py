"""Monocular depth for camera-angle-aware field projection.

Prefers Depth Anything V2 (Hugging Face Transformers) when torch + transformers
are installed. Falls back to a classical relative-depth estimate so desktop
builds without PyTorch still get camera-tilt cues for BEV.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Depth-Anything-V2-Small is Apache-2.0 and light enough for optional use.
_HF_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_PIPE: Any = None
_PIPE_FAILED = False


@dataclass
class DepthResult:
    """Relative depth map + derived camera-angle cues."""

    depth: np.ndarray  # HxW float32, larger ≈ closer (V2 convention)
    source: str  # "depth_anything_v2" | "classical"
    pitch_deg: float
    tilt_strength: float  # 0..1 how strongly the camera looks down on the field
    detail: str

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("depth", None)
        data["shape"] = list(self.depth.shape)
        data["depth_min"] = float(np.min(self.depth)) if self.depth.size else 0.0
        data["depth_max"] = float(np.max(self.depth)) if self.depth.size else 0.0
        return data


def depth_available() -> bool:
    """True when Depth Anything V2 can be loaded via transformers."""
    if _PIPE_FAILED:
        return False
    try:
        import torch  # noqa: F401
        from transformers import pipeline  # noqa: F401

        return True
    except Exception:
        return False


def estimate_depth(frame_bgr: np.ndarray, *, prefer_neural: bool = True) -> DepthResult:
    """Estimate a relative depth map for a BGR frame."""
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        empty = np.zeros((1, 1), dtype=np.float32)
        return DepthResult(empty, "classical", 0.0, 0.0, "Empty frame.")

    if prefer_neural:
        neural = _try_depth_anything(frame_bgr)
        if neural is not None:
            pitch, tilt = _pitch_from_depth(neural)
            return DepthResult(
                depth=neural,
                source="depth_anything_v2",
                pitch_deg=round(pitch, 2),
                tilt_strength=round(tilt, 3),
                detail="Depth Anything V2 relative depth (camera-angle cues for BEV).",
            )

    classical = _classical_depth(frame_bgr)
    pitch, tilt = _pitch_from_depth(classical)
    return DepthResult(
        depth=classical,
        source="classical",
        pitch_deg=round(pitch, 2),
        tilt_strength=round(tilt, 3),
        detail="Classical relative depth (no Depth Anything weights) — tilt from field gradient.",
    )


def sample_depth(depth: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample; returns 0 outside bounds."""
    if depth is None or depth.size == 0:
        return 0.0
    h, w = depth.shape[:2]
    if w < 2 or h < 2:
        return float(depth.flat[0]) if depth.size else 0.0
    xf = float(np.clip(x, 0, w - 1.001))
    yf = float(np.clip(y, 0, h - 1.001))
    x0, y0 = int(xf), int(yf)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = xf - x0, yf - y0
    v00 = float(depth[y0, x0])
    v10 = float(depth[y0, x1])
    v01 = float(depth[y1, x0])
    v11 = float(depth[y1, x1])
    return (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11)


def refine_foot_point(
    bbox: list[float] | tuple[float, float, float, float],
    depth: np.ndarray | None,
    *,
    crop_y0: float = 0.0,
) -> tuple[float, float]:
    """Pick a ground-contact foot using depth inside the detection box.

    Uses the lower band of the bbox and prefers the deepest (closest-to-camera /
    lowest-on-field) contact along the bottom edge — better than raw (cx, y2)
    when the camera is angled.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    fx = (x1 + x2) * 0.5
    fy = y2 + crop_y0
    if depth is None or depth.size == 0:
        return fx, fy

    h, w = depth.shape[:2]
    bx1 = int(np.clip(x1, 0, w - 1))
    bx2 = int(np.clip(x2, 0, w - 1))
    by1 = int(np.clip(y1, 0, h - 1))
    by2 = int(np.clip(y2, 0, h - 1))
    if bx2 <= bx1 or by2 <= by1:
        return fx, fy

    band_top = max(by1, by2 - max(2, (by2 - by1) // 3))
    band = depth[band_top : by2 + 1, bx1 : bx2 + 1]
    if band.size == 0:
        return fx, fy

    # Larger depth ⇒ closer to camera in V2; for top-down angled cams the
    # near-ground contact sits toward the bottom of the box with high depth.
    flat = band.reshape(-1)
    idx = int(np.argmax(flat))
    local_y, local_x = divmod(idx, band.shape[1])
    foot_x = bx1 + local_x
    foot_y = band_top + local_y
    # Blend toward geometric bottom-center so noisy depth does not yank feet.
    blend = 0.55
    fx = (1 - blend) * fx + blend * float(foot_x)
    fy = (1 - blend) * fy + blend * float(foot_y + crop_y0)
    return fx, fy


def _try_depth_anything(frame_bgr: np.ndarray) -> np.ndarray | None:
    global _PIPE, _PIPE_FAILED
    if _PIPE_FAILED:
        return None
    try:
        import cv2
        from PIL import Image

        pipe = _get_pipe()
        if pipe is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Cap long edge so optional inference stays interactive.
        h, w = rgb.shape[:2]
        scale = min(1.0, 768.0 / max(h, w))
        if scale < 0.999:
            rgb_small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb
        image = Image.fromarray(rgb_small)
        out = pipe(image)
        depth_img = out["depth"] if isinstance(out, dict) else out
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        # Normalize to a stable relative range.
        dmin, dmax = float(depth.min()), float(depth.max())
        if dmax - dmin > 1e-6:
            depth = (depth - dmin) / (dmax - dmin)
        return depth.astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        log.info("Depth Anything V2 unavailable (%s) — using classical depth.", exc)
        _PIPE_FAILED = True
        _PIPE = None
        return None


def _get_pipe() -> Any:
    global _PIPE, _PIPE_FAILED
    if _PIPE is not None:
        return _PIPE
    if _PIPE_FAILED:
        return None
    try:
        from transformers import pipeline

        device = _pick_device()
        _PIPE = pipeline(
            task="depth-estimation",
            model=_HF_MODEL,
            device=device,
        )
        return _PIPE
    except Exception as exc:  # noqa: BLE001
        log.info("Could not load Depth Anything V2 pipeline: %s", exc)
        _PIPE_FAILED = True
        return None


def _pick_device() -> int | str:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return -1


def _classical_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Cheap relative depth: lower image rows + soft focus on carpet texture."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    h, w = gray.shape[:2]
    yy = np.linspace(0.15, 1.0, h, dtype=np.float32).reshape(-1, 1)
    row = np.repeat(yy, w, axis=1)
    # Texture: flatter carpet farther up often has lower local variance.
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    local = cv2.GaussianBlur((gray.astype(np.float32) - blur) ** 2, (0, 0), 5)
    local = local / (float(local.max()) + 1e-6)
    depth = 0.72 * row + 0.28 * (1.0 - local)
    depth = depth.astype(np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin > 1e-6:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


def _pitch_from_depth(depth: np.ndarray) -> tuple[float, float]:
    """Estimate camera pitch (degrees looking down) from vertical depth gradient."""
    if depth is None or depth.size < 16:
        return 35.0, 0.45
    h, w = depth.shape[:2]
    # Center strip avoids scoreboard / sideline clutter.
    x0, x1 = int(w * 0.25), int(w * 0.75)
    strip = depth[:, x0:x1]
    row_mean = np.mean(strip, axis=1)
    # Fit depth vs normalized row index.
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    slope = float(np.polyfit(ys, row_mean.astype(np.float64), 1)[0])
    # Positive slope ⇒ bottom closer ⇒ typical overhead/angled broadcast.
    tilt = float(np.clip(abs(slope) / 1.2, 0.0, 1.0))
    # Map to a plausible FRC broadcast pitch range (~20–55° from horizontal).
    pitch = 22.0 + tilt * 33.0
    if slope < 0:
        pitch = max(15.0, 50.0 - pitch * 0.35)
    return pitch, tilt
