"""Detect multi-angle FRC broadcast layouts and prefer the top camera.

Many event VODs stack two feeds vertically: a correctly oriented high/wide
field camera on top, and a sideline/close-up angle below. Tracking should use
the top pane so field geometry matches blue-left / red-right orientation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraLayout:
    """Normalized vertical crop that isolates the preferred camera."""

    mode: str  # "single" | "stacked_top" | "stacked_bottom" | "manual"
    crop_top: float
    crop_bottom: float
    confidence: float
    detail: str
    split_y: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def analyze_frame(frame: np.ndarray) -> CameraLayout:
    """Guess whether a broadcast frame is a vertical dual-camera stack.

    Heuristic: look for a strong horizontal band of low-motion / dark gutter
    near mid-frame (letterbox / divider between stacked angles). Prefer the
    upper pane when found — that feed is typically the correctly oriented
    field overview.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        return CameraLayout("single", 0.10, 0.65, 0.0, "Empty frame.")

    import cv2

    h, w = frame.shape[:2]
    if h < 120 or w < 160:
        return CameraLayout("single", 0.10, 0.65, 0.2, "Frame too small to classify.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    # Row energy: mean absolute horizontal gradient — dividers are flat.
    sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_energy = np.mean(np.abs(sobel), axis=1)
    row_mean = np.mean(gray.astype(np.float32), axis=1)

    # Search for a low-energy gutter in the middle third of the frame.
    y0 = int(h * 0.28)
    y1 = int(h * 0.72)
    window = max(3, h // 80)
    scores: list[tuple[float, int]] = []
    for y in range(y0, y1):
        band = slice(max(0, y - window), min(h, y + window + 1))
        energy = float(np.mean(row_energy[band]))
        brightness = float(np.mean(row_mean[band]))
        # Prefer dark-ish gutters with low texture.
        score = energy + 0.15 * brightness
        scores.append((score, y))
    scores.sort(key=lambda item: item[0])
    best_score, split_y = scores[0]
    median_score = float(np.median([s for s, _ in scores]))
    if median_score <= 1e-3:
        return CameraLayout("single", 0.10, 0.65, 0.2, "Could not measure frame energy.")

    contrast = (median_score - best_score) / median_score
    split_norm = split_y / float(h)

    # Upper and lower panes should each look "busy" relative to the gutter.
    upper = float(np.mean(row_energy[int(h * 0.05) : max(split_y - window, 1)]))
    lower = float(np.mean(row_energy[min(split_y + window, h - 1) : int(h * 0.95)]))
    panes_ok = upper > best_score * 1.35 and lower > best_score * 1.35

    if contrast >= 0.22 and panes_ok and 0.32 <= split_norm <= 0.68:
        # Top camera: from a small headroom crop down to just above the split.
        crop_top = 0.02
        crop_bottom = min(max(split_norm - 0.01, 0.35), 0.70)
        if crop_bottom - crop_top < 0.25:
            crop_bottom = min(crop_top + 0.40, 0.70)
        return CameraLayout(
            mode="stacked_top",
            crop_top=round(crop_top, 3),
            crop_bottom=round(crop_bottom, 3),
            confidence=round(min(0.95, 0.45 + contrast), 3),
            detail=(
                "Detected a stacked dual-camera layout. Using the top camera "
                "(correct field orientation) and ignoring the lower angle."
            ),
            split_y=round(split_norm, 3),
        )

    # Single wide broadcast — keep the classic scorebug-aware crop.
    return CameraLayout(
        mode="single",
        crop_top=0.10,
        crop_bottom=0.65,
        confidence=round(max(0.25, 1.0 - contrast), 3),
        detail="Single-camera (or unclear) layout — using the default field crop.",
        split_y=round(split_norm, 3) if contrast > 0.12 else None,
    )


def analyze_video(video_path: Path | str, sample_times_s: list[float] | None = None) -> CameraLayout:
    """Sample a few mid-match frames and vote on the camera layout."""
    import cv2

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return CameraLayout("single", 0.10, 0.65, 0.0, f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = (total / fps) if total > 0 else 90.0
    times = sample_times_s or [duration * x for x in (0.18, 0.35, 0.55, 0.72)]

    votes: list[CameraLayout] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t * fps), 0))
        ok, frame = cap.read()
        if ok and frame is not None:
            votes.append(analyze_frame(frame))
    cap.release()

    if not votes:
        return CameraLayout("single", 0.10, 0.65, 0.0, "No frames readable for camera layout.")

    stacked = [v for v in votes if v.mode == "stacked_top"]
    if len(stacked) >= max(1, (len(votes) + 1) // 2):
        crop_top = float(np.median([v.crop_top for v in stacked]))
        crop_bottom = float(np.median([v.crop_bottom for v in stacked]))
        conf = float(np.mean([v.confidence for v in stacked]))
        split = float(np.median([v.split_y for v in stacked if v.split_y is not None] or [0.5]))
        return CameraLayout(
            mode="stacked_top",
            crop_top=round(crop_top, 3),
            crop_bottom=round(crop_bottom, 3),
            confidence=round(conf, 3),
            detail=stacked[0].detail,
            split_y=round(split, 3),
        )
    return votes[len(votes) // 2]


def apply_layout(
    layout: CameraLayout,
    *,
    user_crop_top: float | None = None,
    user_crop_bottom: float | None = None,
    auto: bool = True,
) -> tuple[float, float, CameraLayout]:
    """Return crop bounds, honoring manual crops unless auto multi-cam wins."""
    if not auto:
        top = 0.10 if user_crop_top is None else float(user_crop_top)
        bottom = 0.65 if user_crop_bottom is None else float(user_crop_bottom)
        manual = CameraLayout("manual", top, bottom, 1.0, "Using manual crop bounds.")
        return top, bottom, manual

    # If the user already widened/narrowed away from defaults, respect them.
    defaults = (abs((user_crop_top or 0.10) - 0.10) < 0.011) and (
        abs((user_crop_bottom or 0.65) - 0.65) < 0.011
    )
    if user_crop_top is not None and user_crop_bottom is not None and not defaults:
        manual = CameraLayout(
            "manual",
            float(user_crop_top),
            float(user_crop_bottom),
            1.0,
            "Using manual crop bounds (multi-camera auto-detect skipped).",
        )
        return float(user_crop_top), float(user_crop_bottom), manual

    if layout.mode == "stacked_top" and layout.confidence >= 0.5:
        return layout.crop_top, layout.crop_bottom, layout

    top = 0.10 if user_crop_top is None else float(user_crop_top)
    bottom = 0.65 if user_crop_bottom is None else float(user_crop_bottom)
    return top, bottom, layout
