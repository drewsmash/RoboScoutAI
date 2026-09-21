"""Overview crop lock and coordinate transforms.

Top-overview-only tracking crops the broadcast *before* detection, depth,
optical flow, and appearance. Normalized crop coordinates are stored relative
to the original video frame so the UI can re-draw handles after seek/resize.

A crop rectangle is **not** a camera identity: cuts inside the crop still
create gaps and require recalibration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class OverviewCrop:
    """Normalized crop of the top widescreen overview (fractions of full frame).

    Coordinates are inclusive-exclusive in pixel space after denormalization:
    ``[x0, x1) × [y0, y1)``.
    """

    x0: float = 0.0
    y0: float = 0.0
    x1: float = 1.0
    y1: float = 0.55
    locked: bool = False
    source: str = "auto"  # auto | user | interval
    t0: float | None = None  # optional time-interval override
    t1: float | None = None
    confidence: float = 0.0
    detail: str = ""

    def clamp(self) -> OverviewCrop:
        x0 = float(np.clip(self.x0, 0.0, 0.98))
        x1 = float(np.clip(self.x1, x0 + 0.02, 1.0))
        y0 = float(np.clip(self.y0, 0.0, 0.95))
        y1 = float(np.clip(self.y1, y0 + 0.05, 1.0))
        return OverviewCrop(
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            locked=bool(self.locked),
            source=str(self.source or "auto"),
            t0=self.t0,
            t1=self.t1,
            confidence=float(self.confidence or 0.0),
            detail=str(self.detail or ""),
        )

    @property
    def box(self) -> list[float]:
        c = self.clamp()
        return [c.x0, c.y0, c.x1, c.y1]

    def as_dict(self) -> dict[str, Any]:
        c = self.clamp()
        data = asdict(c)
        data["box"] = c.box
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> OverviewCrop:
        if not raw:
            return cls().clamp()
        return cls(
            x0=float(raw.get("x0", raw.get("crop_left", 0.0)) or 0.0),
            y0=float(raw.get("y0", raw.get("crop_top", 0.0)) or 0.0),
            x1=float(raw.get("x1", raw.get("crop_right", 1.0)) or 1.0),
            y1=float(raw.get("y1", raw.get("crop_bottom", 0.55)) or 0.55),
            locked=bool(raw.get("locked", False)),
            source=str(raw.get("source") or "auto"),
            t0=raw.get("t0"),
            t1=raw.get("t1"),
            confidence=float(raw.get("confidence") or 0.0),
            detail=str(raw.get("detail") or ""),
        ).clamp()


@dataclass
class FrameCoords:
    """Pixel sizes for each transform stage."""

    video_w: int
    video_h: int
    crop_w: int
    crop_h: int
    detector_w: int = 0
    detector_h: int = 0
    letterbox_pad: tuple[int, int, int, int] = (0, 0, 0, 0)  # L,T,R,B in detector space

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def propose_top_overview(
    layout_panes: list[dict[str, Any]] | None = None,
    *,
    fallback_bottom: float = 0.55,
) -> OverviewCrop:
    """Propose the top widescreen overview crop from a layout decomposition."""
    panes = list(layout_panes or [])
    overview = next((p for p in panes if str(p.get("role") or "") == "overview"), None)
    if overview is None and panes:
        # Tallest top-most pane as a fallback.
        topish = sorted(
            panes,
            key=lambda p: (float(p.get("crop_top") or 1.0), -(float(p.get("crop_bottom") or 0) - float(p.get("crop_top") or 0))),
        )
        overview = topish[0]
    if overview is not None:
        return OverviewCrop(
            x0=float(overview.get("crop_left") or 0.0),
            y0=float(overview.get("crop_top") or 0.0),
            x1=float(overview.get("crop_right") or 1.0),
            y1=float(overview.get("crop_bottom") or fallback_bottom),
            locked=False,
            source="auto",
            confidence=float(overview.get("confidence") or 0.6),
            detail=str(overview.get("purpose") or overview.get("role") or "overview"),
        ).clamp()
    return OverviewCrop(0.0, 0.0, 1.0, fallback_bottom, source="auto", confidence=0.3, detail="default top half").clamp()


def crop_pixels(crop: OverviewCrop, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
    """Return integer ``(x0, y0, x1, y1)`` pixel bounds in original video space."""
    c = crop.clamp()
    x0 = int(np.clip(round(c.x0 * frame_w), 0, max(frame_w - 2, 0)))
    x1 = int(np.clip(round(c.x1 * frame_w), x0 + 2, frame_w))
    y0 = int(np.clip(round(c.y0 * frame_h), 0, max(frame_h - 2, 0)))
    y1 = int(np.clip(round(c.y1 * frame_h), y0 + 2, frame_h))
    return x0, y0, x1, y1


def slice_overview(frame: np.ndarray, crop: OverviewCrop) -> tuple[np.ndarray, FrameCoords]:
    """Crop ``frame`` to the overview lock and return (crop, coords)."""
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = crop_pixels(crop, w, h)
    sliced = frame[y0:y1, x0:x1]
    coords = FrameCoords(video_w=w, video_h=h, crop_w=x1 - x0, crop_h=y1 - y0)
    return sliced, coords


def video_to_crop(x: float, y: float, crop: OverviewCrop, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Map original-video pixels → crop-local pixels."""
    x0, y0, x1, y1 = crop_pixels(crop, frame_w, frame_h)
    return float(x - x0), float(y - y0)


def crop_to_video(x: float, y: float, crop: OverviewCrop, frame_w: int, frame_h: int) -> tuple[float, float]:
    """Map crop-local pixels → original-video pixels."""
    x0, y0, _, _ = crop_pixels(crop, frame_w, frame_h)
    return float(x + x0), float(y + y0)


def bbox_crop_to_video(bbox: list[float], crop: OverviewCrop, frame_w: int, frame_h: int) -> list[float]:
    """Lift a crop-local ``[x1,y1,x2,y2]`` bbox into original video pixels."""
    if len(bbox) < 4:
        return list(bbox)
    x1, y1 = crop_to_video(float(bbox[0]), float(bbox[1]), crop, frame_w, frame_h)
    x2, y2 = crop_to_video(float(bbox[2]), float(bbox[3]), crop, frame_w, frame_h)
    return [x1, y1, x2, y2]


def letterbox_to_crop(
    x: float,
    y: float,
    *,
    detector_w: int,
    detector_h: int,
    crop_w: int,
    crop_h: int,
    pad: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> tuple[float, float]:
    """Map letterboxed detector pixels back into crop-local pixels."""
    pad_l, pad_t, pad_r, pad_b = pad
    inner_w = max(detector_w - pad_l - pad_r, 1)
    inner_h = max(detector_h - pad_t - pad_b, 1)
    sx = crop_w / inner_w
    sy = crop_h / inner_h
    return (float(x) - pad_l) * sx, (float(y) - pad_t) * sy


@dataclass
class CropLockState:
    """Job-level crop lock with optional per-interval overrides."""

    active: OverviewCrop = field(default_factory=OverviewCrop)
    intervals: list[OverviewCrop] = field(default_factory=list)
    mode: str = "top_overview_only"  # top_overview_only | legacy_multiview

    def crop_at(self, t: float) -> OverviewCrop:
        for iv in self.intervals:
            if iv.t0 is not None and iv.t1 is not None and float(iv.t0) <= t < float(iv.t1):
                return iv.clamp()
        return self.active.clamp()

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "active": self.active.as_dict(),
            "intervals": [c.as_dict() for c in self.intervals],
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> CropLockState:
        if not raw:
            return cls()
        return cls(
            active=OverviewCrop.from_dict(raw.get("active") if isinstance(raw.get("active"), dict) else raw),
            intervals=[OverviewCrop.from_dict(i) for i in (raw.get("intervals") or []) if isinstance(i, dict)],
            mode=str(raw.get("mode") or "top_overview_only"),
        )
