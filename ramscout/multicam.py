"""Detect multi-angle FRC broadcast layouts and section them into roles.

Many event VODs stack feeds:
- Top wide / high camera → overview for general movement (top-down map)
- Bottom pane → often a closer sideline; when wide enough we split it into
  blue-side (left) and red-side (right) for scoring / climb reasoning

Older single-camera broadcasts keep the classic scorebug-aware crop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class CameraPane:
    """One camera crop with a scouting role."""

    role: str  # overview | overview_alt | blue_side | red_side | sideline | graphics | other
    crop_top: float
    crop_bottom: float
    crop_left: float = 0.0
    crop_right: float = 1.0
    purpose: str = ""
    alliance_bias: str | None = None  # blue | red | None
    confidence: float = 1.0
    features: dict[str, float] = field(default_factory=dict)
    scorebug: list[float] | None = None  # normalized box inside this pane

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["confidence"] = round(float(self.confidence), 3)
        return data

    @property
    def box(self) -> list[float]:
        return [float(self.crop_left), float(self.crop_top), float(self.crop_right), float(self.crop_bottom)]

    def slice_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        y0 = int(np.clip(self.crop_top, 0, 1) * h)
        y1 = int(np.clip(self.crop_bottom, 0, 1) * h)
        x0 = int(np.clip(self.crop_left, 0, 1) * w)
        x1 = int(np.clip(self.crop_right, 0, 1) * w)
        y0, y1 = max(0, min(y0, h - 1)), max(y0 + 1, min(y1, h))
        x0, x1 = max(0, min(x0, w - 1)), max(x0 + 1, min(x1, w))
        return frame[y0:y1, x0:x1]


@dataclass
class CameraLayout:
    """Normalized crop layout for preferred + side camera panes."""

    mode: str  # single | stacked_top | stacked_sides | side_by_side | grid | manual
    crop_top: float
    crop_bottom: float
    confidence: float
    detail: str
    split_y: float | None = None
    split_x: float | None = None
    panes: list[CameraPane] = field(default_factory=list)
    # Signal-decomposition extras (see ramscout.layout): per-segment timeline,
    # detected scorebug box, active picture area, analysis method.
    timeline: list[dict[str, Any]] = field(default_factory=list)
    scorebug: list[float] | None = None
    letterbox: list[float] | None = None
    method: str = "heuristic"
    crop_left: float = 0.0
    crop_right: float = 1.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["panes"] = [p.as_dict() if hasattr(p, "as_dict") else p for p in self.panes]
        return data

    def overview_pane(self) -> CameraPane | None:
        for pane in self.panes:
            if pane.role == "overview":
                return pane
        return None

    def side_panes(self) -> list[CameraPane]:
        return [p for p in self.panes if p.role in {"blue_side", "red_side", "sideline"}]

    def overview_box(self) -> list[float]:
        pane = self.overview_pane()
        if pane is not None:
            return pane.box
        return [0.0, float(self.crop_top), 1.0, float(self.crop_bottom)]


def analyze_frame(frame: np.ndarray) -> CameraLayout:
    """Guess whether a broadcast frame is a vertical dual-camera stack.

    Heuristic: look for a strong horizontal band of low-motion / dark gutter
    near mid-frame (letterbox / divider between stacked angles). Prefer the
    upper pane when found — that feed is typically the correctly oriented
    field overview. When stacked, also try to split the lower pane left/right
    for blue vs red side cameras.
    """
    if frame is None or getattr(frame, "size", 0) == 0:
        layout = CameraLayout("single", 0.10, 0.65, 0.0, "Empty frame.")
        layout.panes = [_single_overview(0.10, 0.65)]
        return layout

    import cv2

    h, w = frame.shape[:2]
    if h < 120 or w < 160:
        layout = CameraLayout("single", 0.10, 0.65, 0.2, "Frame too small to classify.")
        layout.panes = [_single_overview(0.10, 0.65)]
        return layout

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
        layout = CameraLayout("single", 0.10, 0.65, 0.2, "Could not measure frame energy.")
        layout.panes = [_single_overview(0.10, 0.65)]
        return layout

    contrast = (median_score - best_score) / median_score
    split_norm = split_y / float(h)

    # Upper and lower panes should each look "busy" relative to the gutter.
    upper = float(np.mean(row_energy[int(h * 0.05) : max(split_y - window, 1)]))
    lower = float(np.mean(row_energy[min(split_y + window, h - 1) : int(h * 0.95)]))
    panes_ok = upper > best_score * 1.35 and lower > best_score * 1.35

    if contrast >= 0.22 and panes_ok and 0.32 <= split_norm <= 0.68:
        crop_top = 0.02
        crop_bottom = min(max(split_norm - 0.01, 0.35), 0.70)
        if crop_bottom - crop_top < 0.25:
            crop_bottom = min(crop_top + 0.40, 0.70)

        lower_top = min(max(split_norm + 0.01, crop_bottom), 0.95)
        lower_bottom = 0.98
        split_x = _vertical_split(gray, lower_top, lower_bottom)

        panes = [
            CameraPane(
                role="overview",
                crop_top=round(crop_top, 3),
                crop_bottom=round(crop_bottom, 3),
                purpose="Top wide-angle — general movement for the top-down map",
            )
        ]
        mode = "stacked_top"
        detail = (
            "Detected a stacked dual-camera layout. Top wide-angle drives movement "
            "tracking; lower angle feeds scoring / climb reasoning."
        )

        if split_x is not None and 0.35 <= split_x <= 0.65:
            panes.extend(
                [
                    CameraPane(
                        role="blue_side",
                        crop_top=round(lower_top, 3),
                        crop_bottom=round(lower_bottom, 3),
                        crop_left=0.0,
                        crop_right=round(split_x, 3),
                        purpose="Bottom-left side camera — blue scoring / climb",
                        alliance_bias="blue",
                    ),
                    CameraPane(
                        role="red_side",
                        crop_top=round(lower_top, 3),
                        crop_bottom=round(lower_bottom, 3),
                        crop_left=round(split_x, 3),
                        crop_right=1.0,
                        purpose="Bottom-right side camera — red scoring / climb",
                        alliance_bias="red",
                    ),
                ]
            )
            mode = "stacked_sides"
            detail = (
                "Stacked layout with split lower pane: top wide-angle for movement, "
                "bottom-left for blue scoring/climb, bottom-right for red scoring/climb."
            )
        else:
            panes.append(
                CameraPane(
                    role="sideline",
                    crop_top=round(lower_top, 3),
                    crop_bottom=round(lower_bottom, 3),
                    purpose="Lower sideline / close-up — scoring and climb cues",
                )
            )

        return CameraLayout(
            mode=mode,
            crop_top=round(crop_top, 3),
            crop_bottom=round(crop_bottom, 3),
            confidence=round(min(0.95, 0.45 + contrast), 3),
            detail=detail,
            split_y=round(split_norm, 3),
            split_x=round(split_x, 3) if split_x is not None else None,
            panes=panes,
        )

    # Single wide broadcast — keep the classic scorebug-aware crop.
    layout = CameraLayout(
        mode="single",
        crop_top=0.10,
        crop_bottom=0.65,
        confidence=round(max(0.25, 1.0 - contrast), 3),
        detail="Single-camera (or unclear) layout — using the default field crop.",
        split_y=round(split_norm, 3) if contrast > 0.12 else None,
    )
    layout.panes = [_single_overview(0.10, 0.65)]
    return layout


def analyze_video(
    video_path: Path | str,
    sample_times_s: list[float] | None = None,
    *,
    decompose: bool = True,
    on_progress: Any = None,
) -> CameraLayout:
    """Decompose the broadcast into panes over time (see :mod:`ramscout.layout`).

    Falls back to the legacy single-frame gutter heuristic when the temporal
    decomposition fails or the caller passes explicit ``sample_times_s``.
    """
    if decompose and sample_times_s is None:
        try:
            return _analyze_video_decomposed(video_path, on_progress=on_progress)
        except Exception as exc:  # noqa: BLE001
            legacy = _analyze_video_legacy(video_path, sample_times_s)
            legacy.detail = f"{legacy.detail} (temporal decomposition failed: {exc})"
            return legacy
    return _analyze_video_legacy(video_path, sample_times_s)


def layout_from_decomposition(frame_layout: Any, *, timeline: Any = None) -> CameraLayout:
    """Convert a :class:`ramscout.layout.FrameLayout` into a CameraLayout."""
    from ramscout.layout import ROLE_PURPOSE

    panes: list[CameraPane] = []
    for dp in frame_layout.panes:
        panes.append(
            CameraPane(
                role=dp.role,
                crop_top=round(dp.box.y0, 4),
                crop_bottom=round(dp.box.y1, 4),
                crop_left=round(dp.box.x0, 4),
                crop_right=round(dp.box.x1, 4),
                purpose=ROLE_PURPOSE.get(dp.role, ""),
                alliance_bias=dp.alliance_bias,
                confidence=float(dp.confidence),
                features=dict(dp.features),
                scorebug=dp.scorebug,
            )
        )
    ov = next((p for p in panes if p.role == "overview"), None)
    if ov is not None:
        crop_top, crop_bottom = ov.crop_top, ov.crop_bottom
        crop_left, crop_right = ov.crop_left, ov.crop_right
    else:
        crop_top, crop_bottom, crop_left, crop_right = 0.10, 0.65, 0.0, 1.0
    row_cuts = list(frame_layout.row_cuts or [])
    col_cuts = list(frame_layout.col_cuts or [])
    layout = CameraLayout(
        mode=frame_layout.mode,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        confidence=float(frame_layout.confidence),
        detail=frame_layout.detail,
        split_y=round(row_cuts[0], 4) if row_cuts else None,
        split_x=round(col_cuts[0], 4) if col_cuts else None,
        panes=panes,
        timeline=[s.as_dict() for s in timeline.segments] if timeline is not None else [],
        scorebug=frame_layout.scorebug,
        letterbox=frame_layout.letterbox.as_list(),
        method="decomposition",
        crop_left=crop_left,
        crop_right=crop_right,
    )
    return layout


def _analyze_video_decomposed(video_path: Path | str, *, on_progress: Any = None) -> CameraLayout:
    from ramscout.layout import analyze_video_layout

    timeline = analyze_video_layout(video_path, on_progress=on_progress)
    primary = timeline.primary()
    if primary is None:
        raise RuntimeError("no layout segments")
    layout = layout_from_decomposition(primary.layout, timeline=timeline)
    coverage = timeline.overview_coverage()
    switches = max(len(timeline.segments) - 1, 0)
    extra = f" Primary layout {primary.t0:.0f}–{primary.t1:.0f}s; overview visible {coverage:.0%} of the video"
    extra += f"; {switches} layout switch(es)." if switches else "."
    layout.detail = layout.detail + extra
    if layout.overview_pane() is None:
        layout.confidence = min(layout.confidence, 0.3)
    return layout


def _analyze_video_legacy(video_path: Path | str, sample_times_s: list[float] | None = None) -> CameraLayout:
    """Sample a few mid-match frames and vote on the camera layout."""
    import cv2

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        layout = CameraLayout("single", 0.10, 0.65, 0.0, f"Could not open video: {path}")
        layout.panes = [_single_overview(0.10, 0.65)]
        return layout

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
        layout = CameraLayout("single", 0.10, 0.65, 0.0, "No frames readable for camera layout.")
        layout.panes = [_single_overview(0.10, 0.65)]
        return layout

    stacked = [v for v in votes if v.mode in {"stacked_top", "stacked_sides"}]
    if len(stacked) >= max(1, (len(votes) + 1) // 2):
        # Prefer the richest sectional vote when available.
        sides = [v for v in stacked if v.mode == "stacked_sides"]
        best = sides[len(sides) // 2] if sides else stacked[len(stacked) // 2]
        crop_top = float(np.median([v.crop_top for v in stacked]))
        crop_bottom = float(np.median([v.crop_bottom for v in stacked]))
        conf = float(np.mean([v.confidence for v in stacked]))
        split = float(np.median([v.split_y for v in stacked if v.split_y is not None] or [0.5]))
        split_x_vals = [v.split_x for v in stacked if v.split_x is not None]
        split_x = float(np.median(split_x_vals)) if split_x_vals else best.split_x
        return CameraLayout(
            mode=best.mode,
            crop_top=round(crop_top, 3),
            crop_bottom=round(crop_bottom, 3),
            confidence=round(conf, 3),
            detail=best.detail,
            split_y=round(split, 3),
            split_x=round(split_x, 3) if split_x is not None else None,
            panes=list(best.panes),
        )
    mid = votes[len(votes) // 2]
    if not mid.panes:
        mid.panes = [_single_overview(mid.crop_top, mid.crop_bottom)]
    return mid


def apply_layout(
    layout: CameraLayout,
    *,
    user_crop_top: float | None = None,
    user_crop_bottom: float | None = None,
    auto: bool = True,
) -> tuple[float, float, CameraLayout]:
    """Return overview crop bounds, honoring manual crops unless auto multi-cam wins."""
    if not auto:
        top = 0.10 if user_crop_top is None else float(user_crop_top)
        bottom = 0.65 if user_crop_bottom is None else float(user_crop_bottom)
        manual = CameraLayout("manual", top, bottom, 1.0, "Using manual crop bounds.")
        manual.panes = [_single_overview(top, bottom)]
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
        manual.panes = [_single_overview(float(user_crop_top), float(user_crop_bottom))]
        return float(user_crop_top), float(user_crop_bottom), manual

    if layout.mode in {"stacked_top", "stacked_sides"} and layout.confidence >= 0.5:
        if not layout.panes:
            layout.panes = [_single_overview(layout.crop_top, layout.crop_bottom)]
        return layout.crop_top, layout.crop_bottom, layout
    if getattr(layout, "method", "") == "decomposition" and layout.overview_pane() is not None and layout.confidence >= 0.4:
        ov = layout.overview_pane()
        # A full-frame single camera still gets the classic scorebug-aware
        # crop unless a scorebug was located explicitly.
        if layout.mode == "single" and ov.crop_top <= 0.01 and ov.crop_bottom >= 0.99:
            top = 0.10 if user_crop_top is None else float(user_crop_top)
            bottom = 0.65 if user_crop_bottom is None else float(user_crop_bottom)
            bug = layout.scorebug
            if bug:
                # Keep the field; cut only the band the scorebug occupies.
                if bug[1] > 0.5:
                    bottom = min(bottom, max(0.5, bug[1] - 0.01)) if user_crop_bottom is None else bottom
                else:
                    top = max(top, min(0.4, bug[3] + 0.01)) if user_crop_top is None else top
            ov.crop_top, ov.crop_bottom = top, bottom
            layout.crop_top, layout.crop_bottom = top, bottom
            return top, bottom, layout
        return ov.crop_top, ov.crop_bottom, layout

    top = 0.10 if user_crop_top is None else float(user_crop_top)
    bottom = 0.65 if user_crop_bottom is None else float(user_crop_bottom)
    if not layout.panes:
        layout.panes = [_single_overview(top, bottom)]
    return top, bottom, layout


def _single_overview(top: float, bottom: float) -> CameraPane:
    return CameraPane(
        role="overview",
        crop_top=top,
        crop_bottom=bottom,
        purpose="Wide field overview — general movement for the top-down map",
    )


def _vertical_split(gray: np.ndarray, top: float, bottom: float) -> float | None:
    """Find a vertical gutter in the lower pane (two side-by-side close-ups)."""
    import cv2

    h, w = gray.shape[:2]
    y0 = int(np.clip(top, 0, 1) * h)
    y1 = int(np.clip(bottom, 0, 1) * h)
    y0, y1 = max(0, min(y0, h - 2)), max(y0 + 1, min(y1, h))
    band = gray[y0:y1, :]
    if band.size == 0 or band.shape[1] < 80:
        return None
    sobel = cv2.Sobel(band, cv2.CV_32F, 0, 1, ksize=3)
    col_energy = np.mean(np.abs(sobel), axis=0)
    x0 = int(w * 0.30)
    x1 = int(w * 0.70)
    window = max(2, w // 100)
    scores: list[tuple[float, int]] = []
    for x in range(x0, x1):
        sl = slice(max(0, x - window), min(w, x + window + 1))
        scores.append((float(np.mean(col_energy[sl])), x))
    scores.sort(key=lambda item: item[0])
    best, best_x = scores[0]
    median = float(np.median([s for s, _ in scores]))
    if median <= 1e-3:
        return None
    contrast = (median - best) / median
    if contrast < 0.18:
        return None
    return best_x / float(w)
