"""Strategy registry + hybrid cascade for multi-tracker robot detection."""

from __future__ import annotations

from typing import Any

from ramscout.trackers.color import ColorTracker
from ramscout.trackers.flow import OpticalFlowTracker
from ramscout.trackers.gemini_vision import GeminiVisionTracker
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.mot import MotTracker
from ramscout.trackers.openai_vision import OpenAIVisionTracker
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import merge_detections
from ramscout.trackers.yolo import YoloTracker

# Pure OpenCV strategies — no models, no API keys, works on low-end machines.
POTATO_STRATEGIES: list[str] = ["motion", "color", "optical_flow"]

TRACKER_MODES: dict[str, dict[str, Any]] = {
    "hybrid": {
        "label": "Hybrid cascade (recommended)",
        "strategies": ["motion", "color", "optical_flow", "gemini", "yolo", "openai"],
        "description": (
            "Novel local cascade: motion/color proposals every frame → optical-flow coast → "
            "sparse Gemini/YOLO/OpenAI confirmation only when needed. SORT-style MOT for stable IDs."
        ),
        "cascade": True,
    },
    "potato": {
        "label": "Potato (no AI)",
        "strategies": list(POTATO_STRATEGIES),
        "description": (
            "Dumb fallback: OpenCV motion blobs + bumper color + optical flow. "
            "No YOLO, no API keys. Browser can also run a JS frame-diff fallback."
        ),
    },
    "auto": {
        "label": "Auto",
        "strategies": ["gemini", "yolo", "openai", "color", "motion", "optical_flow"],
        "description": (
            "Prefer Gemini when a Google key is set, then YOLO/OpenAI/OpenCV. "
            "Uses SORT MOT; falls back to potato OpenCV so paths are never empty."
        ),
    },
    "local": {
        "label": "Local only",
        "strategies": ["yolo", "color", "motion", "optical_flow"],
        "description": "Offline OpenCV + optional YOLO. No cloud calls.",
    },
    "motion": {
        "label": "Motion blobs",
        "strategies": ["motion", "optical_flow"],
        "description": "Background subtraction + optical flow (no object detector).",
    },
    "color": {
        "label": "Bumper color",
        "strategies": ["color", "motion", "optical_flow"],
        "description": "HSV red/blue bumper regions plus motion fill.",
    },
    "yolo": {
        "label": "YOLO neural",
        "strategies": ["yolo", "motion", "optical_flow"],
        "description": "Ultralytics YOLO with motion fallback.",
    },
    "openai": {
        "label": "OpenAI Vision",
        "strategies": ["openai", "motion", "color", "optical_flow"],
        "description": "OpenAI sparse keyframe boxes + local tracking between frames.",
    },
    "gemini": {
        "label": "Google Gemini",
        "strategies": ["gemini", "motion", "color", "optical_flow"],
        "description": "Gemini sparse keyframe boxes + local tracking between frames.",
    },
    "cloud": {
        "label": "Cloud ensemble",
        "strategies": ["gemini", "openai", "color", "motion", "optical_flow"],
        "description": "Prefer Gemini, then OpenAI, when keyed; always keep local fallbacks.",
    },
    "all": {
        "label": "Everything",
        "strategies": ["gemini", "yolo", "openai", "color", "motion", "optical_flow"],
        "description": "Run all configured strategies and merge detections.",
    },
}


def list_strategies(model_path: str | None = None) -> list[dict[str, Any]]:
    trackers = _build_pool(model_path)
    ctx = TrackerContext(frame_w=1280, frame_h=720, crop_w=1280, crop_h=400)
    rows = []
    for name, tracker in trackers.items():
        rows.append(
            {
                "name": name,
                "kind": getattr(tracker, "kind", "local"),
                "description": getattr(tracker, "description", ""),
                "available": bool(tracker.available(ctx)),
            }
        )
    return rows


def resolve_strategies(
    mode: str = "hybrid",
    *,
    model_path: str | None = None,
    openai_key: str = "",
    google_key: str = "",
) -> tuple[list[Any], list[str]]:
    """Return ordered tracker instances for a mode, plus setup warnings."""
    mode = (mode or "hybrid").strip().lower()
    # Backward-compatible alias: auto now means the hybrid cascade.
    if mode == "auto":
        mode = "hybrid"
    if mode not in TRACKER_MODES:
        mode = "hybrid"
    wanted = list(TRACKER_MODES[mode]["strategies"])
    pool = _build_pool(model_path)
    ctx = TrackerContext(
        frame_w=1280,
        frame_h=720,
        crop_w=1280,
        crop_h=400,
        openai_key=openai_key,
        google_key=google_key,
    )
    selected = []
    warnings: list[str] = []
    for name in wanted:
        tracker = pool.get(name)
        if tracker is None:
            continue
        if tracker.available(ctx):
            selected.append(tracker)
        else:
            if name in {"openai", "gemini", "yolo"}:
                warnings.append(
                    f"Tracker '{name}' unavailable for this run "
                    f"({getattr(tracker, 'description', name)})."
                )
    # Always keep potato OpenCV trackers so paths are never empty by design.
    for name in POTATO_STRATEGIES:
        if not any(getattr(t, "name", "") == name for t in selected):
            selected.append(pool[name])
            if mode != "potato" and name == "motion":
                warnings.append(
                    "Falling back to potato OpenCV trackers (motion/color/flow) — "
                    "no model or API key required."
                )
    for tracker in selected:
        tracker.reset()
    return selected, warnings


class EnsembleTracker:
    """Run strategies with optional hybrid cascade, then SORT-style MOT."""

    def __init__(self, strategies: list[Any], *, cascade: bool = False) -> None:
        self.strategies = strategies
        self.cascade = cascade
        self.flow = next((s for s in strategies if getattr(s, "name", "") == "optical_flow"), None)
        self.hits: dict[str, int] = {getattr(s, "name", "unknown"): 0 for s in strategies}
        self.warnings: list[str] = []
        self.mot = MotTracker(max_age=20, min_hits=2, max_tracks=8)
        self._frames = 0
        self._cloud_confirm_every = 12  # processed frames between optional cloud calls

    def detect(self, cropped, ctx: TrackerContext) -> list[Detection]:
        self._frames += 1
        if self.cascade:
            merged = self._detect_cascade(cropped, ctx)
        else:
            merged = self._detect_flat(cropped, ctx)

        if self.flow is not None:
            if merged:
                self.flow.seed(merged)
            try:
                flowed = self.flow.detect(cropped, ctx) or []
            except Exception:  # noqa: BLE001
                flowed = []
            if flowed:
                self.hits["optical_flow"] = self.hits.get("optical_flow", 0) + len(flowed)
                if len(merged) < 5:
                    merged = merge_detections(merged, flowed, min_dist=32.0)

        return self.mot.update(
            merged,
            frame_w=ctx.crop_w,
            frame_h=ctx.crop_h,
            frame_bgr=cropped,
        )

    def _detect_flat(self, cropped, ctx: TrackerContext) -> list[Detection]:
        primary: list[Detection] = []
        support: list[Detection] = []
        gemini_hit = False
        for tracker in self.strategies:
            name = getattr(tracker, "name", "unknown")
            if name == "optical_flow":
                continue
            if name == "openai" and gemini_hit:
                continue
            dets = self._safe_detect(tracker, cropped, ctx)
            if not dets:
                continue
            self.hits[name] = self.hits.get(name, 0) + len(dets)
            if name == "gemini":
                gemini_hit = True
            if name in {"yolo", "openai", "gemini"}:
                primary = merge_detections(primary, dets)
            else:
                support = merge_detections(support, dets)
        return merge_detections(primary, support) if primary else support

    def _detect_cascade(self, cropped, ctx: TrackerContext) -> list[Detection]:
        """Fast local proposals every frame; sparse cloud/YOLO confirmation."""
        proposals: list[Detection] = []
        by_name = {getattr(t, "name", ""): t for t in self.strategies}

        for name in ("motion", "color"):
            tracker = by_name.get(name)
            if tracker is None:
                continue
            dets = self._safe_detect(tracker, cropped, ctx)
            if dets:
                self.hits[name] = self.hits.get(name, 0) + len(dets)
                proposals = merge_detections(proposals, dets, min_dist=28.0)

        need_confirm = self._needs_cloud_confirm(proposals, ctx)
        if need_confirm:
            gemini_ok = False
            for name in ("gemini", "yolo", "openai"):
                tracker = by_name.get(name)
                if tracker is None:
                    continue
                if name == "openai" and gemini_ok:
                    continue
                dets = self._safe_detect(tracker, cropped, ctx)
                if not dets:
                    continue
                self.hits[name] = self.hits.get(name, 0) + len(dets)
                # Cloud/YOLO boxes are primary anchors; keep local fill.
                proposals = merge_detections(dets, proposals, min_dist=30.0)
                if name == "gemini":
                    gemini_ok = True
                    break
                if name == "yolo" and len(dets) >= 3:
                    break
                if name == "openai" and dets:
                    break

        return proposals

    def _needs_cloud_confirm(self, proposals: list[Detection], ctx: TrackerContext) -> bool:
        # Always allow first few frames to warm MOT without cloud.
        if self._frames <= 3:
            return False
        paced = self._frames % max(self._cloud_confirm_every, 1) == 0
        n = len(proposals)
        # Empty proposals: confirm on paced frames only (cloud trackers also self-pace).
        if n == 0:
            return paced or self._frames in {4, 8}
        if n > 8:
            return paced
        if (n < 3 or n > 7) and paced:
            return True
        mean_conf = sum(d.confidence for d in proposals) / max(n, 1)
        if mean_conf < 0.48 and paced:
            return True
        alliances = {d.alliance for d in proposals if d.alliance in {"red", "blue"}}
        if len(alliances) < 2 and n >= 3 and paced:
            return True
        return self._frames % max(self._cloud_confirm_every * 3, 1) == 0

    def _safe_detect(self, tracker: Any, cropped, ctx: TrackerContext) -> list[Detection]:
        name = getattr(tracker, "name", "unknown")
        try:
            dets = tracker.detect(cropped, ctx) or []
        except Exception as exc:  # noqa: BLE001
            msg = f"{name} tracker error ({exc})."
            if msg not in self.warnings:
                self.warnings.append(msg)
            dets = []
        extra_warn = getattr(tracker, "warnings", None)
        if extra_warn:
            for w in extra_warn:
                if w not in self.warnings:
                    self.warnings.append(w)
        return dets


def _build_pool(model_path: str | None = None) -> dict[str, Any]:
    return {
        "motion": MotionTracker(),
        "color": ColorTracker(),
        "optical_flow": OpticalFlowTracker(),
        "yolo": YoloTracker(model_path=model_path),
        "openai": OpenAIVisionTracker(),
        "gemini": GeminiVisionTracker(),
    }
