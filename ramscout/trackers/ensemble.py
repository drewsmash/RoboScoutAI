"""Strategy registry + cascade modes for multi-tracker robot detection."""

from __future__ import annotations

from typing import Any

from ramscout.trackers.color import ColorTracker
from ramscout.trackers.flow import OpticalFlowTracker
from ramscout.trackers.gemini_vision import GeminiVisionTracker
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.openai_vision import OpenAIVisionTracker
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import merge_detections
from ramscout.trackers.yolo import YoloTracker

# Pure OpenCV strategies — no models, no API keys, works on low-end machines.
POTATO_STRATEGIES: list[str] = ["motion", "color", "optical_flow"]

TRACKER_MODES: dict[str, dict[str, Any]] = {
    "potato": {
        "label": "Potato (no AI)",
        "strategies": list(POTATO_STRATEGIES),
        "description": (
            "Dumb fallback: OpenCV motion blobs + bumper color + optical flow. "
            "No YOLO, no API keys. Browser can also run a JS frame-diff fallback."
        ),
    },
    "auto": {
        "label": "Auto (recommended)",
        "strategies": ["gemini", "yolo", "openai", "color", "motion", "optical_flow"],
        "description": (
            "Prefer Gemini when a Google key is set, then YOLO/OpenAI/OpenCV. "
            "Skips missing backends and always keeps potato OpenCV so paths are never empty."
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
        "description": "OpenAI keyframe boxes + local tracking between frames.",
    },
    "gemini": {
        "label": "Google Gemini",
        "strategies": ["gemini", "motion", "color", "optical_flow"],
        "description": "Gemini keyframe boxes + local tracking between frames.",
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
    mode: str = "auto",
    *,
    model_path: str | None = None,
    openai_key: str = "",
    google_key: str = "",
) -> tuple[list[Any], list[str]]:
    """Return ordered tracker instances for a mode, plus setup warnings."""
    mode = (mode or "auto").strip().lower()
    if mode not in TRACKER_MODES:
        mode = "auto"
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
    """Run multiple strategies per frame and merge boxes."""

    def __init__(self, strategies: list[Any]) -> None:
        self.strategies = strategies
        self.flow = next((s for s in strategies if getattr(s, "name", "") == "optical_flow"), None)
        self.hits: dict[str, int] = {getattr(s, "name", "unknown"): 0 for s in strategies}
        self.warnings: list[str] = []

    def detect(self, cropped, ctx: TrackerContext) -> list[Detection]:
        primary: list[Detection] = []
        support: list[Detection] = []
        gemini_hit = False
        for tracker in self.strategies:
            name = getattr(tracker, "name", "unknown")
            if name == "optical_flow":
                continue
            # Gemini-first: skip OpenAI HTTP when Gemini already anchored this frame.
            if name == "openai" and gemini_hit:
                continue
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
            if not dets:
                continue
            self.hits[name] = self.hits.get(name, 0) + len(dets)
            if name == "gemini":
                gemini_hit = True
            # Prefer neural/cloud as primary anchors; color/motion fill gaps.
            if name in {"yolo", "openai", "gemini"}:
                primary = merge_detections(primary, dets)
            else:
                support = merge_detections(support, dets)

        merged = merge_detections(primary, support) if primary else support
        if self.flow is not None:
            if merged:
                self.flow.seed(merged)
            try:
                flowed = self.flow.detect(cropped, ctx) or []
            except Exception:  # noqa: BLE001
                flowed = []
            if flowed:
                self.hits["optical_flow"] = self.hits.get("optical_flow", 0) + len(flowed)
                if len(merged) < 4:
                    merged = merge_detections(merged, flowed)
                elif not merged:
                    merged = flowed
        return merged


def _build_pool(model_path: str | None = None) -> dict[str, Any]:
    return {
        "motion": MotionTracker(),
        "color": ColorTracker(),
        "optical_flow": OpticalFlowTracker(),
        "yolo": YoloTracker(model_path=model_path),
        # OpenAI defaults are thrifty (sparse seconds/frames + max calls); see openai_vision.py.
        "openai": OpenAIVisionTracker(),
        "gemini": GeminiVisionTracker(),
    }
