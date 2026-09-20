"""Strategy registry + hybrid cascade for multi-tracker robot detection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ramscout.trackers.color import ColorTracker
from ramscout.trackers.field_gate import FieldGate
from ramscout.trackers.flow import OpticalFlowTracker
from ramscout.trackers.gemini_vision import GeminiVisionTracker
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.mot import MotTracker
from ramscout.trackers.openai_vision import OpenAIVisionTracker
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import centroid, merge_detections
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
        "label": "Auto (benchmark all)",
        "strategies": ["motion", "color", "optical_flow", "gemini", "yolo", "openai"],
        "description": (
            "Runs every available strategy set (motion, color, potato, hybrid, YOLO if installed, "
            "Gemini/OpenAI if keyed) on the first ~12 s, scores each on robot count, 3v3 balance, "
            "persistence, in-field ratio, speed plausibility and motion, then tracks the whole match "
            "with the winner. The scores are recorded in the job as tracker_selection."
        ),
        "cascade": True,
        "benchmark": True,
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
    """Run strategies with optional hybrid cascade, then SORT-style MOT.

    When a :class:`~ramscout.trackers.field_gate.FieldGate` is attached (see
    :meth:`configure_geometry`), every proposal is checked against the field
    polygon / footprint / static-background model before association, and the
    MOT associates in field inches with physical speed gating.
    """

    def __init__(self, strategies: list[Any], *, cascade: bool = False) -> None:
        self.strategies = strategies
        self.cascade = cascade
        self.flow = next((s for s in strategies if getattr(s, "name", "") == "optical_flow"), None)
        self.hits: dict[str, int] = {getattr(s, "name", "unknown"): 0 for s in strategies}
        self.warnings: list[str] = []
        self.mot = MotTracker(max_age=15, min_hits=3, max_tracks=8)
        self.field_gate: FieldGate | None = None
        self._frames = 0
        self._cloud_confirm_every = 12  # processed frames between optional cloud calls

    def configure_geometry(self, gate: FieldGate | None, *, dt_s: float, static_window_s: float = 4.0) -> None:
        """Attach the field gate and switch MOT to BEV (field-inch) association."""
        self.field_gate = gate
        self.mot.dt_s = float(max(dt_s, 1e-3))
        self.mot.field_projector = gate.projector() if gate is not None else None
        self.mot.require_motion_to_confirm = gate is not None
        self.mot.static_after_hits = max(10, int(round(static_window_s / self.mot.dt_s)))

    def detect(self, cropped, ctx: TrackerContext) -> list[Detection]:
        self._frames += 1
        gate = self.field_gate
        if gate is not None:
            gate.observe(cropped)

        if self.cascade:
            merged = self._detect_cascade(cropped, ctx)
        else:
            merged = self._detect_flat(cropped, ctx)

        if gate is not None:
            merged = gate.filter(merged)

        if self.flow is not None:
            if merged:
                self.flow.seed(merged)
            try:
                flowed = self.flow.detect(cropped, ctx) or []
            except Exception:  # noqa: BLE001
                flowed = []
            if flowed and gate is not None:
                flowed = gate.filter(flowed, mark=False)
            if flowed:
                self.hits["optical_flow"] = self.hits.get("optical_flow", 0) + len(flowed)
                if len(merged) < 5:
                    merged = merge_detections(merged, flowed, min_dist=32.0)

        tracks = self.mot.update(
            merged,
            frame_w=ctx.crop_w,
            frame_h=ctx.crop_h,
            frame_bgr=cropped,
        )
        if gate is not None:
            tracks = gate.filter_tracks(tracks)
        return tracks

    def _detect_flat(self, cropped, ctx: TrackerContext) -> list[Detection]:
        primary: list[Detection] = []
        support: list[Detection] = []
        verifier: list[Detection] = []
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
            if name == "yolo" and not getattr(tracker, "robot_tuned", True):
                verifier = dets
            elif name in {"yolo", "openai", "gemini"}:
                primary = merge_detections(primary, dets)
            else:
                support = merge_detections(support, dets)
        if verifier:
            support = verify_proposals(support, verifier)
        return merge_detections(primary, support) if primary else support

    def _detect_cascade(self, cropped, ctx: TrackerContext) -> list[Detection]:
        """Fast local proposals every frame; sparse cloud/YOLO confirmation."""
        by_name = {getattr(t, "name", ""): t for t in self.strategies}

        local: dict[str, list[Detection]] = {}
        for name in ("motion", "color"):
            tracker = by_name.get(name)
            if tracker is None:
                continue
            dets = self._safe_detect(tracker, cropped, ctx)
            if dets:
                self.hits[name] = self.hits.get(name, 0) + len(dets)
                local[name] = dets
        proposals = fuse_local_proposals(local.get("motion", []), local.get("color", []))

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
                if name == "yolo" and not getattr(tracker, "robot_tuned", True):
                    # COCO classes (person / car / ...) cannot *propose* FRC
                    # robots; they only corroborate overlapping local boxes.
                    proposals = verify_proposals(proposals, dets)
                    continue
                # Cloud / FRC-tuned YOLO boxes are primary anchors; keep local fill.
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


def fuse_local_proposals(
    motion: list[Detection],
    color: list[Detection],
    *,
    min_dist: float = 28.0,
    min_iou: float = 0.2,
) -> list[Detection]:
    """Merge motion and bumper-color proposals, rewarding agreement.

    A motion blob that also contains a bumper-colored blob is very likely a
    robot: it inherits the alliance label and a confidence boost. Motion-only
    and color-only boxes are kept (with their own confidence) so recall is not
    lost, but they no longer outrank agreeing pairs in MOT.
    """
    if not motion:
        return list(color)
    if not color:
        return list(motion)
    fused: list[Detection] = []
    used_color: set[int] = set()
    for m in motion:
        mcx, mcy = centroid(m.bbox)
        best_j, best_score = -1, 0.0
        for j, c in enumerate(color):
            if j in used_color:
                continue
            ccx, ccy = centroid(c.bbox)
            dist = float(np.hypot(mcx - ccx, mcy - ccy))
            iou = _iou(m.bbox, c.bbox)
            # A bumper blob sits inside / on the lower part of the robot blob.
            inside = _contains(m.bbox, (ccx, ccy))
            score = iou + (0.5 if inside else 0.0) + (0.3 if dist < min_dist else 0.0)
            if (iou >= min_iou or inside or dist < min_dist) and score > best_score:
                best_j, best_score = j, score
        if best_j >= 0:
            c = color[best_j]
            used_color.add(best_j)
            meta = dict(m.meta or {})
            meta["agree"] = ["motion", "color"]
            # Only a blob in the lower part of the robot box is a bumper; a
            # colored mechanism / jersey / LED up top must not set the alliance.
            _ccx, ccy = centroid(c.bbox)
            lower = ccy >= m.bbox[1] + 0.4 * (m.bbox[3] - m.bbox[1])
            fused.append(
                Detection(
                    track_id=m.track_id,
                    bbox=list(m.bbox),
                    source="motion",
                    confidence=float(min(0.92, max(m.confidence, c.confidence) + (0.15 if lower else 0.05))),
                    alliance=c.alliance if lower else m.alliance,
                    team=m.team or c.team,
                    meta=meta,
                )
            )
        else:
            fused.append(m)
    for j, c in enumerate(color):
        if j not in used_color:
            fused.append(c)
    return merge_detections([], fused, min_dist=min_dist)


def verify_proposals(
    proposals: list[Detection],
    verifier: list[Detection],
    *,
    min_iou: float = 0.25,
    boost: float = 0.15,
) -> list[Detection]:
    """Raise confidence of local proposals that a verifier box overlaps.

    Used for detectors that are *not* robot-tuned (COCO YOLO): their boxes
    never enter the proposal set, so a bleacher "person" cannot start a
    track, but a proposal they agree with is trusted more.
    """
    out: list[Detection] = []
    for p in proposals:
        best = max((_iou(p.bbox, v.bbox) for v in verifier), default=0.0)
        if best >= min_iou:
            meta = dict(p.meta or {})
            meta["verified_by"] = "yolo"
            meta["verify_iou"] = round(float(best), 3)
            p = Detection(
                track_id=p.track_id,
                bbox=list(p.bbox),
                source=p.source,
                confidence=float(min(0.95, p.confidence + boost)),
                alliance=p.alliance,
                team=p.team,
                meta=meta,
            )
        out.append(p)
    return out


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / union) if union > 0 else 0.0


def _contains(bbox: list[float], point: tuple[float, float]) -> bool:
    x, y = point
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def _build_pool(model_path: str | None = None) -> dict[str, Any]:
    return {
        "motion": MotionTracker(),
        "color": ColorTracker(),
        "optical_flow": OpticalFlowTracker(),
        "yolo": YoloTracker(model_path=model_path),
        "openai": OpenAIVisionTracker(),
        "gemini": GeminiVisionTracker(),
    }
