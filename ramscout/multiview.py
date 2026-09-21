"""Process sectional camera panes separately for movement vs scoring/climb.

Overview (top wide-angle) owns top-down paths. Side panes add confidence to
hub scoring and climb heuristics for the matching alliance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.multicam import CameraLayout, CameraPane

ProgressFn = Callable[[str, float], None]


@dataclass
class SideCue:
    t: float
    alliance: str
    kind: str  # hub_activity | climb_activity | motion
    confidence: float
    detail: str
    role: str
    x_norm: float = 0.5
    y_norm: float = 0.5

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "alliance": self.alliance,
            "kind": self.kind,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
            "role": self.role,
            "x_norm": round(self.x_norm, 3),
            "y_norm": round(self.y_norm, 3),
            "source": "side_cam",
        }


@dataclass
class MultiViewResult:
    layout: dict[str, Any]
    side_cues: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "side_cues": self.side_cues,
            "warnings": self.warnings,
        }


def process_side_views(
    video_path: Path | str,
    layout: CameraLayout,
    *,
    frame_stride: int = 6,
    max_frames: int | None = 400,
    on_progress: ProgressFn | None = None,
) -> MultiViewResult:
    """Scan side/sideline panes for scoring and climb motion cues."""
    import cv2

    panes = layout.side_panes()
    result = MultiViewResult(layout=layout.as_dict())
    if not panes:
        result.warnings.append("No side camera panes — scoring cues use overview tracks only.")
        return result

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        result.warnings.append(f"Could not open video for side views: {path}")
        return result

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    subtractors = {pane.role: cv2.createBackgroundSubtractorMOG2(history=80, varThreshold=32, detectShadows=False) for pane in panes}
    prev_centroids: dict[str, tuple[float, float] | None] = {pane.role: None for pane in panes}
    cues: list[SideCue] = []
    frame_index = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        t = frame_index / fps
        for pane in panes:
            crop = pane.slice_frame(frame)
            if crop.size == 0:
                continue
            cue = _analyze_side_crop(
                crop,
                subtractors[pane.role],
                pane,
                t=t,
                prev=prev_centroids[pane.role],
            )
            if cue is not None:
                prev_centroids[pane.role] = (cue.x_norm, cue.y_norm)
                cues.append(cue)
            else:
                # Keep prev if we briefly lose the blob.
                pass

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Reading side cameras…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    # Thin dense cues: keep local maxima per ~2s / alliance / kind.
    result.side_cues = [c.as_dict() for c in _thin_cues(cues)]
    if result.side_cues:
        result.warnings.append(
            f"Side cameras contributed {len(result.side_cues)} scoring/climb cues "
            f"across {len(panes)} pane(s)."
        )
    else:
        result.warnings.append("Side cameras scanned but no strong scoring/climb cues found.")
    return result


def detect_all_panes(
    video_path: Path | str,
    layout: CameraLayout,
    *,
    frame_stride: int = 5,
    max_frames: int | None = 500,
    on_progress: ProgressFn | None = None,
    skip_roles: set[str] | None = None,
) -> dict[str, Any]:
    """Run bumper-color + motion detection independently on every non-graphics pane.

    Overview tracking still owns the top-down map; this pass produces per-pane
    detections (absolute frame coordinates) so we can correlate blue/red angle
    shots and overview_alt with the main feed.
    """
    import cv2

    from ramscout.trackers.color import ColorTracker
    from ramscout.trackers.motion import MotionTracker
    from ramscout.trackers.types import TrackerContext

    skip = set(skip_roles or {"graphics", "scorebug", "other"})
    panes = [p for p in (layout.panes or []) if p.role not in skip]
    # Prefer side + alt; overview is optional here (already tracked separately).
    panes = [p for p in panes if p.role != "overview"] or panes
    out: dict[str, Any] = {"detections": [], "warnings": [], "panes": [p.role for p in panes]}
    if not panes:
        out["warnings"].append("No panes available for per-view detection.")
        return out

    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        out["warnings"].append(f"Could not open video for pane detection: {path}")
        return out

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    color = ColorTracker()
    motion = MotionTracker()
    dets: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        fh, fw = frame.shape[:2]
        t = frame_index / fps
        for pane in panes:
            crop = pane.slice_frame(frame)
            if crop.size == 0 or crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            ch, cw = crop.shape[:2]
            x0 = int(np.clip(pane.crop_left, 0, 1) * fw)
            y0 = int(np.clip(pane.crop_top, 0, 1) * fh)
            ctx = TrackerContext(frame_w=fw, frame_h=fh, crop_w=cw, crop_h=ch, t=t, frame_index=frame_index)
            found: list[Any] = []
            try:
                found.extend(color.detect(crop, ctx))
            except Exception:  # noqa: BLE001
                pass
            try:
                found.extend(motion.detect(crop, ctx))
            except Exception:  # noqa: BLE001
                pass
            # Deduplicate by IoU-ish center distance inside the crop.
            kept: list[Any] = []
            for det in found:
                bx0, by0, bx1, by1 = det.bbox
                cx = 0.5 * (bx0 + bx1)
                cy = 0.5 * (by0 + by1)
                if any(abs(cx - 0.5 * (k.bbox[0] + k.bbox[2])) < 18 and abs(cy - 0.5 * (k.bbox[1] + k.bbox[3])) < 18 for k in kept):
                    continue
                kept.append(det)
            alliance_bias = pane.alliance_bias or (
                "blue" if pane.role.startswith("blue") else "red" if pane.role.startswith("red") else None
            )
            for det in kept[:6]:
                bx0, by0, bx1, by1 = det.bbox
                alliance = det.alliance if det.alliance in {"red", "blue"} else (alliance_bias or "")
                dets.append(
                    {
                        "t": round(t, 3),
                        "role": pane.role,
                        "alliance": alliance or "",
                        "confidence": float(det.confidence or 0.4),
                        "x_norm": float(np.clip((0.5 * (bx0 + bx1)) / max(cw, 1), 0, 1)),
                        "y_norm": float(np.clip((0.5 * (by0 + by1)) / max(ch, 1), 0, 1)),
                        "bbox": [
                            float(x0 + bx0),
                            float(y0 + by0),
                            float(x0 + bx1),
                            float(y0 + by1),
                        ],
                        "source": f"pane:{det.source}",
                        "view": pane.role,
                    }
                )

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Detecting robots per camera pane…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    out["detections"] = _thin_pane_detections(dets)
    out["warnings"].append(
        f"Per-pane detection: {len(out['detections'])} robot hits across "
        f"{len(panes)} pane(s) ({', '.join(p.role for p in panes)})."
    )
    return out


def _thin_pane_detections(dets: list[dict[str, Any]], *, bin_s: float = 1.0) -> list[dict[str, Any]]:
    """Keep the strongest detection per role/alliance/~1s bin."""
    best: dict[tuple[str, str, int], dict[str, Any]] = {}
    for d in dets:
        key = (
            str(d.get("role") or ""),
            str(d.get("alliance") or ""),
            int(float(d.get("t") or 0.0) / max(bin_s, 0.25)),
        )
        prev = best.get(key)
        if prev is None or float(d.get("confidence") or 0) >= float(prev.get("confidence") or 0):
            best[key] = d
    return sorted(best.values(), key=lambda r: float(r.get("t") or 0.0))


def merge_side_cues_into_events(
    events: list[dict[str, Any]],
    side_cues: list[dict[str, Any]],
    *,
    cards: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Boost matching hub/climb events or invent soft candidates from side cues."""
    if not side_cues:
        return events

    out = [dict(e) for e in events]
    teams_by_alliance: dict[str, list[str]] = {"blue": [], "red": []}
    for card in cards or []:
        alliance = str(card.get("alliance") or "")
        team = str(card.get("team") or "")
        if alliance in teams_by_alliance and team:
            teams_by_alliance[alliance].append(team)

    for cue in side_cues:
        alliance = cue.get("alliance") if cue.get("alliance") in {"red", "blue"} else None
        if not alliance:
            continue
        kind = cue.get("kind")
        t = float(cue.get("t") or 0.0)
        conf = float(cue.get("confidence") or 0.4)
        matched = False
        for event in out:
            if abs(float(event.get("t") or 0.0) - t) > 3.5:
                continue
            etype = str(event.get("type") or "")
            event_alliance = _event_alliance(event, teams_by_alliance)
            if event_alliance and event_alliance != alliance:
                continue
            if kind == "hub_activity" and "hub" in etype:
                event["confidence"] = float(min(0.98, float(event.get("confidence") or 0.5) + 0.18 * conf))
                detail = str(event.get("detail") or "")
                note = "side-cam confirm"
                if note not in detail:
                    event["detail"] = f"{detail}; {note}".strip("; ")
                event["side_confirmed"] = True
                matched = True
            elif kind == "climb_activity" and "climb" in etype:
                event["confidence"] = float(min(0.98, float(event.get("confidence") or 0.5) + 0.22 * conf))
                detail = str(event.get("detail") or "")
                note = "side-cam confirm"
                if note not in detail:
                    event["detail"] = f"{detail}; {note}".strip("; ")
                event["side_confirmed"] = True
                matched = True
        if matched:
            continue

        # Soft candidate when overview tracking missed the action.
        # Prefer a correlated team id; fall back to first alliance card.
        teams = teams_by_alliance.get(alliance) or []
        team = str(cue.get("team") or "") or (teams[0] if teams else alliance)
        if kind == "hub_activity" and conf >= 0.45:
            out.append(
                {
                    "team": team,
                    "type": "hub_score_candidate",
                    "t": t,
                    "zone": f"{alliance}_hub",
                    "confidence": round(0.35 + 0.4 * conf, 3),
                    "detail": cue.get("detail") or "Side camera hub activity",
                    "period": "",
                    "duration_s": 0.0,
                    "source": cue.get("source") or "side_cam",
                    "track_id": cue.get("track_id"),
                }
            )
        elif kind == "climb_activity" and conf >= 0.5:
            out.append(
                {
                    "team": team,
                    "type": "climb_attempt",
                    "t": t,
                    "zone": f"{alliance}_tower",
                    "confidence": round(0.4 + 0.4 * conf, 3),
                    "detail": cue.get("detail") or "Side camera climb activity",
                    "period": "",
                    "duration_s": 0.0,
                    "source": cue.get("source") or "side_cam",
                    "track_id": cue.get("track_id"),
                }
            )

    out.sort(key=lambda e: float(e.get("t") or 0.0))
    return out


def _analyze_side_crop(
    crop: np.ndarray,
    subtractor: Any,
    pane: CameraPane,
    *,
    t: float,
    prev: tuple[float, float] | None,
) -> SideCue | None:
    import cv2

    h, w = crop.shape[:2]
    if h < 24 or w < 24:
        return None
    mask = subtractor.apply(crop)
    mask = cv2.medianBlur(mask, 5)
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < (h * w) * 0.004:
        return None
    x, y, bw, bh = cv2.boundingRect(contour)
    cx = (x + bw * 0.5) / w
    cy = (y + bh * 0.5) / h
    aspect = bh / max(bw, 1)

    alliance = pane.alliance_bias or ("blue" if cx < 0.5 else "red")
    # Climb: tall vertical motion high in the frame during endgame-ish times.
    if aspect >= 1.35 and cy < 0.55 and t >= 110:
        conf = float(np.clip(0.4 + 0.3 * min(aspect / 2.5, 1.0) + (0.15 if t >= 125 else 0), 0, 0.95))
        return SideCue(
            t=t,
            alliance=alliance,
            kind="climb_activity",
            confidence=conf,
            detail=f"{pane.role} vertical activity (possible climb)",
            role=pane.role,
            x_norm=cx,
            y_norm=cy,
        )

    # Hub / scoring: sustained mid-frame activity, especially toward alliance hubs.
    hub_side = (alliance == "blue" and cx < 0.62) or (alliance == "red" and cx > 0.38)
    if hub_side and 0.2 < cy < 0.85 and area > (h * w) * 0.01:
        speed = 0.0
        if prev is not None:
            speed = ((cx - prev[0]) ** 2 + (cy - prev[1]) ** 2) ** 0.5
        # Low speed dwell ≈ scoring; high speed ≈ transit (ignore).
        if speed < 0.08:
            conf = float(np.clip(0.35 + 0.4 * min(area / (h * w * 0.08), 1.0), 0, 0.9))
            return SideCue(
                t=t,
                alliance=alliance,
                kind="hub_activity",
                confidence=conf,
                detail=f"{pane.role} dwell near scoring area",
                role=pane.role,
                x_norm=cx,
                y_norm=cy,
            )

    if area > (h * w) * 0.02:
        return SideCue(
            t=t,
            alliance=alliance,
            kind="motion",
            confidence=0.25,
            detail=f"{pane.role} motion",
            role=pane.role,
            x_norm=cx,
            y_norm=cy,
        )
    return None


def _thin_cues(cues: list[SideCue], window_s: float = 2.0) -> list[SideCue]:
    best: dict[tuple[str, str, int], SideCue] = {}
    for cue in cues:
        if cue.kind == "motion":
            continue
        bucket = int(cue.t / window_s)
        key = (cue.alliance, cue.kind, bucket)
        prev = best.get(key)
        if prev is None or cue.confidence > prev.confidence:
            best[key] = cue
    return sorted(best.values(), key=lambda c: c.t)


def _event_alliance(event: dict[str, Any], teams_by_alliance: dict[str, list[str]]) -> str | None:
    team = str(event.get("team") or "")
    for alliance, teams in teams_by_alliance.items():
        if team in teams:
            return alliance
    zone = str(event.get("zone") or "")
    if zone.startswith("blue"):
        return "blue"
    if zone.startswith("red"):
        return "red"
    return None
