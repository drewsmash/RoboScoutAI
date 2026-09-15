"""SORT-inspired multi-object tracker for FRC robot boxes.

Associates per-frame detections (any source) into stable track IDs using:
- constant-velocity Kalman prediction
- IoU + centroid distance + alliance / color cues
- short-gap rebirth so occlusions do not mint endless new IDs

This is the glue that makes potato/cloud/YOLO detections temporally coherent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ramscout.trackers.types import Detection
from ramscout.trackers.utils import centroid


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def _clamp_bbox(bbox: list[float], w: float, h: float) -> list[float]:
    x1, y1, x2, y2 = bbox
    x1 = float(np.clip(x1, 0, max(w - 1, 0)))
    y1 = float(np.clip(y1, 0, max(h - 1, 0)))
    x2 = float(np.clip(x2, x1 + 1, max(w, x1 + 1)))
    y2 = float(np.clip(y2, y1 + 1, max(h, y1 + 1)))
    return [x1, y1, x2, y2]


@dataclass
class _KalmanBox:
    """Simple constant-velocity filter on (cx, cy, w, h)."""

    mean: np.ndarray  # 8: cx,cy,w,h,vx,vy,vw,vh
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    @classmethod
    def from_bbox(cls, bbox: list[float]) -> "_KalmanBox":
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return cls(mean=mean)

    def predict(self) -> None:
        self.mean[0] += self.mean[4]
        self.mean[1] += self.mean[5]
        self.mean[2] = max(1.0, self.mean[2] + self.mean[6])
        self.mean[3] = max(1.0, self.mean[3] + self.mean[7])
        # Dampen velocity slightly so coasting does not runaway.
        self.mean[4:8] *= 0.92
        self.age += 1
        self.time_since_update += 1

    def update(self, bbox: list[float]) -> None:
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        prev = self.mean.copy()
        self.mean[0], self.mean[1], self.mean[2], self.mean[3] = cx, cy, w, h
        self.mean[4] = 0.6 * prev[4] + 0.4 * (cx - prev[0])
        self.mean[5] = 0.6 * prev[5] + 0.4 * (cy - prev[1])
        self.mean[6] = 0.5 * prev[6] + 0.5 * (w - prev[2])
        self.mean[7] = 0.5 * prev[7] + 0.5 * (h - prev[3])
        self.hits += 1
        self.time_since_update = 0

    def bbox(self) -> list[float]:
        cx, cy, w, h = self.mean[:4]
        return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


@dataclass
class _Track:
    track_id: int
    kalman: _KalmanBox
    alliance: str = "unknown"
    team: str = ""
    source: str = "mot"
    confidence: float = 0.5
    color_hist: np.ndarray | None = None
    confirmed: bool = False


@dataclass
class MotTracker:
    """Global multi-object tracker over merged detector outputs."""

    max_age: int = 18
    min_hits: int = 2
    iou_threshold: float = 0.18
    max_tracks: int = 8
    next_id: int = 1000
    tracks: dict[int, _Track] = field(default_factory=dict)

    def reset(self) -> None:
        self.tracks = {}
        self.next_id = 1000

    def update(
        self,
        detections: list[Detection],
        *,
        frame_w: int,
        frame_h: int,
        frame_bgr: np.ndarray | None = None,
    ) -> list[Detection]:
        # Predict all live tracks forward one step.
        for track in self.tracks.values():
            track.kalman.predict()

        track_ids = list(self.tracks.keys())
        if not track_ids and not detections:
            return []

        cost = self._cost_matrix(track_ids, detections, frame_w, frame_h, frame_bgr)
        matches, unmatched_tracks, unmatched_dets = self._associate(cost, track_ids, detections)

        for ti, di in matches:
            tid = track_ids[ti]
            det = detections[di]
            track = self.tracks[tid]
            track.kalman.update(det.bbox)
            track.confidence = max(track.confidence * 0.7, float(det.confidence))
            track.source = det.source or track.source
            if det.alliance in {"red", "blue"}:
                track.alliance = det.alliance
            if det.team:
                track.team = det.team
            if frame_bgr is not None:
                track.color_hist = _patch_hist(frame_bgr, det.bbox)
            if track.kalman.hits >= self.min_hits:
                track.confirmed = True

        for ti in unmatched_tracks:
            tid = track_ids[ti]
            track = self.tracks[tid]
            if track.kalman.time_since_update > self.max_age:
                self.tracks.pop(tid, None)

        for di in unmatched_dets:
            if len(self.tracks) >= self.max_tracks:
                # Drop oldest unmatched tentative track to make room.
                self._evict_weakest()
            if len(self.tracks) >= self.max_tracks:
                break
            det = detections[di]
            hist = _patch_hist(frame_bgr, det.bbox) if frame_bgr is not None else None
            tid = self.next_id
            self.next_id += 1
            track = _Track(
                track_id=tid,
                kalman=_KalmanBox.from_bbox(det.bbox),
                alliance=det.alliance if det.alliance in {"red", "blue"} else "unknown",
                team=det.team or "",
                source=det.source or "mot",
                confidence=float(det.confidence),
                color_hist=hist,
                confirmed=False,
            )
            track.kalman.hits = 1
            track.kalman.time_since_update = 0
            self.tracks[tid] = track

        out: list[Detection] = []
        for track in self.tracks.values():
            # Emit confirmed tracks. Allow immediate emit for strong cloud/YOLO anchors.
            if not track.confirmed:
                strong_anchor = track.confidence >= 0.55 and track.source in {
                    "gemini",
                    "openai",
                    "yolo",
                    "color",
                }
                if track.kalman.hits < self.min_hits and not strong_anchor:
                    continue
                if track.kalman.time_since_update > 0:
                    continue
            bbox = _clamp_bbox(track.kalman.bbox(), frame_w, frame_h)
            out.append(
                Detection(
                    track_id=track.track_id,
                    bbox=bbox,
                    source=track.source,
                    confidence=track.confidence,
                    alliance=track.alliance,
                    team=track.team,
                )
            )
        # Prefer longer-lived / higher-confidence when overfilled.
        out.sort(key=lambda d: (d.confidence, -abs(d.track_id)), reverse=True)
        return out[: self.max_tracks]

    def _cost_matrix(
        self,
        track_ids: list[int],
        detections: list[Detection],
        frame_w: int,
        frame_h: int,
        frame_bgr: np.ndarray | None,
    ) -> np.ndarray:
        if not track_ids or not detections:
            return np.zeros((len(track_ids), len(detections)), dtype=np.float64)
        diag = float(np.hypot(frame_w, frame_h) or 1.0)
        cost = np.full((len(track_ids), len(detections)), 1e3, dtype=np.float64)
        for i, tid in enumerate(track_ids):
            track = self.tracks[tid]
            tb = track.kalman.bbox()
            tcx, tcy = centroid(tb)
            for j, det in enumerate(detections):
                iou = _iou(tb, det.bbox)
                dcx, dcy = centroid(det.bbox)
                dist = float(np.hypot(tcx - dcx, tcy - dcy)) / diag
                alliance_pen = 0.0
                if (
                    track.alliance in {"red", "blue"}
                    and det.alliance in {"red", "blue"}
                    and track.alliance != det.alliance
                ):
                    alliance_pen = 0.55
                hist_pen = 0.0
                if track.color_hist is not None and frame_bgr is not None:
                    dh = _patch_hist(frame_bgr, det.bbox)
                    if dh is not None:
                        # Bhattacharyya distance in [0,1]; smaller is better.
                        hist_pen = float(cv2_compare_hist(track.color_hist, dh))
                # Lower cost is better.
                cost[i, j] = (1.0 - iou) + 1.4 * dist + alliance_pen + 0.35 * hist_pen
                if iou < self.iou_threshold * 0.35 and dist > 0.18:
                    cost[i, j] += 2.0
        return cost

    def _associate(
        self,
        cost: np.ndarray,
        track_ids: list[int],
        detections: list[Detection],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        from ramscout.identity import linear_assignment

        if cost.size == 0:
            return [], list(range(len(track_ids))), list(range(len(detections)))
        pairs = linear_assignment(cost)
        matches: list[tuple[int, int]] = []
        used_t: set[int] = set()
        used_d: set[int] = set()
        for ti, di in pairs:
            if ti >= cost.shape[0] or di >= cost.shape[1]:
                continue
            if cost[ti, di] > 1.35:
                continue
            matches.append((ti, di))
            used_t.add(ti)
            used_d.add(di)
        unmatched_tracks = [i for i in range(len(track_ids)) if i not in used_t]
        unmatched_dets = [j for j in range(len(detections)) if j not in used_d]
        return matches, unmatched_tracks, unmatched_dets

    def _evict_weakest(self) -> None:
        if not self.tracks:
            return
        weakest = min(
            self.tracks.values(),
            key=lambda t: (t.confirmed, t.kalman.hits, -t.kalman.time_since_update, t.confidence),
        )
        if not weakest.confirmed or weakest.kalman.time_since_update > 4:
            self.tracks.pop(weakest.track_id, None)


def _patch_hist(frame_bgr: np.ndarray, bbox: list[float]) -> np.ndarray | None:
    import cv2

    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    patch = frame_bgr[y1:y2, x1:x2]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.astype(np.float32)


def cv2_compare_hist(a: np.ndarray, b: np.ndarray) -> float:
    import cv2

    try:
        return float(cv2.compareHist(a, b, cv2.HISTCMP_BHATTACHARYYA))
    except Exception:  # noqa: BLE001
        return 0.5
