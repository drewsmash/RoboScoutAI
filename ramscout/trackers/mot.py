"""SORT-inspired multi-object tracker for FRC robot boxes.

Associates per-frame detections (any source) into stable track IDs using:
- constant-velocity Kalman prediction in pixel space
- **constant-velocity Kalman in field inches (BEV)** when a projector is set,
  with hard gating by physical speed (FRC robots top out near 20 ft/s)
- IoU + centroid distance + alliance / color cues + optical-flow direction
- short-gap rebirth so occlusions do not mint endless new IDs
- per-source confirmation (local proposals need more hits than YOLO / cloud)
- static-track pruning (blobs that never move are walls / field elements)
- output NMS so two strategies never emit the same robot twice

This is the glue that makes potato/cloud/YOLO detections temporally coherent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ramscout.trackers.types import Detection
from ramscout.trackers.utils import centroid

STRONG_SOURCES = {"gemini", "openai", "yolo"}
FieldProjector = Callable[[Sequence[float]], tuple[float, float]]

# 20 ft/s ≈ 240 in/s. Real robots rarely exceed ~17 ft/s; slack covers
# projection noise and box jitter.
DEFAULT_MAX_SPEED_IN_S = 240.0


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


def _containment(inner: list[float], outer: list[float]) -> float:
    """Fraction of ``inner`` area that lies inside ``outer``."""
    ix = max(0.0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0.0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    own = max((inner[2] - inner[0]) * (inner[3] - inner[1]), 1e-6)
    return float(ix * iy / own)


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
    last_z: np.ndarray | None = None  # last measured cx,cy,w,h

    @classmethod
    def from_bbox(cls, bbox: list[float]) -> "_KalmanBox":
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        mean = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return cls(mean=mean, last_z=mean[:4].copy())

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
        z = np.array([cx, cy, w, h], dtype=np.float64)
        prev_v = self.mean[4:8].copy()
        # Velocity from measured displacement per step (not the innovation,
        # which would systematically under-estimate speed by ~50 %).
        steps = float(max(self.time_since_update, 1))
        last = self.last_z if self.last_z is not None else z
        dz = (z - last) / steps
        self.mean[0], self.mean[1], self.mean[2], self.mean[3] = cx, cy, w, h
        self.mean[4] = 0.6 * prev_v[0] + 0.4 * dz[0]
        self.mean[5] = 0.6 * prev_v[1] + 0.4 * dz[1]
        self.mean[6] = 0.5 * prev_v[2] + 0.5 * dz[2]
        self.mean[7] = 0.5 * prev_v[3] + 0.5 * dz[3]
        self.last_z = z
        self.hits += 1
        self.time_since_update = 0

    def bbox(self) -> list[float]:
        cx, cy, w, h = self.mean[:4]
        return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


@dataclass
class _FieldKalman:
    """Constant-velocity filter in field inches: (x, y, vx, vy)."""

    mean: np.ndarray  # x, y, vx, vy (in, in, in/s, in/s)
    dt: float
    path_in: float = 0.0
    origin: np.ndarray | None = None
    last_z: np.ndarray | None = None

    @classmethod
    def from_xy(cls, x: float, y: float, dt: float) -> "_FieldKalman":
        mean = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        return cls(mean=mean, dt=dt, origin=mean[:2].copy(), last_z=mean[:2].copy())

    def predict(self) -> None:
        self.mean[0] += self.mean[2] * self.dt
        self.mean[1] += self.mean[3] * self.dt
        self.mean[2:4] *= 0.9

    def update(self, x: float, y: float, elapsed: float) -> None:
        prev_v = self.mean[2:4].copy()
        last = self.last_z if self.last_z is not None else np.array([x, y])
        step = float(np.hypot(x - last[0], y - last[1]))
        self.path_in += step
        el = max(elapsed, 1e-3)
        self.mean[0], self.mean[1] = x, y
        self.mean[2] = 0.6 * prev_v[0] + 0.4 * (x - last[0]) / el
        self.mean[3] = 0.6 * prev_v[1] + 0.4 * (y - last[1]) / el
        self.last_z = np.array([x, y], dtype=np.float64)

    @property
    def speed_in_s(self) -> float:
        return float(np.hypot(self.mean[2], self.mean[3]))

    def displacement_in(self) -> float:
        if self.origin is None:
            return 0.0
        return float(np.hypot(self.mean[0] - self.origin[0], self.mean[1] - self.origin[1]))


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
    field: _FieldKalman | None = None
    origin_px: tuple[float, float] | None = None
    path_px: float = 0.0
    strong_hits: int = 0
    speed_rejects: int = 0

    def displacement_px(self) -> float:
        if self.origin_px is None:
            return 0.0
        cx, cy = self.kalman.mean[0], self.kalman.mean[1]
        return float(np.hypot(cx - self.origin_px[0], cy - self.origin_px[1]))


@dataclass
class MotTracker:
    """Global multi-object tracker over merged detector outputs."""

    max_age: int = 15
    min_hits: int = 3
    iou_threshold: float = 0.18
    max_tracks: int = 8
    next_id: int = 1000
    tracks: dict[int, _Track] = field(default_factory=dict)
    # BEV association (optional): bbox → field inches, seconds per update.
    field_projector: FieldProjector | None = None
    dt_s: float = 0.1
    max_speed_in_s: float = DEFAULT_MAX_SPEED_IN_S
    speed_slack_in: float = 18.0
    field_scale_in: float = 30.0  # ~one robot length
    # Static rejection: a track that has not moved this much after this many
    # updates is a wall / field element (unless YOLO / cloud anchored it).
    static_after_hits: int = 20
    static_min_disp_px: float = 8.0
    static_min_disp_in: float = 10.0
    strong_min_hits: int = 2
    output_nms_iou: float = 0.5
    # With a field gate attached: never spawn from a static-flagged proposal
    # and require real displacement before a local track is confirmed.
    require_motion_to_confirm: bool = False
    confirm_min_disp_px: float = 6.0
    confirm_min_disp_in: float = 8.0

    def reset(self) -> None:
        self.tracks = {}
        self.next_id = 1000

    # ------------------------------------------------------------------ helpers
    def _field_xy(self, det: Detection) -> tuple[float, float] | None:
        xy = None
        if det.meta:
            xy = det.meta.get("field_xy")
        if xy is None and self.field_projector is not None:
            try:
                xy = self.field_projector(det.bbox)
            except Exception:  # noqa: BLE001
                xy = None
        if xy is None:
            return None
        x, y = float(xy[0]), float(xy[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return x, y

    def allowed_step_in(self, time_since_update: int) -> float:
        """Max plausible field displacement for a track coasting this long."""
        elapsed = self.dt_s * max(int(time_since_update), 1)
        return self.max_speed_in_s * elapsed * 1.5 + self.speed_slack_in

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
            if track.field is not None:
                track.field.predict()

        track_ids = list(self.tracks.keys())
        if not track_ids and not detections:
            return []

        det_field = [self._field_xy(d) for d in detections]
        cost = self._cost_matrix(track_ids, detections, det_field, frame_w, frame_h, frame_bgr)
        matches, unmatched_tracks, unmatched_dets = self._associate(cost, track_ids, detections)

        for ti, di in matches:
            tid = track_ids[ti]
            det = detections[di]
            track = self.tracks[tid]
            elapsed_steps = max(track.kalman.time_since_update, 1)
            track.kalman.update(det.bbox)
            track.confidence = max(track.confidence * 0.7, float(det.confidence))
            track.source = det.source or track.source
            if det.source in STRONG_SOURCES:
                track.strong_hits += 1
            if det.alliance in {"red", "blue"}:
                track.alliance = det.alliance
            if det.team:
                track.team = det.team
            if frame_bgr is not None:
                track.color_hist = _patch_hist(frame_bgr, det.bbox)
            xy = det_field[di]
            if xy is not None:
                if track.field is None:
                    track.field = _FieldKalman.from_xy(xy[0], xy[1], self.dt_s)
                else:
                    track.field.update(xy[0], xy[1], self.dt_s * elapsed_steps)
            cx, cy = centroid(det.bbox)
            if track.origin_px is None:
                track.origin_px = (cx, cy)
            if self._can_confirm(track):
                track.confirmed = True

        for ti in unmatched_tracks:
            tid = track_ids[ti]
            track = self.tracks[tid]
            if track.kalman.time_since_update > self.max_age:
                self.tracks.pop(tid, None)

        for di in unmatched_dets:
            det = detections[di]
            if (
                det.meta
                and det.meta.get("static")
                and det.source not in STRONG_SOURCES
            ):
                # Static structures never get to start a track.
                continue
            if len(self.tracks) >= self.max_tracks:
                # Drop oldest unmatched tentative track to make room.
                self._evict_weakest()
            if len(self.tracks) >= self.max_tracks:
                break
            hist = _patch_hist(frame_bgr, det.bbox) if frame_bgr is not None else None
            tid = self.next_id
            self.next_id += 1
            xy = det_field[di]
            track = _Track(
                track_id=tid,
                kalman=_KalmanBox.from_bbox(det.bbox),
                alliance=det.alliance if det.alliance in {"red", "blue"} else "unknown",
                team=det.team or "",
                source=det.source or "mot",
                confidence=float(det.confidence),
                color_hist=hist,
                confirmed=False,
                field=_FieldKalman.from_xy(xy[0], xy[1], self.dt_s) if xy is not None else None,
                origin_px=centroid(det.bbox),
                strong_hits=1 if det.source in STRONG_SOURCES else 0,
            )
            track.kalman.hits = 1
            track.kalman.time_since_update = 0
            track.confirmed = self._can_confirm(track)
            self.tracks[tid] = track

        self._prune_static()

        out: list[Detection] = []
        for track in self.tracks.values():
            # Emit confirmed tracks. Allow immediate emit for strong cloud/YOLO anchors.
            if not track.confirmed:
                strong_anchor = track.confidence >= 0.5 and track.source in STRONG_SOURCES
                if not strong_anchor:
                    continue
                if track.kalman.time_since_update > 0:
                    continue
            bbox = _clamp_bbox(track.kalman.bbox(), frame_w, frame_h)
            meta: dict = {
                "hits": int(track.kalman.hits),
                "age": int(track.kalman.age),
                "coasting": int(track.kalman.time_since_update),
                "moving": self._is_moving(track),
            }
            if track.field is not None:
                meta["field_xy"] = (float(track.field.mean[0]), float(track.field.mean[1]))
                meta["speed_in_s"] = round(track.field.speed_in_s, 1)
                meta["path_in"] = round(track.field.path_in, 1)
            out.append(
                Detection(
                    track_id=track.track_id,
                    bbox=bbox,
                    source=track.source,
                    confidence=track.confidence,
                    alliance=track.alliance,
                    team=track.team,
                    meta=meta,
                )
            )
        # Prefer longer-lived / higher-confidence when overfilled.
        out.sort(key=lambda d: (d.meta.get("hits", 0), d.confidence), reverse=True)
        out = self._nms(out)
        return out[: self.max_tracks]

    def _can_confirm(self, track: _Track) -> bool:
        need = self.strong_min_hits if track.strong_hits > 0 else self.min_hits
        if track.kalman.hits < need:
            return False
        if track.strong_hits > 0 or not self.require_motion_to_confirm:
            return True
        if track.displacement_px() >= self.confirm_min_disp_px:
            return True
        return track.field is not None and track.field.displacement_in() >= self.confirm_min_disp_in

    def _is_moving(self, track: _Track) -> bool:
        if track.field is not None and track.field.displacement_in() >= self.static_min_disp_in:
            return True
        return track.displacement_px() >= self.static_min_disp_px or track.kalman.hits < 4

    def _prune_static(self) -> None:
        """Remove tracks that have never moved since birth (walls, field elements)."""
        for tid, track in list(self.tracks.items()):
            if track.strong_hits > 0:
                continue
            if track.kalman.hits < self.static_after_hits:
                continue
            disp_px = track.displacement_px()
            disp_in = track.field.displacement_in() if track.field is not None else None
            path_in = track.field.path_in if track.field is not None else 0.0
            static_px = disp_px < self.static_min_disp_px
            static_in = disp_in is not None and disp_in < self.static_min_disp_in
            wandered = path_in >= 4.0 * self.static_min_disp_in
            if static_px and (disp_in is None or static_in) and not wandered:
                self.tracks.pop(tid, None)

    def _nms(self, dets: list[Detection]) -> list[Detection]:
        """Suppress duplicates: high IoU, or a box mostly contained in a
        stronger one (a bumper / mechanism blob riding inside the robot box)."""
        kept: list[Detection] = []
        for det in dets:
            dup = False
            for k in kept:
                if _iou(det.bbox, k.bbox) >= self.output_nms_iou or _containment(det.bbox, k.bbox) >= 0.7:
                    dup = True
                    break
            if not dup:
                kept.append(det)
        return kept

    def _cost_matrix(
        self,
        track_ids: list[int],
        detections: list[Detection],
        det_field: list[tuple[float, float] | None],
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
            tvx, tvy = float(track.kalman.mean[4]), float(track.kalman.mean[5])
            allowed = self.allowed_step_in(track.kalman.time_since_update)
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
                c = (1.0 - iou) + 1.4 * dist + alliance_pen + 0.35 * hist_pen
                if iou < self.iou_threshold * 0.35 and dist > 0.18:
                    c += 2.0

                # BEV gating: impossible physical speed ⇒ never associate.
                xy = det_field[j]
                if xy is not None and track.field is not None:
                    fd = float(np.hypot(xy[0] - track.field.mean[0], xy[1] - track.field.mean[1]))
                    if fd > allowed:
                        cost[i, j] = 1e3
                        continue
                    # Soft term scaled by one robot length so it complements
                    # (rather than overrides) IoU and alliance cues.
                    c += 0.4 * min(fd / self.field_scale_in, 1.5)

                # Flow-direction consistency: a proposal drifting against the
                # track's velocity is a different object passing by.
                flow = det.meta.get("flow") if det.meta else None
                if flow is not None and np.hypot(tvx, tvy) > 2.0:
                    fx, fy = float(flow[0]), float(flow[1])
                    if np.hypot(fx, fy) > 2.0:
                        cosang = (fx * tvx + fy * tvy) / (np.hypot(fx, fy) * np.hypot(tvx, tvy) + 1e-6)
                        if cosang < 0.0:
                            c += 0.3
                cost[i, j] = c
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
