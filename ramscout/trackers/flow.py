"""Lucas–Kanade optical-flow track continuation (no neural net)."""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import plausible_robot_size


class OpticalFlowTracker:
    """Propagates previous detections with sparse optical flow between keyframes."""

    name = "optical_flow"
    kind = "local"
    description = "Lucas–Kanade optical flow to keep tracks alive between detections"

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.points: dict[int, np.ndarray] = {}  # track_id -> (x,y) center
        self.sizes: dict[int, tuple[float, float]] = {}
        self.next_id = 8000

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return True

    def reset(self) -> None:
        self.prev_gray = None
        self.points = {}
        self.sizes = {}
        self.next_id = 8000

    def seed(self, detections: list[Detection]) -> None:
        """Ingest fresh detections so flow has points to follow."""
        for det in detections:
            cx = (det.bbox[0] + det.bbox[2]) * 0.5
            cy = (det.bbox[1] + det.bbox[3]) * 0.5
            self.points[det.track_id] = np.array([cx, cy], dtype=np.float32)
            self.sizes[det.track_id] = (det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1])

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        import cv2

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        out: list[Detection] = []
        if self.prev_gray is None or not self.points:
            self.prev_gray = gray
            return out

        ids = list(self.points.keys())
        pts = np.array([self.points[i] for i in ids], dtype=np.float32).reshape(-1, 1, 2)
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            pts,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        self.prev_gray = gray
        if nxt is None or status is None:
            return out

        alive: dict[int, np.ndarray] = {}
        for tid, pt, ok in zip(ids, nxt.reshape(-1, 2), status.reshape(-1)):
            if not ok:
                self.sizes.pop(tid, None)
                continue
            cx, cy = float(pt[0]), float(pt[1])
            if cx < 0 or cy < 0 or cx >= ctx.crop_w or cy >= ctx.crop_h:
                self.sizes.pop(tid, None)
                continue
            bw, bh = self.sizes.get(tid, (36.0, 28.0))
            bbox = [cx - bw * 0.5, cy - bh * 0.5, cx + bw * 0.5, cy + bh * 0.5]
            if not plausible_robot_size(bw, bh, ctx.crop_w, ctx.crop_h):
                continue
            alive[tid] = np.array([cx, cy], dtype=np.float32)
            out.append(Detection(track_id=tid, bbox=bbox, source=self.name, confidence=0.35))
        self.points = alive
        return out
