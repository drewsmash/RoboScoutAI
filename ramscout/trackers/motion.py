"""OpenCV motion + background-subtraction tracker (no neural net).

Uses MOG2 + KNN fusion, frame-diff, and light morphological cleanup to propose
robot-sized blobs even on busy FRC carpets.
"""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import apply_field_mask, drop_nested_boxes, plausible_robot_size


class MotionTracker:
    name = "motion"
    kind = "local"
    description = "OpenCV MOG2/KNN + frame-diff blobs (works offline, no model)"

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.mog2 = None
        self.knn = None
        self.warm_frames = 0

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return True

    def reset(self) -> None:
        self.prev_gray = None
        self.mog2 = None
        self.knn = None
        self.warm_frames = 0

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        import cv2

        if self.mog2 is None:
            self.mog2 = cv2.createBackgroundSubtractorMOG2(
                history=120,
                varThreshold=20,
                detectShadows=False,
            )
        if self.knn is None:
            try:
                self.knn = cv2.createBackgroundSubtractorKNN(
                    history=100,
                    dist2Threshold=420.0,
                    detectShadows=False,
                )
            except Exception:  # noqa: BLE001
                self.knn = None

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        learn = 0.03 if self.warm_frames < 15 else 0.004
        fg_mog = self.mog2.apply(cropped, learningRate=learn)
        _, fg_mog = cv2.threshold(fg_mog, 190, 255, cv2.THRESH_BINARY)
        if self.knn is not None:
            fg_knn = self.knn.apply(cropped, learningRate=learn)
            _, fg_knn = cv2.threshold(fg_knn, 190, 255, cv2.THRESH_BINARY)
            # KNN marks the whole frame as foreground until it has history;
            # fusing it in during warm-up would swallow every robot.
            fg = cv2.bitwise_or(fg_mog, fg_knn) if self.warm_frames >= 8 else fg_mog
        else:
            fg = fg_mog

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            delta = cv2.absdiff(self.prev_gray, gray)
            _, diff = cv2.threshold(delta, 14, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(fg, diff)
        else:
            mask = fg

        self.prev_gray = gray
        self.warm_frames += 1
        mask = apply_field_mask(mask, ctx)

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
        # RETR_LIST, not RETR_EXTERNAL: when the field perimeter (LED strips,
        # flickering wall) forms a closed ring in the mask, EXTERNAL returns
        # only the ring and silently drops every robot inside it.
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        blobs: list[tuple[float, list[float]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not plausible_robot_size(float(w), float(h), ctx.crop_w, ctx.crop_h):
                continue
            area = float(cv2.contourArea(contour))
            if area < 40:
                continue
            # Reject very elongated scorebug / wall lines.
            aspect = w / max(h, 1)
            if aspect > 3.6 or aspect < 0.28:
                continue
            # Ring-like contours (wall segments) enclose mostly empty space.
            fill = area / float(max(w * h, 1))
            if fill < 0.22:
                continue
            blobs.append((area, [float(x), float(y), float(x + w), float(y + h)]))

        blobs.sort(key=lambda row: row[0], reverse=True)
        blobs = drop_nested_boxes(blobs)
        # FRC has at most 6 robots; keep a couple extras for noise before MOT.
        blobs = blobs[:8]
        out: list[Detection] = []
        for i, (area, bbox) in enumerate(blobs):
            conf = float(np.clip(0.35 + min(area, 4000) / 8000.0, 0.35, 0.65))
            out.append(
                Detection(
                    track_id=9000 + i,  # MOT reassigns; local id is only a hint
                    bbox=bbox,
                    source=self.name,
                    confidence=conf,
                )
            )
        return out
