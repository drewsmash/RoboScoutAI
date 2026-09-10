"""OpenCV motion + background-subtraction tracker (no neural net)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import plausible_robot_size


class MotionTracker:
    name = "motion"
    kind = "local"
    description = "OpenCV MOG2 + frame-diff blobs (works offline, no model)"

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.subtractor = None
        self.tracks: dict[int, dict[str, float]] = {}
        self.next_id = 9000
        self.warm_frames = 0

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return True

    def reset(self) -> None:
        self.prev_gray = None
        self.subtractor = None
        self.tracks = {}
        self.next_id = 9000
        self.warm_frames = 0

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        import cv2

        if self.subtractor is None:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=90,
                varThreshold=24,
                detectShadows=False,
            )

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        fg = self.subtractor.apply(cropped, learningRate=0.02 if self.warm_frames < 12 else 0.005)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            delta = cv2.absdiff(self.prev_gray, gray)
            _, diff = cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(fg, diff)
        else:
            mask = fg

        self.prev_gray = gray
        self.warm_frames += 1

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids: list[tuple[float, float, list[float]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not plausible_robot_size(float(w), float(h), ctx.crop_w, ctx.crop_h):
                continue
            cx, cy = x + w * 0.5, y + h * 0.5
            centroids.append((cx, cy, [float(x), float(y), float(x + w), float(y + h)]))

        centroids.sort(key=lambda row: (row[2][2] - row[2][0]) * (row[2][3] - row[2][1]), reverse=True)
        centroids = centroids[:8]

        match_radius = max(56.0, min(ctx.crop_w, ctx.crop_h) * 0.08)
        used: set[int] = set()
        out: list[Detection] = []
        for cx, cy, bbox in centroids:
            best_id = None
            best_dist = match_radius
            for tid, state in self.tracks.items():
                if tid in used:
                    continue
                dist = ((state["x"] - cx) ** 2 + (state["y"] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            self.tracks[best_id] = {"x": cx, "y": cy, "age": 0.0}
            out.append(Detection(track_id=best_id, bbox=bbox, source=self.name, confidence=0.45))

        for tid in [t for t in self.tracks if t not in used]:
            self.tracks[tid]["age"] = self.tracks[tid].get("age", 0.0) + 1.0
            if self.tracks[tid]["age"] > 10:
                self.tracks.pop(tid, None)
        return out
