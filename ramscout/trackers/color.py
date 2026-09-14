"""HSV bumper-color blob tracker (no neural net)."""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import plausible_robot_size


class ColorTracker:
    name = "color"
    kind = "local"
    description = "Detect red/blue bumper color regions on the field crop"

    def __init__(self) -> None:
        self.tracks: dict[int, dict[str, float | str]] = {}
        self.next_id = 7000

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return True

    def reset(self) -> None:
        self.tracks = {}
        self.next_id = 7000

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        red = cv2.inRange(hsv, (0, 90, 70), (12, 255, 255)) | cv2.inRange(hsv, (165, 90, 70), (180, 255, 255))
        blue = cv2.inRange(hsv, (95, 80, 60), (135, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)

        blobs: list[tuple[str, list[float]]] = []
        for alliance, mask in (("red", red), ("blue", blue)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not plausible_robot_size(float(w), float(h), ctx.crop_w, ctx.crop_h):
                    continue
                # Prefer somewhat square-ish bumper plates over thin lines.
                if h < 10 or w < 12:
                    continue
                blobs.append((alliance, [float(x), float(y), float(x + w), float(y + h)]))

        blobs.sort(key=lambda row: (row[1][2] - row[1][0]) * (row[1][3] - row[1][1]), reverse=True)
        blobs = blobs[:8]

        match_radius = max(50.0, min(ctx.crop_w, ctx.crop_h) * 0.07)
        used: set[int] = set()
        out: list[Detection] = []
        for alliance, bbox in blobs:
            cx = (bbox[0] + bbox[2]) * 0.5
            cy = (bbox[1] + bbox[3]) * 0.5
            best_id = None
            best_dist = match_radius
            for tid, state in self.tracks.items():
                if tid in used:
                    continue
                if state.get("alliance") not in {alliance, "unknown"}:
                    continue
                dist = ((float(state["x"]) - cx) ** 2 + (float(state["y"]) - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            self.tracks[best_id] = {"x": cx, "y": cy, "age": 0.0, "alliance": alliance}
            out.append(
                Detection(
                    track_id=best_id,
                    bbox=bbox,
                    source=self.name,
                    confidence=0.5,
                    alliance=alliance,
                )
            )

        for tid in [t for t in self.tracks if t not in used]:
            self.tracks[tid]["age"] = float(self.tracks[tid].get("age", 0.0)) + 1.0
            if float(self.tracks[tid]["age"]) > 12:
                self.tracks.pop(tid, None)
        return out
