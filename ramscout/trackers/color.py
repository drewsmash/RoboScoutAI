"""HSV bumper-color blob tracker (no neural net)."""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import plausible_robot_size


class ColorTracker:
    name = "color"
    kind = "local"
    description = "Detect red/blue bumper color regions on the field crop"

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return True

    def reset(self) -> None:
        return None

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        import cv2

        hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)
        # Slightly wider ranges for broadcast compression / LED wash.
        red = cv2.inRange(hsv, (0, 70, 55), (14, 255, 255)) | cv2.inRange(hsv, (162, 70, 55), (180, 255, 255))
        blue = cv2.inRange(hsv, (90, 60, 45), (138, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        blobs: list[tuple[str, float, list[float]]] = []
        for alliance, mask in (("red", red), ("blue", blue)):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not plausible_robot_size(float(w), float(h), ctx.crop_w, ctx.crop_h):
                    continue
                if h < 8 or w < 10:
                    continue
                area = float(cv2.contourArea(contour))
                if area < 35:
                    continue
                blobs.append((alliance, area, [float(x), float(y), float(x + w), float(y + h)]))

        blobs.sort(key=lambda row: row[1], reverse=True)
        blobs = blobs[:8]
        out: list[Detection] = []
        for i, (alliance, area, bbox) in enumerate(blobs):
            conf = float(np.clip(0.42 + min(area, 3500) / 7000.0, 0.42, 0.72))
            out.append(
                Detection(
                    track_id=7000 + i,
                    bbox=bbox,
                    source=self.name,
                    confidence=conf,
                    alliance=alliance,
                )
            )
        return out
