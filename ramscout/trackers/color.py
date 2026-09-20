"""HSV bumper-color blob tracker (no neural net)."""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import drop_nested_boxes, plausible_robot_size


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
            # RETR_LIST: an alliance-colored LED strip along the whole wall is a
            # closed ring in this mask and RETR_EXTERNAL would hide every
            # bumper of that color inside it.
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            sized: list[tuple[float, list[float]]] = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if not plausible_robot_size(float(w), float(h), ctx.crop_w, ctx.crop_h):
                    continue
                if h < 8 or w < 10:
                    continue
                area = float(cv2.contourArea(contour))
                if area < 35:
                    continue
                if area / float(max(w * h, 1)) < 0.25:
                    continue  # hollow / ring-like, not a bumper
                sized.append((area, [float(x), float(y), float(x + w), float(y + h)]))
            sized.sort(key=lambda row: row[0], reverse=True)
            for area, bbox in drop_nested_boxes(sized):
                blobs.append((alliance, area, bbox))

        blobs.sort(key=lambda row: row[1], reverse=True)
        blobs = lowest_band_rule(blobs)
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


def lowest_band_rule(
    blobs: list[tuple[str, float, list[float]]],
    *,
    x_overlap: float = 0.5,
    reach: float = 1.6,
) -> list[tuple[str, float, list[float]]]:
    """Keep only the lowest colored band in a vertical stack.

    Bumpers sit at the bottom of a robot; a red/blue mechanism, jersey, or
    LED strip above them (within ``reach`` × width) is a decoy, not a second
    robot. Blobs without a lower neighbour are untouched.
    """
    kept: list[tuple[str, float, list[float]]] = []
    for alliance, area, bbox in blobs:
        x1, y1, x2, y2 = bbox
        w = max(x2 - x1, 1.0)
        shadowed = False
        for _oa, _oarea, ob in blobs:
            if ob is bbox:
                continue
            ox1, oy1, ox2, oy2 = ob
            if oy2 <= y2:
                continue  # not lower than us
            ow = max(ox2 - ox1, 1.0)
            overlap = max(0.0, min(x2, ox2) - max(x1, ox1)) / min(w, ow)
            if overlap < x_overlap:
                continue
            if (oy1 - y2) <= reach * max(w, ow):
                shadowed = True
                break
        if not shadowed:
            kept.append((alliance, area, bbox))
    return kept
