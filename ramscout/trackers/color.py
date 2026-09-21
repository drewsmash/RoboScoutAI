"""HSV bumper-color blob tracker (no neural net)."""

from __future__ import annotations

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import apply_field_mask, drop_nested_boxes, plausible_robot_size


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
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        # Slightly wider ranges for broadcast compression / LED wash.
        red = cv2.inRange(hsv, (0, 70, 55), (14, 255, 255)) | cv2.inRange(hsv, (162, 70, 55), (180, 255, 255))
        blue = cv2.inRange(hsv, (90, 60, 45), (138, 255, 255))
        kernel = np.ones((5, 5), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kernel)
        blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        blue = cv2.morphologyEx(blue, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        red = apply_field_mask(red, ctx)
        blue = apply_field_mask(blue, ctx)

        blobs: list[tuple[str, float, list[float]]] = []
        for alliance, mask in (("red", red), ("blue", blue)):
            # RETR_LIST: an alliance-colored LED strip along the whole wall is a
            # closed ring in this mask and RETR_EXTERNAL would hide every
            # bumper of that color inside it.
            contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            sized: list[tuple[float, list[float]]] = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < 10 or h < 3 or w > 0.25 * ctx.crop_w:
                    continue
                area = float(cv2.contourArea(contour))
                if area < 30:
                    continue
                if area / float(max(w * h, 1)) < 0.25:
                    continue  # hollow / ring-like, not a bumper
                # Bumpers are *thin*: ~5 in tall on a ~30 in side. A saturated
                # blob as tall as it is wide is a scale plate, alliance-wall
                # panel or LED sign, never a bumper.
                if h > 0.6 * w:
                    continue
                robot = bumper_to_robot_box(float(x), float(y), float(w), float(h), ctx.crop_w, ctx.crop_h)
                rw, rh = robot[2] - robot[0], robot[3] - robot[1]
                if not plausible_robot_size(rw, rh, ctx.crop_w, ctx.crop_h):
                    continue
                if not body_differs_from_surroundings(gray, robot):
                    # Nothing stands on this strip: alliance tape / zone
                    # outline / LED wall segment painted on bare carpet.
                    continue
                sized.append((area, robot))
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


def body_differs_from_surroundings(gray: np.ndarray, robot: list[float], *, min_diff: float = 9.0) -> bool:
    """Is there a robot body above the bumper strip, or just carpet?

    Compares the upper part of the robot box with the carpet immediately to
    its left and right (same rows): mean brightness and texture (std). Tape
    rectangles, zone outlines and LED wall strips fail because the "body"
    region is the same carpet as its neighbours.
    """
    x1, y1, x2, y2 = [int(round(v)) for v in robot]
    h, w = gray.shape[:2]
    bw = max(x2 - x1, 1)
    by1, by2 = max(y1, 0), max(y1, 0) + max(int((y2 - y1) * 0.6), 2)
    by2 = min(by2, h)
    if by2 - by1 < 2:
        return True
    body = gray[by1:by2, max(x1, 0) : min(x2, w)]
    if body.size < 8:
        return True
    ctx_parts = []
    lx1, lx2 = max(x1 - bw, 0), max(x1, 0)
    rx1, rx2 = min(x2, w), min(x2 + bw, w)
    if lx2 - lx1 >= 3:
        ctx_parts.append(gray[by1:by2, lx1:lx2])
    if rx2 - rx1 >= 3:
        ctx_parts.append(gray[by1:by2, rx1:rx2])
    if not ctx_parts:
        return True
    ctx = np.concatenate([c.reshape(-1) for c in ctx_parts])
    diff = abs(float(body.mean()) - float(ctx.mean())) + abs(float(body.std()) - float(ctx.std()))
    return diff >= min_diff


def bumper_to_robot_box(x: float, y: float, w: float, h: float, crop_w: int, crop_h: int) -> list[float]:
    """Grow a bumper strip into the robot box that stands on it.

    The strip is the bottom of the robot; the body rises roughly 0.7–0.9 of
    the bumper width above it in a broadcast view. Wider (corner-on) strips
    already include perspective, so the growth is capped.
    """
    body = float(np.clip(0.8 * w, min(2.5 * h, w), w))
    x1 = max(0.0, x - 0.04 * w)
    x2 = min(float(crop_w), x + w + 0.04 * w)
    y2 = min(float(crop_h), y + h)
    y1 = max(0.0, y2 - body - h)
    return [x1, y1, x2, y2]


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
