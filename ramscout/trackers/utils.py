"""Geometry helpers shared by local trackers."""

from __future__ import annotations

from typing import Any

import numpy as np

from ramscout.trackers.types import Detection


def plausible_robot_size(bw: float, bh: float, frame_w: int, frame_h: int) -> bool:
    if bw <= 3 or bh <= 3:
        return False
    if bw > frame_w * 0.40 or bh > frame_h * 0.60:
        return False
    area = bw * bh
    frame_area = max(frame_w * frame_h, 1)
    if area < frame_area * 0.00025:
        return False
    if area > frame_area * 0.18:
        return False
    aspect = bw / max(bh, 1.0)
    return 0.28 <= aspect <= 4.0


def merge_detections(
    primary: list[Detection],
    secondary: list[Detection],
    min_dist: float = 40.0,
) -> list[Detection]:
    """Merge boxes; prefer primary on near-duplicates (centroid or IoU)."""
    merged = list(primary)
    for det in secondary:
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        conflict = False
        for existing in merged:
            ex1, ey1, ex2, ey2 = existing.bbox
            ecx, ecy = (ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5
            dist = ((ecx - cx) ** 2 + (ecy - cy) ** 2) ** 0.5
            if dist < min_dist:
                conflict = True
                break
            # IoU overlap also counts as duplicate.
            ix1, iy1 = max(x1, ex1), max(y1, ey1)
            ix2, iy2 = min(x2, ex2), min(y2, ey2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter > 0:
                area_a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                area_b = max(0.0, ex2 - ex1) * max(0.0, ey2 - ey1)
                iou = inter / max(area_a + area_b - inter, 1e-6)
                if iou >= 0.35:
                    conflict = True
                    break
        if not conflict:
            merged.append(det)
    return merged


def drop_nested_boxes(
    blobs: list[tuple[float, list[float]]],
    containment: float = 0.8,
) -> list[tuple[float, list[float]]]:
    """Drop (area, bbox) rows mostly contained in a larger kept box.

    ``RETR_LIST`` contour retrieval returns inner contours too (holes inside a
    blob); this keeps the outer box only. Input should be sorted by area desc.
    """
    kept: list[tuple[float, list[float]]] = []
    for area, bbox in blobs:
        x1, y1, x2, y2 = bbox
        own = max((x2 - x1) * (y2 - y1), 1.0)
        nested = False
        for _a, kb in kept:
            ix = max(0.0, min(x2, kb[2]) - max(x1, kb[0]))
            iy = max(0.0, min(y2, kb[3]) - max(y1, kb[1]))
            if (ix * iy) / own >= containment:
                nested = True
                break
        if not nested:
            kept.append((area, bbox))
    return kept


def detections_to_dicts(dets: list[Detection]) -> list[dict[str, Any]]:
    return [d.as_dict() for d in dets]


def centroid(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5


def encode_jpeg_bgr(frame_bgr: np.ndarray, quality: int = 70) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Could not encode JPEG for cloud vision.")
    return bytes(buf)
