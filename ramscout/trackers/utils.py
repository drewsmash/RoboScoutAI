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
    merged = list(primary)
    for det in secondary:
        x1, y1, x2, y2 = det.bbox
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        conflict = False
        for existing in merged:
            ex1, ey1, ex2, ey2 = existing.bbox
            ecx, ecy = (ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5
            if ((ecx - cx) ** 2 + (ecy - cy) ** 2) ** 0.5 < min_dist:
                conflict = True
                break
        if not conflict:
            merged.append(det)
    return merged


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
