"""Burn robot paths / labels onto a short annotated match clip."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def export_annotated_clip(
    video_path: Path | str,
    out_path: Path | str,
    samples: list[dict[str, Any]],
    *,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    max_seconds: float = 45.0,
    stride: int = 2,
) -> dict[str, Any]:
    """Write a lightweight MP4 with team trails overlaid on the crop."""
    import cv2

    from ramscout.geometry import crop_bounds

    src = Path(video_path)
    dest = Path(out_path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {src}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    y0, y1 = crop_bounds(height, crop_top, crop_bottom)
    out_h = max(y1 - y0, 1)
    out_w = width

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(dest), fourcc, max(fps / max(stride, 1), 1.0), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open writer: {dest}")

    by_t: dict[int, list[dict[str, Any]]] = {}
    for sample in samples or []:
        key = int(float(sample.get("t") or 0) * 5)
        by_t.setdefault(key, []).append(sample)

    colors = {
        "blue": (220, 160, 60),
        "red": (60, 60, 220),
    }
    frame_i = 0
    written = 0
    max_frames = int(max_seconds * fps)
    while frame_i < max_frames:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_i % max(stride, 1) != 0:
            frame_i += 1
            continue
        t = frame_i / fps
        crop = frame[y0:y1, :].copy()
        bucket = by_t.get(int(t * 5), [])
        for sample in bucket:
            # Samples are field inches; without inverse homography we draw a
            # legend strip instead of wrong pixel projections.
            alliance = str(sample.get("alliance") or "blue")
            team = str(sample.get("team") or "?")
            color = colors.get(alliance, (200, 200, 200))
            x = int(np.clip(float(sample.get("fx") or sample.get("px") or 0), 0, out_w - 1))
            y = int(np.clip(float(sample.get("fy") or sample.get("py") or 0) - y0, 0, out_h - 1))
            if x <= 0 and y <= 0:
                continue
            cv2.circle(crop, (x, y), 10, color, -1)
            cv2.putText(
                crop,
                team,
                (x + 12, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        cv2.putText(
            crop,
            f"t={t:.1f}s  RamScoutAI annotated",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(crop)
        written += 1
        frame_i += 1

    cap.release()
    writer.release()
    return {
        "path": str(dest),
        "frames": written,
        "width": out_w,
        "height": out_h,
        "seconds": round(written * stride / fps, 2),
    }
