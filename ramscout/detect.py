"""Robot detection and tracking on a recorded match video.

Orchestrates multiple strategies via ramscout.trackers:
- Local: motion, bumper color, optical flow
- Neural: Ultralytics YOLO (optional)
- Cloud: OpenAI Vision / Google Gemini (optional API keys)

Paths are produced even when neural/cloud backends are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.geometry import crop_bounds, default_source_points, homography_from_corners, project_points
from ramscout.identity import bumper_alliance, constrain_to_teams, ocr_digits
from ramscout.trackers.ensemble import TRACKER_MODES, EnsembleTracker, list_strategies, resolve_strategies
from ramscout.trackers.types import TrackerContext
from ramscout.trackers.yolo import (
    ensure_detector_weights,
    find_local_model,
    load_detector,
    robot_class_ids,
    ultralytics_available,
)

ProgressFn = Callable[[str, float], None]

__all__ = [
    "TRACKER_MODES",
    "ensure_detector_weights",
    "find_local_model",
    "list_strategies",
    "load_detector",
    "robot_class_ids",
    "save_jpeg",
    "track_video",
    "ultralytics_available",
]


def track_video(
    video_path: Path,
    homography: np.ndarray | None = None,
    src_points: list[list[float]] | None = None,
    model_path: str | None = None,
    team_numbers: list[str] | None = None,
    frame_stride: int = 4,
    max_frames: int | None = None,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    on_progress: ProgressFn | None = None,
    motion_only: bool = False,
    tracker_mode: str = "hybrid",
    openai_key: str = "",
    google_key: str = "",
    openai_model: str = "",
    google_model: str = "",
) -> dict[str, Any]:
    """Run multi-strategy detection + tracking. Returns field-space samples."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cal_index = int(min(fps * 5.0, max(total * 0.08, 1))) if total else int(fps * 5.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cal_index, 0))
    ok, first = cap.read()
    if not ok or first is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Could not read a calibration frame from the video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if homography is None:
        pts = src_points or default_source_points(frame_w, frame_h, crop_top, crop_bottom).tolist()
        homography = homography_from_corners(pts)
    y0, y1 = crop_bounds(frame_h, crop_top, crop_bottom)
    crop_h = max(y1 - y0, 1)
    crop_w = max(frame_w, 1)

    raw_mode = "motion" if motion_only else (tracker_mode or "hybrid")
    mode = "hybrid" if (raw_mode or "").strip().lower() == "auto" else (raw_mode or "hybrid")
    strategies, setup_warnings = resolve_strategies(
        mode,
        model_path=model_path,
        openai_key=openai_key,
        google_key=google_key,
    )
    cascade = bool(TRACKER_MODES.get(mode, {}).get("cascade")) or mode == "hybrid"
    ensemble = EnsembleTracker(strategies, cascade=cascade)
    warnings = list(setup_warnings)
    strategy_names = [getattr(s, "name", "?") for s in strategies]
    label = "hybrid cascade" if cascade else mode
    warnings.append(f"Tracking mode '{label}' using: {', '.join(strategy_names)} + SORT MOT.")

    samples: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0
    source_hits: dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        t = frame_index / fps
        cropped = frame[y0:y1, :]
        ctx = TrackerContext(
            frame_w=frame_w,
            frame_h=frame_h,
            crop_w=crop_w,
            crop_h=crop_h,
            t=t,
            frame_index=frame_index,
            team_numbers=list(team_numbers or []),
            openai_key=openai_key,
            google_key=google_key,
            openai_model=openai_model or "gpt-4o-mini",
            google_model=google_model or "gemini-3.5-flash",
        )
        detections = ensemble.detect(cropped, ctx)

        feet: list[list[float]] = []
        meta: list[tuple[int, str, str, list[float], float, float, str]] = []
        for det in detections:
            x1, y1b, x2, y2 = det.bbox
            fx = (x1 + x2) * 0.5
            fy = y2 + y0
            roi = cropped[max(int(y1b), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)]
            alliance = det.alliance if det.alliance in {"red", "blue"} else bumper_alliance(roi)
            if alliance == "unknown":
                alliance = "blue" if fx < frame_w * 0.5 else "red"
            team = det.team or ""
            if not team and team_numbers:
                raw = ocr_digits(roi)
                team = constrain_to_teams(raw, team_numbers) or ""
            feet.append([fx, fy])
            meta.append(
                (
                    int(det.track_id),
                    alliance,
                    team,
                    [x1, y1b + y0, x2, y2 + y0],
                    fx,
                    fy,
                    det.source,
                )
            )
            source_hits[det.source] = source_hits.get(det.source, 0) + 1

        if feet:
            mapped = project_points(feet, homography)
            for (mx, my), (tid, alliance, team, bbox, px, py, source) in zip(mapped, meta):
                samples.append(
                    {
                        "t": round(float(t), 3),
                        "frame": frame_index,
                        "track_id": tid,
                        "x": float(mx),
                        "y": float(my),
                        "px": float(px),
                        "py": float(py),
                        "alliance": alliance,
                        "team": team,
                        "bbox": bbox,
                        "source": source,
                    }
                )

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Tracking robots…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    for w in ensemble.warnings:
        if w not in warnings:
            warnings.append(w)

    used_model = any(n in {"yolo", "openai", "gemini"} for n in source_hits)
    if not samples:
        warnings.append(
            "No robot tracks found in the video crop. Recalibrate the four field corners, "
            "widen the field crop (Advanced), try another tracking mode, or upload a clearer wide-angle VOD."
        )
    elif not used_model:
        warnings.append(
            "Tracked with local OpenCV methods only (no YOLO/cloud hits). Paths are approximate — confirm before pick lists."
        )

    return {
        "samples": samples,
        "warnings": warnings,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "first_frame": first,
        "homography": homography.tolist(),
        "used_model": used_model,
        "crop": [crop_top, crop_bottom],
        "tracker_mode": mode,
        "strategies": strategy_names,
        "source_hits": source_hits,
        "yolo_hits": source_hits.get("yolo", 0),
        "motion_hits": source_hits.get("motion", 0),
    }


def save_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 85) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
