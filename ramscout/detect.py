"""Robot detection and tracking on a recorded match video."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.geometry import crop_bounds, default_source_points, homography_from_corners, project_points
from ramscout.identity import bumper_alliance, constrain_to_teams, ocr_digits

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def find_local_model(search_dirs: list[Path]) -> Path | None:
    for folder in search_dirs:
        if not folder.is_dir():
            continue
        for name in ("robot.pt", "robots.pt", "yolo11n-robot.pt", "rtdetr-l.pt", "yolo11n.pt"):
            candidate = folder / name
            if candidate.exists():
                return candidate
        for candidate in sorted(folder.glob("*.pt")):
            return candidate
    return None


def load_detector(weights: str):
    from ultralytics import RTDETR, YOLO

    try:
        return RTDETR(weights), "RT-DETR"
    except Exception:
        return YOLO(weights), "YOLO"


def robot_class_ids(model) -> list[int] | None:
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = [int(i) for i, name in items if "robot" in str(name).lower()]
    return ids or None


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
) -> dict[str, Any]:
    """Run detection + tracking. Returns field-space samples and warnings."""
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

    warnings: list[str] = []
    model = None
    class_filter = None
    if model_path:
        try:
            model, family = load_detector(model_path)
            class_filter = robot_class_ids(model)
            if class_filter is None:
                warnings.append(
                    f"{family} weights have no 'robot' class. Tracking may follow people or other objects. "
                    "Drop a robot-trained .pt in the project folder for better results."
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not load detector ({exc}). Path overlay will be empty until a model is available.")
            model = None

    samples: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0

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
        if model is None:
            processed += 1
            frame_index += 1
            continue

        kwargs: dict[str, Any] = {
            "source": cropped,
            "persist": True,
            "tracker": "bytetrack.yaml",
            "verbose": False,
            "conf": 0.25,
        }
        if class_filter is not None:
            kwargs["classes"] = class_filter
        results = model.track(**kwargs)
        if not results:
            processed += 1
            frame_index += 1
            continue
        boxes = results[0].boxes
        if boxes is None or boxes.xyxy is None:
            processed += 1
            frame_index += 1
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else np.full((len(xyxy),), -1)
        feet = []
        meta = []
        for box, tid in zip(xyxy, ids):
            if tid < 0:
                continue
            x1, y1b, x2, y2 = [float(v) for v in box]
            fx = (x1 + x2) * 0.5
            fy = y2 + y0
            roi = cropped[max(int(y1b), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)]
            alliance = bumper_alliance(roi)
            team = ""
            if team_numbers:
                raw = ocr_digits(roi)
                team = constrain_to_teams(raw, team_numbers) or ""
            feet.append([fx, fy])
            meta.append((int(tid), alliance, team, [x1, y1b + y0, x2, y2 + y0]))

        if feet:
            mapped = project_points(feet, homography)
            for (mx, my), (tid, alliance, team, bbox) in zip(mapped, meta):
                samples.append(
                    {
                        "t": round(float(t), 3),
                        "frame": frame_index,
                        "track_id": tid,
                        "x": float(mx),
                        "y": float(my),
                        "px": float(fx),
                        "py": float(fy),
                        "alliance": alliance,
                        "team": team,
                        "bbox": bbox,
                    }
                )

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Tracking robots…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    calibration_jpeg = first
    return {
        "samples": samples,
        "warnings": warnings,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "first_frame": calibration_jpeg,
        "homography": homography.tolist(),
        "used_model": bool(model is not None),
        "crop": [crop_top, crop_bottom],
    }


def save_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 85) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
