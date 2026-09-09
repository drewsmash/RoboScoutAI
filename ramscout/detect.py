"""Robot detection and tracking on a recorded match video.

Uses Ultralytics YOLO when weights are available (auto-downloads a nano COCO
model into models/), and falls back to OpenCV motion blobs so scouting still
gets field paths when the neural detector misses robots.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.geometry import crop_bounds, default_source_points, homography_from_corners, project_points
from ramscout.identity import bumper_alliance, constrain_to_teams, ocr_digits
from ramscout.paths import app_dir, bundle_root, models_dirs

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

_DEFAULT_WEIGHT = "yolo11n.pt"
_WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"

# COCO classes that often fire on FRC robots from a wide broadcast camera.
_FALLBACK_CLASSES = [0, 2, 3, 5, 7, 32, 36]  # person, car, motorcycle, bus, truck, sports ball, skateboard


def find_local_model(search_dirs: list[Path] | None = None) -> Path | None:
    dirs = search_dirs or models_dirs()
    for folder in dirs:
        if not folder.is_dir():
            continue
        for name in ("robot.pt", "robots.pt", "yolo11n-robot.pt", "rtdetr-l.pt", "yolo11n.pt", "yolov8n.pt"):
            candidate = folder / name
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate
        for candidate in sorted(folder.glob("*.pt")):
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate
    return None


def ensure_detector_weights() -> Path | None:
    """Return a usable .pt path, downloading YOLO nano into models/ if needed."""
    existing = find_local_model()
    if existing is not None:
        return existing

    targets = [
        app_dir() / "models" / _DEFAULT_WEIGHT,
        bundle_root() / "models" / _DEFAULT_WEIGHT,
        Path.cwd() / "models" / _DEFAULT_WEIGHT,
        Path.cwd() / _DEFAULT_WEIGHT,
    ]
    dest = targets[0]
    dest.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading default detector weights to %s", dest)
    try:
        urllib.request.urlretrieve(_WEIGHT_URL, dest)
        if dest.is_file() and dest.stat().st_size > 1000:
            return dest
    except Exception as exc:  # noqa: BLE001
        log.warning("Direct weight download failed (%s); trying Ultralytics", exc)

    try:
        from ultralytics import YOLO

        model = YOLO(_DEFAULT_WEIGHT)
        # Ultralytics may have cached/downloaded next to CWD.
        for candidate in [Path(_DEFAULT_WEIGHT).resolve(), Path.cwd() / _DEFAULT_WEIGHT, dest]:
            if candidate.is_file() and candidate.stat().st_size > 1000:
                if candidate.resolve() != dest.resolve():
                    dest.write_bytes(candidate.read_bytes())
                # Touch the model so weights stay warm.
                _ = model.names
                return dest if dest.is_file() else candidate
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not obtain detector weights: %s", exc)
    return find_local_model()


def load_detector(weights: str):
    from ultralytics import RTDETR, YOLO

    lower = str(weights).lower()
    if "rtdetr" in lower or "rt-detr" in lower:
        try:
            return RTDETR(weights), "RT-DETR"
        except Exception:
            pass
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
    crop_h = max(y1 - y0, 1)
    crop_w = max(frame_w, 1)

    warnings: list[str] = []
    model = None
    class_filter: list[int] | None = None
    resolved_weights = model_path or (str(ensure_detector_weights() or "") or None)
    if resolved_weights:
        try:
            model, family = load_detector(resolved_weights)
            class_filter = robot_class_ids(model)
            if class_filter is None:
                class_filter = [c for c in _FALLBACK_CLASSES if c in (model.names or {})]
                warnings.append(
                    f"{family} has no dedicated 'robot' class — using broadcast-friendly COCO classes "
                    f"plus motion fallback. Drop a robot-trained .pt in models/ for better results."
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Could not load detector ({exc}). Falling back to motion tracking.")
            model = None
    else:
        warnings.append("No detector weights available. Using OpenCV motion tracking only.")

    samples: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0
    motion_prev = None
    motion_tracks: dict[int, dict[str, float]] = {}
    next_motion_id = 9000
    yolo_hits = 0
    motion_hits = 0

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
        detections: list[dict[str, Any]] = []

        if model is not None:
            try:
                kwargs: dict[str, Any] = {
                    "source": cropped,
                    "persist": True,
                    "tracker": "bytetrack.yaml",
                    "verbose": False,
                    "conf": 0.18,
                    "iou": 0.45,
                }
                if class_filter:
                    kwargs["classes"] = class_filter
                results = model.track(**kwargs)
                if results and results[0].boxes is not None and results[0].boxes.xyxy is not None:
                    boxes = results[0].boxes
                    xyxy = boxes.xyxy.cpu().numpy()
                    ids = (
                        boxes.id.cpu().numpy().astype(int)
                        if boxes.id is not None
                        else np.arange(len(xyxy)) + 1
                    )
                    for box, tid in zip(xyxy, ids):
                        x1, y1b, x2, y2 = [float(v) for v in box]
                        bw, bh = x2 - x1, y2 - y1b
                        if not _plausible_robot_size(bw, bh, crop_w, crop_h):
                            continue
                        detections.append(
                            {
                                "track_id": int(tid),
                                "bbox": [x1, y1b, x2, y2],
                                "source": "yolo",
                            }
                        )
                        yolo_hits += 1
            except Exception as exc:  # noqa: BLE001
                if processed == 0:
                    warnings.append(f"YOLO tracking error ({exc}); relying on motion fallback.")

        if len(detections) < 2:
            motion_dets, motion_prev, motion_tracks, next_motion_id = _motion_detect(
                cropped,
                motion_prev,
                motion_tracks,
                next_motion_id,
                crop_w,
                crop_h,
            )
            # Prefer YOLO IDs; fill gaps with motion blobs.
            if not detections:
                detections = motion_dets
                motion_hits += len(motion_dets)
            elif len(detections) < 4:
                detections = _merge_detections(detections, motion_dets)
                motion_hits += max(0, len(detections) - yolo_hits)

        feet: list[list[float]] = []
        meta: list[tuple[int, str, str, list[float], float, float]] = []
        for det in detections:
            x1, y1b, x2, y2 = det["bbox"]
            fx = (x1 + x2) * 0.5
            fy = y2 + y0
            roi = cropped[max(int(y1b), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)]
            alliance = bumper_alliance(roi)
            if alliance == "unknown":
                alliance = "blue" if fx < frame_w * 0.5 else "red"
            team = ""
            if team_numbers:
                raw = ocr_digits(roi)
                team = constrain_to_teams(raw, team_numbers) or ""
            feet.append([fx, fy])
            meta.append((int(det["track_id"]), alliance, team, [x1, y1b + y0, x2, y2 + y0], fx, fy))

        if feet:
            mapped = project_points(feet, homography)
            for (mx, my), (tid, alliance, team, bbox, px, py) in zip(mapped, meta):
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
                    }
                )

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Tracking robots…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    if not samples:
        warnings.append("No robot tracks found in the video crop. Recalibrate the four field corners or widen the crop.")
    elif yolo_hits == 0 and motion_hits > 0:
        warnings.append("Used motion-blob tracking (YOLO found no robots). Paths are approximate.")

    return {
        "samples": samples,
        "warnings": warnings,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "first_frame": first,
        "homography": homography.tolist(),
        "used_model": bool(model is not None),
        "crop": [crop_top, crop_bottom],
        "yolo_hits": yolo_hits,
        "motion_hits": motion_hits,
    }


def save_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 85) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


def _plausible_robot_size(bw: float, bh: float, frame_w: int, frame_h: int) -> bool:
    if bw <= 4 or bh <= 4:
        return False
    # Reject tiny noise and huge scorebug/overlay boxes.
    if bw > frame_w * 0.35 or bh > frame_h * 0.55:
        return False
    if bw * bh < (frame_w * frame_h) * 0.0004:
        return False
    aspect = bw / max(bh, 1.0)
    return 0.35 <= aspect <= 3.5


def _motion_detect(
    cropped: np.ndarray,
    prev_gray: np.ndarray | None,
    tracks: dict[int, dict[str, float]],
    next_id: int,
    frame_w: int,
    frame_h: int,
) -> tuple[list[dict[str, Any]], np.ndarray, dict[int, dict[str, float]], int]:
    import cv2

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    detections: list[dict[str, Any]] = []
    if prev_gray is None or prev_gray.shape != gray.shape:
        return detections, gray, tracks, next_id

    delta = cv2.absdiff(prev_gray, gray)
    _, mask = cv2.threshold(delta, 18, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    centroids: list[tuple[float, float, list[float]]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not _plausible_robot_size(float(w), float(h), frame_w, frame_h):
            continue
        cx, cy = x + w * 0.5, y + h * 0.5
        centroids.append((cx, cy, [float(x), float(y), float(x + w), float(y + h)]))

    # Greedy match to existing motion tracks.
    used_tracks: set[int] = set()
    assigned: list[dict[str, Any]] = []
    for cx, cy, bbox in centroids:
        best_id = None
        best_dist = 80.0
        for tid, state in tracks.items():
            if tid in used_tracks:
                continue
            dist = ((state["x"] - cx) ** 2 + (state["y"] - cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_id = tid
        if best_id is None:
            best_id = next_id
            next_id += 1
        used_tracks.add(best_id)
        tracks[best_id] = {"x": cx, "y": cy, "age": 0}
        assigned.append({"track_id": best_id, "bbox": bbox, "source": "motion"})

    # Age out stale tracks.
    stale = [tid for tid, state in tracks.items() if tid not in used_tracks]
    for tid in stale:
        tracks[tid]["age"] = tracks[tid].get("age", 0) + 1
        if tracks[tid]["age"] > 8:
            tracks.pop(tid, None)

    return assigned, gray, tracks, next_id


def _merge_detections(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    min_dist: float = 40.0,
) -> list[dict[str, Any]]:
    merged = list(primary)
    for det in secondary:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
        conflict = False
        for existing in merged:
            ex1, ey1, ex2, ey2 = existing["bbox"]
            ecx, ecy = (ex1 + ex2) * 0.5, (ey1 + ey2) * 0.5
            if ((ecx - cx) ** 2 + (ecy - cy) ** 2) ** 0.5 < min_dist:
                conflict = True
                break
        if not conflict:
            merged.append(det)
    return merged
