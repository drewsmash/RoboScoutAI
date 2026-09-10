"""Robot detection and tracking on a recorded match video.

Uses Ultralytics YOLO when it is installed and weights are available.
Otherwise (and as a supplement) uses OpenCV motion / background-subtraction
blobs so scouting still gets field paths — including in slim desktop builds
that intentionally omit PyTorch/ultralytics.
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


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


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
    """Return a usable .pt path when Ultralytics can load it.

    Slim desktop builds omit ultralytics/torch on purpose — skip the download
    and let OpenCV motion tracking handle paths instead.
    """
    if not ultralytics_available():
        return None

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
        for candidate in [Path(_DEFAULT_WEIGHT).resolve(), Path.cwd() / _DEFAULT_WEIGHT, dest]:
            if candidate.is_file() and candidate.stat().st_size > 1000:
                if candidate.resolve() != dest.resolve():
                    dest.write_bytes(candidate.read_bytes())
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


class MotionTracker:
    """OpenCV motion + background-subtraction tracker for broadcast FOV robots."""

    def __init__(self) -> None:
        self.prev_gray: np.ndarray | None = None
        self.subtractor = None
        self.tracks: dict[int, dict[str, float]] = {}
        self.next_id = 9000
        self.warm_frames = 0

    def detect(self, cropped: np.ndarray, frame_w: int, frame_h: int) -> list[dict[str, Any]]:
        import cv2

        if self.subtractor is None:
            self.subtractor = cv2.createBackgroundSubtractorMOG2(
                history=90,
                varThreshold=24,
                detectShadows=False,
            )

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        fg = self.subtractor.apply(cropped, learningRate=0.02 if self.warm_frames < 12 else 0.005)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            delta = cv2.absdiff(self.prev_gray, gray)
            _, diff = cv2.threshold(delta, 16, 255, cv2.THRESH_BINARY)
            mask = cv2.bitwise_or(fg, diff)
        else:
            mask = fg

        self.prev_gray = gray
        self.warm_frames += 1

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centroids: list[tuple[float, float, list[float]]] = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not _plausible_robot_size(float(w), float(h), frame_w, frame_h):
                continue
            cx, cy = x + w * 0.5, y + h * 0.5
            centroids.append((cx, cy, [float(x), float(y), float(x + w), float(y + h)]))

        # Prefer larger blobs first (robots over scorebug flicker).
        centroids.sort(key=lambda row: (row[2][2] - row[2][0]) * (row[2][3] - row[2][1]), reverse=True)
        centroids = centroids[:8]

        match_radius = max(56.0, min(frame_w, frame_h) * 0.08)
        used_tracks: set[int] = set()
        assigned: list[dict[str, Any]] = []
        for cx, cy, bbox in centroids:
            best_id = None
            best_dist = match_radius
            for tid, state in self.tracks.items():
                if tid in used_tracks:
                    continue
                dist = ((state["x"] - cx) ** 2 + (state["y"] - cy) ** 2) ** 0.5
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used_tracks.add(best_id)
            self.tracks[best_id] = {"x": cx, "y": cy, "age": 0.0}
            assigned.append({"track_id": best_id, "bbox": bbox, "source": "motion"})

        stale = [tid for tid in self.tracks if tid not in used_tracks]
        for tid in stale:
            self.tracks[tid]["age"] = self.tracks[tid].get("age", 0.0) + 1.0
            if self.tracks[tid]["age"] > 10:
                self.tracks.pop(tid, None)

        return assigned


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
) -> dict[str, Any]:
    """Run detection + tracking. Returns field-space samples and warnings.

    Always attempts OpenCV motion tracking so paths are produced even when
    Ultralytics/YOLO is not installed (desktop slim builds).
    """
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
    can_yolo = (not motion_only) and ultralytics_available()
    resolved_weights = None
    if can_yolo:
        resolved_weights = model_path or (str(ensure_detector_weights() or "") or None)
    elif not motion_only:
        warnings.append(
            "Ultralytics is not installed — using OpenCV motion tracking for robot paths. "
            "Install with `pip install ultralytics` (or drop a robot .pt in models/) for better detection."
        )

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
            warnings.append(f"Could not load detector ({exc}). Using OpenCV motion tracking for paths.")
            model = None

    samples: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0
    motion = MotionTracker()
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

        # Always refresh motion; use it when YOLO is thin or absent.
        motion_dets = motion.detect(cropped, crop_w, crop_h)
        if not detections:
            detections = motion_dets
            motion_hits += len(motion_dets)
        elif len(detections) < 4:
            before = len(detections)
            detections = _merge_detections(detections, motion_dets)
            motion_hits += max(0, len(detections) - before)

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
        warnings.append(
            "No robot tracks found in the video crop. Recalibrate the four field corners, "
            "widen the field crop (Advanced), or upload a clearer wide-angle VOD."
        )
    elif yolo_hits == 0 and motion_hits > 0:
        warnings.append("Tracked robots with OpenCV motion (no YOLO). Paths are approximate — confirm before pick lists.")

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
    if bw <= 3 or bh <= 3:
        return False
    # Reject tiny noise and huge scorebug/overlay boxes.
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
