"""Ultralytics YOLO / RT-DETR tracker (optional neural net)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from ramscout.paths import app_dir, bundle_root, models_dirs
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import plausible_robot_size

log = logging.getLogger(__name__)

_DEFAULT_WEIGHT = "yolo11n.pt"
_WEIGHT_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
_FALLBACK_CLASSES = [0, 2, 3, 5, 7, 32, 36]
# FRC-robot fine-tuned weights: point at a .pt (or .onnx) file whose model has
# a "robot" class. Downloaded once into the models dir as robot.pt.
ROBOT_WEIGHTS_ENV = ("ROBOSCOUT_ROBOT_WEIGHTS_URL", "RAMSCOUT_ROBOT_WEIGHTS_URL")
ROBOT_WEIGHTS_PATH_ENV = ("ROBOSCOUT_ROBOT_WEIGHTS", "RAMSCOUT_ROBOT_WEIGHTS")
_ROBOT_WEIGHT = "robot.pt"


def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401

        return True
    except Exception:
        return False


def robot_weights_url() -> str:
    import os

    for key in ROBOT_WEIGHTS_ENV:
        val = (os.environ.get(key) or "").strip()
        if val:
            return val
    return ""


def robot_weights_path() -> Path | None:
    """Explicit FRC-robot weights (env path) or a cached robot.pt in a models dir."""
    import os

    for key in ROBOT_WEIGHTS_PATH_ENV:
        val = (os.environ.get(key) or "").strip()
        if val and Path(val).is_file():
            return Path(val)
    for folder in models_dirs():
        for name in ("robot.pt", "robots.pt", "robot.onnx", "yolo11n-robot.pt"):
            candidate = folder / name
            if candidate.is_file() and candidate.stat().st_size > 1000:
                return candidate
    return None


def ensure_robot_weights() -> Path | None:
    """Download FRC-robot fine-tuned weights from the configured URL (once)."""
    existing = robot_weights_path()
    if existing is not None:
        return existing
    url = robot_weights_url()
    if not url:
        return None
    dest = app_dir() / "models" / _ROBOT_WEIGHT
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(url, tmp)
        if tmp.is_file() and tmp.stat().st_size > 1000:
            tmp.replace(dest)
            return dest
        tmp.unlink(missing_ok=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("FRC robot weights download failed (%s)", exc)
    return None


def find_local_model(search_dirs: list[Path] | None = None) -> Path | None:
    dirs = search_dirs or models_dirs()
    explicit = robot_weights_path()
    if explicit is not None and search_dirs is None:
        return explicit
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
    if not ultralytics_available():
        return None
    robot = ensure_robot_weights()
    if robot is not None:
        return robot
    existing = find_local_model()
    if existing is not None:
        return existing
    dest = app_dir() / "models" / _DEFAULT_WEIGHT
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import urllib.request

        urllib.request.urlretrieve(_WEIGHT_URL, dest)
        if dest.is_file() and dest.stat().st_size > 1000:
            return dest
    except Exception as exc:  # noqa: BLE001
        log.warning("Direct weight download failed (%s)", exc)
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


class YoloTracker:
    name = "yolo"
    kind = "neural"
    description = "Ultralytics YOLO/ByteTrack when installed (best local accuracy)"

    def __init__(self, model_path: str | None = None) -> None:
        self.model_path = model_path
        self.model = None
        self.family = "YOLO"
        self.class_filter: list[int] | None = None
        self.warnings: list[str] = []
        self._next_fallback_id = 1000
        # True when the loaded model has a dedicated "robot" class. COCO
        # models (person / car / ...) are useless as *proposers* for FRC
        # robots, so the ensemble uses them for verification only.
        self.robot_tuned = False

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return ultralytics_available()

    def reset(self) -> None:
        self.model = None
        self.class_filter = None
        self.warnings = []
        self._next_fallback_id = 1000
        self.robot_tuned = False

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        if not ultralytics_available():
            return False
        weights = self.model_path or (str(ensure_detector_weights() or "") or None)
        if not weights:
            local = find_local_model()
            weights = str(local) if local else None
        if not weights:
            self.warnings.append("YOLO requested but no detector weights were available.")
            return False
        try:
            self.model, self.family = load_detector(weights)
            self.class_filter = robot_class_ids(self.model)
            self.robot_tuned = self.class_filter is not None
            if self.class_filter is None:
                names = self.model.names or {}
                self.class_filter = [c for c in _FALLBACK_CLASSES if c in names]
                self.warnings.append(
                    f"{self.family} has no dedicated 'robot' class — COCO boxes are used to verify local "
                    "proposals only. Set ROBOSCOUT_ROBOT_WEIGHTS_URL to FRC-tuned weights for a real robot detector."
                )
            return True
        except Exception as exc:  # noqa: BLE001
            self.warnings.append(f"Could not load YOLO detector ({exc}).")
            self.model = None
            return False

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        if not self._ensure_model() or self.model is None:
            return []
        try:
            kwargs: dict[str, Any] = {
                "source": cropped,
                "persist": True,
                "tracker": "bytetrack.yaml",
                "verbose": False,
                "conf": 0.18,
                "iou": 0.45,
            }
            if self.class_filter:
                kwargs["classes"] = self.class_filter
            results = self.model.track(**kwargs)
        except Exception as exc:  # noqa: BLE001
            if not self.warnings or "YOLO tracking error" not in self.warnings[-1]:
                self.warnings.append(f"YOLO tracking error ({exc}).")
            return []

        out: list[Detection] = []
        if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
            return out
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(len(xyxy)) + self._next_fallback_id
            self._next_fallback_id += len(xyxy)
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.full(len(xyxy), 0.5)
        for box, tid, conf in zip(xyxy, ids, confs):
            x1, y1, x2, y2 = [float(v) for v in box]
            if not plausible_robot_size(x2 - x1, y2 - y1, ctx.crop_w, ctx.crop_h):
                continue
            out.append(
                Detection(
                    track_id=int(tid),
                    bbox=[x1, y1, x2, y2],
                    source=self.name,
                    confidence=float(conf),
                )
            )
        return out
