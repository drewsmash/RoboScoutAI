"""Packaged local FRC robot detector via ONNX Runtime.

The desktop app must run detection without installing Ultralytics. This module
loads a single-class ``robot`` ONNX graph, reports the execution provider in
use, and refuses to claim FRC-ready weights unless metadata verifies them.

Weights provenance is explicit:
- ``models/robot.onnx`` + ``models/robot.onnx.meta.json`` when present
- Env ``ROBOSCOUT_ROBOT_ONNX`` / ``RAMSCOUT_ROBOT_ONNX`` for an explicit path
- Until a verified FRC checkpoint is shipped, ``frc_ready`` stays False
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ramscout.paths import models_dirs

log = logging.getLogger(__name__)

ONNX_ENV = ("ROBOSCOUT_ROBOT_ONNX", "RAMSCOUT_ROBOT_ONNX")
META_NAME = "robot.onnx.meta.json"
WEIGHT_NAME = "robot.onnx"


@dataclass
class DetectorMeta:
    version: str = ""
    class_names: list[str] = field(default_factory=lambda: ["robot"])
    input_size: tuple[int, int] = (640, 640)
    sha256: str = ""
    license: str = ""
    source: str = ""
    frc_ready: bool = False
    notes: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "class_names": list(self.class_names),
            "input_size": list(self.input_size),
            "sha256": self.sha256,
            "license": self.license,
            "source": self.source,
            "frc_ready": bool(self.frc_ready),
            "notes": self.notes,
        }


@dataclass
class OnnxDetection:
    bbox: list[float]  # x1,y1,x2,y2 in crop pixels
    score: float
    class_id: int = 0
    label: str = "robot"


def onnxruntime_available() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:
        return False


def find_robot_onnx() -> Path | None:
    import os

    for key in ONNX_ENV:
        val = (os.environ.get(key) or "").strip()
        if val and Path(val).is_file():
            return Path(val)
    for folder in models_dirs():
        candidate = folder / WEIGHT_NAME
        if candidate.is_file() and candidate.stat().st_size > 1000:
            return candidate
    return None


def load_meta(weights: Path | None) -> DetectorMeta:
    meta = DetectorMeta(
        notes="No verified FRC-trained checkpoint is packaged. Train/export workflow is supported; do not claim FRC-ready until meta.frc_ready is true."
    )
    if weights is None:
        return meta
    side = weights.with_name(weights.name + ".meta.json")
    alt = weights.parent / META_NAME
    for path in (side, alt):
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not read detector meta %s: %s", path, exc)
            continue
        meta.version = str(raw.get("version") or "")
        meta.class_names = list(raw.get("class_names") or ["robot"])
        size = raw.get("input_size") or [640, 640]
        meta.input_size = (int(size[0]), int(size[1]))
        meta.sha256 = str(raw.get("sha256") or "")
        meta.license = str(raw.get("license") or "")
        meta.source = str(raw.get("source") or "")
        meta.frc_ready = bool(raw.get("frc_ready"))
        meta.notes = str(raw.get("notes") or meta.notes)
        break
    if meta.sha256 and weights.is_file():
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest.lower() != meta.sha256.lower():
            meta.frc_ready = False
            meta.notes = f"Integrity check failed (expected {meta.sha256[:12]}…, got {digest[:12]}…). {meta.notes}"
    return meta


def letterbox(image: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Resize with padding; return (chw float32 0..1, pad LTRB, scale)."""
    th, tw = size[1], size[0]
    h, w = image.shape[:2]
    scale = min(tw / max(w, 1), th / max(h, 1))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    import cv2

    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    pad_l = (tw - nw) // 2
    pad_t = (th - nh) // 2
    canvas[pad_t : pad_t + nh, pad_l : pad_l + nw] = resized
    pad = (pad_l, pad_t, tw - nw - pad_l, th - nh - pad_t)
    blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return blob, pad, float(scale)


class OnnxRobotDetector:
    """Single-class robot detector. Alliance/identity are estimated separately."""

    def __init__(self, weights: Path | None = None) -> None:
        self.weights = weights or find_robot_onnx()
        self.meta = load_meta(self.weights)
        self.session = None
        self.provider = "none"
        self.input_name = "images"
        self._load()

    def _load(self) -> None:
        if self.weights is None or not onnxruntime_available():
            return
        try:
            import onnxruntime as ort

            available = ort.get_available_providers()
            preferred = []
            for name in ("CoreMLExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"):
                if name in available:
                    preferred.append(name)
            if not preferred:
                preferred = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(self.weights), providers=preferred)
            self.provider = self.session.get_providers()[0] if self.session.get_providers() else "CPUExecutionProvider"
            self.input_name = self.session.get_inputs()[0].name
        except Exception as exc:  # noqa: BLE001
            log.warning("ONNX detector load failed: %s", exc)
            self.session = None
            self.provider = "none"

    @property
    def available(self) -> bool:
        return self.session is not None

    def status(self) -> dict[str, Any]:
        return {
            "available": self.available,
            "provider": self.provider,
            "weights": str(self.weights) if self.weights else None,
            "onnxruntime": onnxruntime_available(),
            "meta": self.meta.as_dict(),
            "frc_ready": bool(self.meta.frc_ready and self.available),
        }

    def detect(self, frame_bgr: np.ndarray, *, conf: float = 0.25) -> list[OnnxDetection]:
        if self.session is None:
            return []
        blob, pad, scale = letterbox(frame_bgr, self.meta.input_size)
        outputs = self.session.run(None, {self.input_name: blob[None, ...]})
        return _parse_yolo_like(outputs[0], pad=pad, scale=scale, conf=conf, class_names=self.meta.class_names)


def _parse_yolo_like(
    out: np.ndarray,
    *,
    pad: tuple[int, int, int, int],
    scale: float,
    conf: float,
    class_names: list[str],
) -> list[OnnxDetection]:
    """Best-effort decode for Ultralytics-export ONNX (1, 4+nc, N) or (1, N, 4+nc)."""
    arr = np.asarray(out)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.shape[0] in {4, 5, 6} or (arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 84):
        arr = arr.T
    dets: list[OnnxDetection] = []
    pad_l, pad_t, _, _ = pad
    for row in arr:
        if row.shape[0] < 5:
            continue
        # xywh + objectness/class scores
        if row.shape[0] == 5 + len(class_names) or row.shape[0] >= 6:
            cx, cy, bw, bh = row[:4]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            score = float(scores[class_id])
        else:
            continue
        if score < conf:
            continue
        x1 = (float(cx) - float(bw) / 2.0 - pad_l) / max(scale, 1e-6)
        y1 = (float(cy) - float(bh) / 2.0 - pad_t) / max(scale, 1e-6)
        x2 = (float(cx) + float(bw) / 2.0 - pad_l) / max(scale, 1e-6)
        y2 = (float(cy) + float(bh) / 2.0 - pad_t) / max(scale, 1e-6)
        label = class_names[class_id] if 0 <= class_id < len(class_names) else "robot"
        dets.append(OnnxDetection(bbox=[x1, y1, x2, y2], score=score, class_id=class_id, label=label))
    return dets


def detector_status() -> dict[str, Any]:
    return OnnxRobotDetector().status()
