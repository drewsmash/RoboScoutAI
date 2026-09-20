"""Monocular depth for camera-angle-aware field projection.

Backend chain (first that works wins, per process):

1. ``onnx``     – Depth Anything V2 Small exported to ONNX, run with
                  ``onnxruntime`` (CPU). ~100 MB weights are downloaded once
                  to the per-user models folder. This is what frozen desktop
                  builds use: real neural depth with no PyTorch.
2. ``torch``    – Depth Anything V2 Small through Hugging Face Transformers
                  (``torch`` + ``transformers`` + ``Pillow``). GPU/MPS when present.
3. ``classical`` – row-gradient / texture prior so BEV still gets a camera
                   tilt cue when no weights or runtime are available.

``depth_backends()`` reports which backends are importable, which one is
active, and *why* the others were skipped (exact missing module names) so the
UI and job metadata can show it instead of silently degrading.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

# Depth-Anything-V2-Small is Apache-2.0 and light enough for optional use.
_HF_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"
_ONNX_REPO = "onnx-community/depth-anything-v2-small"
_ONNX_FILE = "onnx/model.onnx"
_ONNX_URL = f"https://huggingface.co/{_ONNX_REPO}/resolve/main/{_ONNX_FILE}"
_ONNX_LOCAL_NAME = "depth-anything-v2-small.onnx"
_ONNX_MIN_BYTES = 50_000_000  # the fp32 export is ~99 MB; anything smaller is a broken download
_ONNX_INPUT = 518  # DINOv2 patch 14 → 518 = 37 × 14
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_LOCK = threading.RLock()
_PIPE: Any = None
_PIPE_FAILED = False
_ONNX_SESSION: Any = None
_ONNX_FAILED = False
_ONNX_INPUT_NAME = "pixel_values"
_ACTIVE_BACKEND: str | None = None
_BACKEND_NOTES: dict[str, str] = {}


@dataclass
class DepthResult:
    """Relative depth map + derived camera-angle cues."""

    depth: np.ndarray  # HxW float32, larger ≈ closer (V2 convention)
    source: str  # "depth_anything_v2_onnx" | "depth_anything_v2" | "classical"
    pitch_deg: float
    tilt_strength: float  # 0..1 how strongly the camera looks down on the field
    detail: str

    @property
    def neural(self) -> bool:
        return self.source != "classical"

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("depth", None)
        data["shape"] = list(self.depth.shape)
        data["depth_min"] = float(np.min(self.depth)) if self.depth.size else 0.0
        data["depth_max"] = float(np.max(self.depth)) if self.depth.size else 0.0
        data["neural"] = self.neural
        return data


# --------------------------------------------------------------------------- status


def models_dir() -> Path:
    """Writable per-user folder for downloaded weights.

    ``%LOCALAPPDATA%\\RoboScoutAI\\models`` on Windows, ``~/.local/share/RoboScoutAI/models``
    on Linux, ``~/Library/Application Support/RoboScoutAI/models`` on macOS.
    ``ROBOSCOUT_MODELS`` overrides.
    """
    override = (os.environ.get("ROBOSCOUT_MODELS") or "").strip()
    if override:
        path = Path(override)
    else:
        from ramscout.paths import user_data_root

        path = user_data_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _module_missing(name: str) -> str | None:
    try:
        __import__(name)
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc) or f"No module named '{name}'"


def onnx_model_path() -> Path:
    return models_dir() / _ONNX_LOCAL_NAME


def onnx_model_cached() -> bool:
    path = onnx_model_path()
    return path.is_file() and path.stat().st_size >= _ONNX_MIN_BYTES


def depth_backends() -> dict[str, Any]:
    """Importability of every backend + which one is active right now."""
    onnx_missing = _module_missing("onnxruntime")
    torch_missing = _module_missing("torch")
    tf_missing = _module_missing("transformers")
    pil_missing = _module_missing("PIL")
    torch_problems = [m for m in (torch_missing, tf_missing, pil_missing) if m]
    backends = {
        "onnx": {
            "label": "Depth Anything V2 Small (ONNX Runtime)",
            "importable": onnx_missing is None,
            "model_cached": onnx_model_cached(),
            "model_path": str(onnx_model_path()),
            "model_url": _ONNX_URL,
            "missing": onnx_missing,
            "note": _BACKEND_NOTES.get("onnx"),
        },
        "torch": {
            "label": "Depth Anything V2 Small (transformers + torch)",
            "importable": not torch_problems,
            "missing": "; ".join(torch_problems) if torch_problems else None,
            "note": _BACKEND_NOTES.get("torch"),
        },
        "classical": {
            "label": "Classical row/texture prior",
            "importable": True,
            "missing": None,
            "note": None,
        },
    }
    return {
        "active": _ACTIVE_BACKEND,
        "preferred_order": ["onnx", "torch", "classical"],
        "backends": backends,
        "install_hints": {
            "onnx": "pip install onnxruntime  (weights auto-download on first use)",
            "torch": "pip install -r requirements-depth.txt  (CPU wheels: --index-url https://download.pytorch.org/whl/cpu)",
        },
    }


def depth_available() -> bool:
    """True when some neural Depth Anything V2 backend can run."""
    status = depth_backends()["backends"]
    if status["onnx"]["importable"] and not _ONNX_FAILED:
        return True
    return bool(status["torch"]["importable"] and not _PIPE_FAILED)


def active_backend() -> str | None:
    return _ACTIVE_BACKEND


def reset_backends() -> None:
    """Forget cached sessions / failures (tests, or after installing deps)."""
    global _PIPE, _PIPE_FAILED, _ONNX_SESSION, _ONNX_FAILED, _ACTIVE_BACKEND
    with _LOCK:
        _PIPE = None
        _PIPE_FAILED = False
        _ONNX_SESSION = None
        _ONNX_FAILED = False
        _ACTIVE_BACKEND = None
        _BACKEND_NOTES.clear()


# ------------------------------------------------------------------------ estimate


def estimate_depth(frame_bgr: np.ndarray, *, prefer_neural: bool = True) -> DepthResult:
    """Estimate a relative depth map for a BGR frame."""
    global _ACTIVE_BACKEND
    if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
        empty = np.zeros((1, 1), dtype=np.float32)
        return DepthResult(empty, "classical", 0.0, 0.0, "Empty frame.")

    if prefer_neural:
        neural = _try_onnx(frame_bgr)
        if neural is not None:
            _ACTIVE_BACKEND = "onnx"
            pitch, tilt = _pitch_from_depth(neural)
            return DepthResult(
                depth=neural,
                source="depth_anything_v2_onnx",
                pitch_deg=round(pitch, 2),
                tilt_strength=round(tilt, 3),
                detail="Depth Anything V2 Small (ONNX Runtime) relative depth — camera-angle cues for BEV.",
            )
        neural = _try_depth_anything(frame_bgr)
        if neural is not None:
            _ACTIVE_BACKEND = "torch"
            pitch, tilt = _pitch_from_depth(neural)
            return DepthResult(
                depth=neural,
                source="depth_anything_v2",
                pitch_deg=round(pitch, 2),
                tilt_strength=round(tilt, 3),
                detail="Depth Anything V2 Small (transformers) relative depth — camera-angle cues for BEV.",
            )

    classical = _classical_depth(frame_bgr)
    if prefer_neural and _ACTIVE_BACKEND is None:
        _ACTIVE_BACKEND = "classical"
    pitch, tilt = _pitch_from_depth(classical)
    why = _classical_reason() if prefer_neural else ""
    return DepthResult(
        depth=classical,
        source="classical",
        pitch_deg=round(pitch, 2),
        tilt_strength=round(tilt, 3),
        detail=("Classical relative depth — tilt from field gradient." + (f" ({why})" if why else "")),
    )


def _classical_reason() -> str:
    notes = [f"{k}: {v}" for k, v in _BACKEND_NOTES.items() if v]
    if notes:
        return "; ".join(notes)
    status = depth_backends()["backends"]
    missing = []
    if status["onnx"]["missing"]:
        missing.append(f"onnxruntime missing")
    if status["torch"]["missing"]:
        missing.append(status["torch"]["missing"])
    return "; ".join(missing) if missing else "no neural backend"


def sample_depth(depth: np.ndarray, x: float, y: float) -> float:
    """Bilinear sample; returns 0 outside bounds."""
    if depth is None or depth.size == 0:
        return 0.0
    h, w = depth.shape[:2]
    if w < 2 or h < 2:
        return float(depth.flat[0]) if depth.size else 0.0
    xf = float(np.clip(x, 0, w - 1.001))
    yf = float(np.clip(y, 0, h - 1.001))
    x0, y0 = int(xf), int(yf)
    x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
    tx, ty = xf - x0, yf - y0
    v00 = float(depth[y0, x0])
    v10 = float(depth[y0, x1])
    v01 = float(depth[y1, x0])
    v11 = float(depth[y1, x1])
    return (1 - ty) * ((1 - tx) * v00 + tx * v10) + ty * ((1 - tx) * v01 + tx * v11)


def refine_foot_point(
    bbox: list[float] | tuple[float, float, float, float],
    depth: np.ndarray | None,
    *,
    crop_y0: float = 0.0,
) -> tuple[float, float]:
    """Pick a ground-contact foot using depth inside the detection box.

    Uses the lower band of the bbox and prefers the deepest (closest-to-camera /
    lowest-on-field) contact along the bottom edge — better than raw (cx, y2)
    when the camera is angled.
    """
    x1, y1, x2, y2 = [float(v) for v in bbox]
    fx = (x1 + x2) * 0.5
    fy = y2 + crop_y0
    if depth is None or depth.size == 0:
        return fx, fy

    h, w = depth.shape[:2]
    bx1 = int(np.clip(x1, 0, w - 1))
    bx2 = int(np.clip(x2, 0, w - 1))
    by1 = int(np.clip(y1, 0, h - 1))
    by2 = int(np.clip(y2, 0, h - 1))
    if bx2 <= bx1 or by2 <= by1:
        return fx, fy

    band_top = max(by1, by2 - max(2, (by2 - by1) // 3))
    band = depth[band_top : by2 + 1, bx1 : bx2 + 1]
    if band.size == 0:
        return fx, fy

    # Larger depth ⇒ closer to camera in V2; for top-down angled cams the
    # near-ground contact sits toward the bottom of the box with high depth.
    flat = band.reshape(-1)
    idx = int(np.argmax(flat))
    local_y, local_x = divmod(idx, band.shape[1])
    foot_x = bx1 + local_x
    foot_y = band_top + local_y
    # Blend toward geometric bottom-center so noisy depth does not yank feet.
    blend = 0.55
    fx = (1 - blend) * fx + blend * float(foot_x)
    fy = (1 - blend) * fy + blend * float(foot_y + crop_y0)
    return fx, fy


# ------------------------------------------------------------------- ONNX backend


def ensure_onnx_model(*, download: bool = True, on_progress: Any = None) -> Path | None:
    """Return the cached ONNX weights, downloading them once when allowed."""
    path = onnx_model_path()
    if onnx_model_cached():
        return path
    if not download or (os.environ.get("ROBOSCOUT_NO_DOWNLOAD") or "").strip() in {"1", "true", "yes"}:
        return None
    tmp = path.with_suffix(".part")
    try:
        import urllib.request

        log.info("Downloading Depth Anything V2 Small ONNX weights → %s", path)
        with urllib.request.urlopen(_ONNX_URL, timeout=60) as resp, open(tmp, "wb") as out:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if on_progress and total:
                    on_progress("Downloading depth model…", 100.0 * done / total)
        if tmp.stat().st_size < _ONNX_MIN_BYTES:
            raise RuntimeError(f"download too small ({tmp.stat().st_size} bytes)")
        tmp.replace(path)
        return path
    except Exception as exc:  # noqa: BLE001
        _BACKEND_NOTES["onnx"] = f"weights download failed: {exc}"
        log.info("Depth Anything V2 ONNX weights unavailable (%s).", exc)
        try:
            tmp.unlink(missing_ok=True)
        except Exception:  # noqa: BLE001
            pass
        return None


def _get_onnx_session() -> Any:
    global _ONNX_SESSION, _ONNX_FAILED, _ONNX_INPUT_NAME
    with _LOCK:
        if _ONNX_SESSION is not None:
            return _ONNX_SESSION
        if _ONNX_FAILED:
            return None
        try:
            import onnxruntime as ort  # type: ignore
        except Exception as exc:  # noqa: BLE001
            _BACKEND_NOTES["onnx"] = f"onnxruntime not installed ({exc})"
            _ONNX_FAILED = True
            return None
        path = ensure_onnx_model()
        if path is None:
            _ONNX_FAILED = True
            _BACKEND_NOTES.setdefault("onnx", "weights not cached and download disabled")
            return None
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = max(1, min(4, os.cpu_count() or 1))
            opts.log_severity_level = 3
            providers = ["CPUExecutionProvider"]
            avail = set(ort.get_available_providers())
            if "CUDAExecutionProvider" in avail:
                providers.insert(0, "CUDAExecutionProvider")
            elif "CoreMLExecutionProvider" in avail:
                providers.insert(0, "CoreMLExecutionProvider")
            elif "DmlExecutionProvider" in avail:
                providers.insert(0, "DmlExecutionProvider")
            sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
            _ONNX_INPUT_NAME = sess.get_inputs()[0].name
            _ONNX_SESSION = sess
            _BACKEND_NOTES.pop("onnx", None)
            return sess
        except Exception as exc:  # noqa: BLE001
            _BACKEND_NOTES["onnx"] = f"session failed: {exc}"
            log.info("Could not create ONNX depth session: %s", exc)
            _ONNX_FAILED = True
            return None


def _preprocess(frame_bgr: np.ndarray, size: int = _ONNX_INPUT) -> np.ndarray:
    import cv2

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) if frame_bgr.ndim == 3 else cv2.cvtColor(frame_bgr, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    normed = (resized - _IMAGENET_MEAN) / _IMAGENET_STD
    return normed.transpose(2, 0, 1)[None].astype(np.float32)


def _postprocess(raw: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    import cv2

    depth = np.asarray(raw, dtype=np.float32)
    while depth.ndim > 2:
        depth = depth[0]
    if depth.shape[:2] != (out_h, out_w):
        depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin > 1e-6:
        depth = (depth - dmin) / (dmax - dmin)
    return depth.astype(np.float32)


def _try_onnx(frame_bgr: np.ndarray) -> np.ndarray | None:
    global _ONNX_FAILED
    if _ONNX_FAILED:
        return None
    sess = _get_onnx_session()
    if sess is None:
        return None
    try:
        h, w = frame_bgr.shape[:2]
        x = _preprocess(frame_bgr)
        out = sess.run(None, {_ONNX_INPUT_NAME: x})[0]
        return _postprocess(out, h, w)
    except Exception as exc:  # noqa: BLE001
        _BACKEND_NOTES["onnx"] = f"inference failed: {exc}"
        log.info("ONNX depth inference failed (%s) — trying transformers / classical.", exc)
        _ONNX_FAILED = True
        return None


# ------------------------------------------------------------------ torch backend


def _try_depth_anything(frame_bgr: np.ndarray) -> np.ndarray | None:
    global _PIPE, _PIPE_FAILED
    if _PIPE_FAILED:
        return None
    try:
        import cv2

        try:
            from PIL import Image
        except ImportError:
            _BACKEND_NOTES["torch"] = "No module named 'PIL' (pip install Pillow)"
            log.info(
                "Depth Anything V2 unavailable (No module named 'PIL') — "
                "install with: pip install Pillow — using classical depth."
            )
            _PIPE_FAILED = True
            _PIPE = None
            return None

        pipe = _get_pipe()
        if pipe is None:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Cap long edge so optional inference stays interactive.
        h, w = rgb.shape[:2]
        scale = min(1.0, 768.0 / max(h, w))
        if scale < 0.999:
            rgb_small = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            rgb_small = rgb
        image = Image.fromarray(rgb_small)
        out = pipe(image)
        depth_img = out["depth"] if isinstance(out, dict) else out
        depth = np.asarray(depth_img, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[..., 0]
        return _postprocess(depth, h, w)
    except Exception as exc:  # noqa: BLE001
        hint = ""
        if "PIL" in str(exc) or "Pillow" in str(exc):
            hint = " — install with: pip install Pillow"
        _BACKEND_NOTES["torch"] = f"{exc}"
        log.info(
            "Depth Anything V2 unavailable (%s)%s — using classical depth.",
            exc,
            hint,
        )
        _PIPE_FAILED = True
        _PIPE = None
        return None


def _get_pipe() -> Any:
    global _PIPE, _PIPE_FAILED
    with _LOCK:
        if _PIPE is not None:
            return _PIPE
        if _PIPE_FAILED:
            return None
        try:
            from transformers import pipeline

            device = _pick_device()
            _PIPE = pipeline(
                task="depth-estimation",
                model=_HF_MODEL,
                device=device,
            )
            _BACKEND_NOTES.pop("torch", None)
            return _PIPE
        except Exception as exc:  # noqa: BLE001
            _BACKEND_NOTES["torch"] = f"{exc}"
            log.info("Could not load Depth Anything V2 pipeline: %s", exc)
            _PIPE_FAILED = True
            return None


def _pick_device() -> int | str:
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return -1


# --------------------------------------------------------------- classical prior


def _classical_depth(frame_bgr: np.ndarray) -> np.ndarray:
    """Cheap relative depth: lower image rows + soft focus on carpet texture."""
    import cv2

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    h, w = gray.shape[:2]
    yy = np.linspace(0.15, 1.0, h, dtype=np.float32).reshape(-1, 1)
    row = np.repeat(yy, w, axis=1)
    # Texture: flatter carpet farther up often has lower local variance.
    blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 3)
    local = cv2.GaussianBlur((gray.astype(np.float32) - blur) ** 2, (0, 0), 5)
    local = local / (float(local.max()) + 1e-6)
    depth = 0.72 * row + 0.28 * (1.0 - local)
    depth = depth.astype(np.float32)
    dmin, dmax = float(depth.min()), float(depth.max())
    if dmax - dmin > 1e-6:
        depth = (depth - dmin) / (dmax - dmin)
    return depth


def _pitch_from_depth(depth: np.ndarray) -> tuple[float, float]:
    """Estimate camera pitch (degrees looking down) from vertical depth gradient."""
    if depth is None or depth.size < 16:
        return 35.0, 0.45
    h, w = depth.shape[:2]
    # Center strip avoids scoreboard / sideline clutter.
    x0, x1 = int(w * 0.25), int(w * 0.75)
    strip = depth[:, x0:x1]
    row_mean = np.mean(strip, axis=1)
    # Fit depth vs normalized row index.
    ys = np.linspace(0.0, 1.0, h, dtype=np.float64)
    slope = float(np.polyfit(ys, row_mean.astype(np.float64), 1)[0])
    # Positive slope ⇒ bottom closer ⇒ typical overhead/angled broadcast.
    tilt = float(np.clip(abs(slope) / 1.2, 0.0, 1.0))
    # Map to a plausible FRC broadcast pitch range (~20–55° from horizontal).
    pitch = 22.0 + tilt * 33.0
    if slope < 0:
        pitch = max(15.0, 50.0 - pitch * 0.35)
    return pitch, tilt


def ground_plane_pitch(depth: np.ndarray, *, field_mask: np.ndarray | None = None) -> tuple[float, float]:
    """Pitch from a plane fit to (row, depth) over carpet pixels.

    A camera looking straight down produces near-constant depth over the
    floor; an oblique camera produces a strong linear depth-vs-row ramp. The
    ratio of ramp to spread is a decent stand-in for tilt without intrinsics.
    Returns (pitch_deg, tilt_strength) like :func:`_pitch_from_depth`.
    """
    if depth is None or depth.size < 64:
        return _pitch_from_depth(depth)
    h, w = depth.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w]
    if field_mask is not None and field_mask.shape[:2] == depth.shape[:2] and np.count_nonzero(field_mask) > 64:
        sel = field_mask.astype(bool)
    else:
        sel = np.ones_like(depth, dtype=bool)
        sel[: int(h * 0.15)] = False
    d = depth[sel].astype(np.float64)
    yn = (ys[sel] / max(h - 1, 1)).astype(np.float64)
    xn = (xs[sel] / max(w - 1, 1)).astype(np.float64)
    A = np.stack([yn, xn, np.ones_like(yn)], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(A, d, rcond=None)
    except Exception:  # noqa: BLE001
        return _pitch_from_depth(depth)
    slope_y = float(coef[0])
    spread = float(np.std(d)) + 1e-6
    tilt = float(np.clip(abs(slope_y) / (1.2 + 0.5 * spread), 0.0, 1.0))
    pitch = 22.0 + tilt * 33.0
    if slope_y < 0:
        pitch = max(15.0, 50.0 - pitch * 0.35)
    return pitch, tilt
