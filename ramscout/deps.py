"""Runtime dependency inventory for RoboScoutAI.

Required packages must import for the local app to start and track.
Optional packages unlock neural depth/training/cloud-adjacent features.

Run::

    python -m ramscout.deps
    python -m ramscout.deps --strict   # exit 1 if any required package is missing
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DepSpec:
    key: str
    import_name: str
    pip: str
    required: bool
    purpose: str
    install_extra: str = ""  # e.g. requirements-depth.txt


# Keep in sync with requirements.txt / requirements-desktop.txt.
REQUIRED: tuple[DepSpec, ...] = (
    DepSpec("fastapi", "fastapi", "fastapi>=0.115.0", True, "HTTP API"),
    DepSpec("uvicorn", "uvicorn", "uvicorn[standard]>=0.32.0", True, "ASGI server"),
    DepSpec("multipart", "python_multipart", "python-multipart>=0.0.12", True, "Video upload forms"),
    DepSpec("pydantic", "pydantic", "pydantic>=2.9.0", True, "Request models"),
    DepSpec("httpx", "httpx", "httpx>=0.27.0", True, "HTTP client (TBA / updates)"),
    DepSpec("numpy", "numpy", "numpy>=1.26.0", True, "Arrays / geometry"),
    DepSpec("scipy", "scipy", "scipy>=1.11.0", True, "Assignment / smoothing"),
    DepSpec("opencv", "cv2", "opencv-python-headless>=4.10.0", True, "Video + OpenCV tracking"),
    DepSpec("pillow", "PIL", "Pillow>=10.0.0", True, "Image encode/decode"),
    DepSpec("yt_dlp", "yt_dlp", "yt-dlp[default]>=2025.9.26", True, "YouTube VOD download"),
    DepSpec("onnxruntime", "onnxruntime", "onnxruntime>=1.17.0", True, "Neural depth + packaged robot ONNX detector"),
    DepSpec("openpyxl", "openpyxl", "openpyxl>=3.1.5", True, "Scout sheet export"),
)

OPTIONAL: tuple[DepSpec, ...] = (
    DepSpec(
        "ultralytics",
        "ultralytics",
        "ultralytics>=8.3.0",
        False,
        "Training / Ultralytics YOLO track (source installs; not in desktop freeze)",
        "requirements.txt",
    ),
    DepSpec(
        "torch",
        "torch",
        "torch>=2.2.0",
        False,
        "Optional Depth Anything torch backend",
        "requirements-depth.txt",
    ),
    DepSpec(
        "transformers",
        "transformers",
        "transformers>=4.41.0",
        False,
        "Optional Depth Anything transformers backend",
        "requirements-depth.txt",
    ),
    DepSpec(
        "laya",
        "laya",
        "laya>=0.3.0",
        False,
        "Local scout verification (System-1); ~10× faster than cloud Jev",
        "requirements-laya.txt",
    ),
    DepSpec(
        "nodriver",
        "nodriver",
        "nodriver>=0.40.0",
        False,
        "YouTube download assistance (source installs)",
        "requirements.txt",
    ),
    DepSpec(
        "webview",
        "webview",
        "pywebview>=5.0",
        False,
        "Desktop chrome-less window",
        "requirements-desktop.txt",
    ),
    DepSpec(
        "pytest",
        "pytest",
        "pytest>=8.3.0",
        False,
        "Test suite",
        "requirements.txt",
    ),
)


def _version(mod: Any) -> str:
    for attr in ("__version__", "VERSION", "version"):
        val = getattr(mod, attr, None)
        if isinstance(val, bytes):
            val = val.decode("utf-8", "ignore")
        if callable(val):
            continue
        if isinstance(val, type) or (val is not None and type(val).__name__ == "module"):
            continue
        if val is not None and val != "" and not str(val).startswith("<module"):
            return str(val)
    return ""


def check_one(spec: DepSpec) -> dict[str, Any]:
    row: dict[str, Any] = {
        "key": spec.key,
        "import": spec.import_name,
        "pip": spec.pip,
        "required": spec.required,
        "purpose": spec.purpose,
        "install_extra": spec.install_extra,
        "ok": False,
        "version": "",
        "error": "",
    }
    try:
        mod = importlib.import_module(spec.import_name)
        row["ok"] = True
        row["version"] = _version(mod)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def check_deps(*, include_optional: bool = True) -> dict[str, Any]:
    required = [check_one(s) for s in REQUIRED]
    optional = [check_one(s) for s in OPTIONAL] if include_optional else []
    missing_required = [r["key"] for r in required if not r["ok"]]
    missing_optional = [r["key"] for r in optional if not r["ok"]]
    install_cmds: list[str] = []
    if missing_required:
        install_cmds.append("pip install -r requirements.txt")
        install_cmds.append("pip install -r requirements-desktop.txt  # slim desktop / freeze set")
    # Point at extras for optional gaps.
    extras_needed = sorted(
        {
            next(s.install_extra for s in OPTIONAL if s.key == key)
            for key in missing_optional
            if next((s.install_extra for s in OPTIONAL if s.key == key), "")
        }
    )
    for extra in extras_needed:
        if extra and extra not in {"requirements.txt", "requirements-desktop.txt"}:
            install_cmds.append(f"pip install -r {extra}")

    # Detector / depth status without claiming FRC-ready weights.
    detector: dict[str, Any] = {}
    depth: dict[str, Any] = {}
    try:
        from ramscout.onnx_detector import detector_status

        detector = detector_status()
    except Exception as exc:  # noqa: BLE001
        detector = {"error": str(exc), "frc_ready": False}
    try:
        from ramscout.depth import depth_backends

        depth = depth_backends()
    except Exception as exc:  # noqa: BLE001
        depth = {"error": str(exc)}

    return {
        "ok": not missing_required,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "required": required,
        "optional": optional,
        "install": install_cmds,
        "detector": detector,
        "depth": depth,
        "notes": [
            "onnxruntime is required for neural depth and the packaged robot ONNX detector.",
            "ultralytics is for training/export and optional YOLO tracking in source installs — not required at desktop runtime.",
            "Do not claim FRC-ready detector weights until models/robot.onnx.meta.json has frc_ready=true.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    strict = "--strict" in args
    quiet = "--quiet" in args
    report = check_deps()
    if not quiet:
        print(f"RoboScoutAI dependencies: {'OK' if report['ok'] else 'MISSING REQUIRED'}")
        for row in report["required"]:
            mark = "OK  " if row["ok"] else "MISS"
            ver = f" {row['version']}" if row["version"] else ""
            err = f" ({row['error']})" if row["error"] else ""
            print(f"  [{mark}] {row['key']}{ver}{err} — {row['purpose']}")
        print("Optional:")
        for row in report["optional"]:
            mark = "OK  " if row["ok"] else "miss"
            ver = f" {row['version']}" if row["version"] else ""
            print(f"  [{mark}] {row['key']}{ver} — {row['purpose']}")
        if report["install"]:
            print("Install hints:")
            for cmd in report["install"]:
                print(f"  {cmd}")
        det = report.get("detector") or {}
        print(
            f"Detector: onnxruntime={'yes' if det.get('onnxruntime') else 'no'} "
            f"weights={'yes' if det.get('weights') else 'no'} "
            f"frc_ready={bool(det.get('frc_ready'))}"
        )
    if strict and not report["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
