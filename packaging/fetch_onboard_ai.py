"""Prefetch onboard models so the desktop freeze can pack them.

Writes under ``models/`` (gitignored):

* ``depth-anything-v2-small.onnx`` — Depth Anything V2 Small
* ``hf/`` — Laya typed-decisions weights, when the ``laya`` package imports

``ROBOSCOUT_REQUIRE_ONBOARD_AI=1`` fails the script if Laya cannot be imported.
A weight download that fails still continues so torch and the package stay in
the build; the first scout can finish the download into the user models folder.
"""

from __future__ import annotations

import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
ONNX_NAME = "depth-anything-v2-small.onnx"
ONNX_URL = (
    "https://huggingface.co/onnx-community/depth-anything-v2-small/resolve/main/onnx/model.onnx"
)
MIN_BYTES = 50_000_000


def _required() -> bool:
    return (os.environ.get("ROBOSCOUT_REQUIRE_ONBOARD_AI") or "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def fetch_onnx() -> None:
    MODELS.mkdir(parents=True, exist_ok=True)
    dest = MODELS / ONNX_NAME
    if dest.is_file() and dest.stat().st_size >= MIN_BYTES:
        print(f"depth onnx already present ({dest.stat().st_size} bytes)")
        return
    print(f"downloading {ONNX_URL}")
    urllib.request.urlretrieve(ONNX_URL, dest)
    size = dest.stat().st_size
    if size < MIN_BYTES:
        raise SystemExit(f"depth onnx too small: {size} bytes at {dest}")
    print(f"depth onnx saved ({size} bytes)")


def fetch_laya() -> None:
    require = _required()
    try:
        import laya
    except Exception as exc:  # noqa: BLE001
        message = f"laya is not installed ({exc}). pip install -r requirements-laya.txt"
        if require:
            raise SystemExit(message) from exc
        print("warning:", message)
        return
    cache = MODELS / "hf"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "hub")
    os.environ["USE_TF"] = "0"
    try:
        laya.load("convaiinnovations/laya", subfolder="typed-decisions")
        print("laya weights cached at", cache)
    except Exception as exc:  # noqa: BLE001
        print("warning: laya weight download failed:", exc)
        print("torch and the laya package will still be packed; first scout can download weights")


def main() -> None:
    fetch_onnx()
    fetch_laya()


if __name__ == "__main__":
    main()
    sys.exit(0)
