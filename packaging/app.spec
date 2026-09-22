# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the RoboScoutAI **app** build (FastAPI + web UI).

Output: ``dist/RoboScoutAI-app.exe`` (Windows) / ``dist/RoboScoutAI-app`` (macOS).
On Windows this binary is not run directly by users: ``RoboScoutAI.exe`` (the
manager, see ``manager.spec``) installs it side-by-side under
``%LOCALAPPDATA%\\RoboScoutAI\\app\\<version>\\`` and launches it with ``--managed``.

Onboard AI ships in this freeze: Laya (torch + transformers) and the Depth
Anything V2 Small ONNX file under ``models/``. Release builds set
``ROBOSCOUT_REQUIRE_ONBOARD_AI=1`` so a missing Laya/torch install fails.
Ultralytics stays excluded. UPX is off because it crashes torch. Place
``models/robot.onnx`` for FRC detection when available. Demo mode, overlay
OCR, YouTube ingest and the manager-driven updater all work.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

ROOT = Path(SPECPATH).resolve().parent
block_cipher = None

# Desktop app icon (Windows .ico / macOS .icns). Falls back gracefully if missing.
if sys.platform == "win32":
    APP_ICON = str(ROOT / "web" / "icons" / "app.ico")
elif sys.platform == "darwin":
    APP_ICON = str(ROOT / "web" / "icons" / "app.icns")
else:
    APP_ICON = str(ROOT / "web" / "icons" / "app.png")
if not Path(APP_ICON).is_file():
    APP_ICON = None

datas = [
    (str(ROOT / "web"), "web"),
    (str(ROOT / "ramscout" / "games"), "ramscout/games"),
]
# Optional freeze-time update channel (branch name). Overridable by RAMSCOUT_GIT_BRANCH.
_channel = ROOT / "release-artifacts" / "update-channel.txt"
if _channel.is_file():
    datas.append((str(_channel), "."))
datas += collect_data_files("yt_dlp")
_models = ROOT / "models"
if _models.is_dir() and any(_models.iterdir()):
    datas.append((str(_models), "models"))

hiddenimports = [
    "uvicorn.logging",
    "uvicorn.loops",
    "uvicorn.loops.auto",
    "uvicorn.protocols",
    "uvicorn.protocols.http",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan",
    "uvicorn.lifespan.on",
    "anyio._backends._asyncio",
    "httpx",
    "multipart",
    "python_multipart",
    "PIL",
    "PIL.Image",
    "numpy",
    "scipy",
    "cv2",
    "onnxruntime",
    "openpyxl",
    "yt_dlp",
    "ramscout",
    "ramscout.trackers",
    "ramscout.crop",
    "ramscout.identity_book",
    "ramscout.onnx_detector",
    "ramscout.deps",
    "ramscout.managed",
    "roboscout_manager",
    "roboscout_manager.ipc",
    "roboscout_manager.state",
    "desktop",
    "desktop.main",
    "desktop.app_window",
    "app",
    "webview",
    "laya",
    "torch",
    "transformers",
]
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("ramscout")
hiddenimports += collect_submodules("roboscout_manager")
hiddenimports += collect_submodules("webview")

# These are required at runtime. The freeze must contain them; a missing
# package used to be swallowed and the installed app then failed mid-track.
REQUIRED_FREEZE = ("cv2", "numpy", "scipy", "PIL", "onnxruntime", "openpyxl")
binaries = []
_missing_freeze: list[str] = []
for _pkg in REQUIRED_FREEZE:
    try:
        tmp_datas, tmp_binaries, tmp_hidden = collect_all(_pkg)
    except Exception as exc:  # noqa: BLE001
        _missing_freeze.append(f"{_pkg}: {exc}")
        continue
    if not (tmp_datas or tmp_binaries or tmp_hidden):
        _missing_freeze.append(f"{_pkg}: collect_all returned nothing")
        continue
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hidden
if _missing_freeze:
    raise SystemExit(
        "Desktop build is missing required packages (install requirements-desktop.txt first):\n  "
        + "\n  ".join(_missing_freeze)
    )

# Laya + torch + transformers are the onboard scout runtime. Local pytest can
# compile this spec without them; the release workflow requires them.
ONBOARD_AI = ("laya", "torch", "transformers")
_require_ai = (os.environ.get("ROBOSCOUT_REQUIRE_ONBOARD_AI") or "").strip().lower() in {
    "1",
    "true",
    "yes",
}
_missing_ai: list[str] = []
for _pkg in ONBOARD_AI:
    try:
        tmp_datas, tmp_binaries, tmp_hidden = collect_all(_pkg)
    except Exception as exc:  # noqa: BLE001
        _missing_ai.append(f"{_pkg}: {exc}")
        continue
    if not (tmp_datas or tmp_binaries or tmp_hidden):
        _missing_ai.append(f"{_pkg}: collect_all returned nothing")
        continue
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hidden
if _require_ai and _missing_ai:
    raise SystemExit(
        "Desktop build is missing onboard AI (pip install torch and requirements-laya.txt):\n  "
        + "\n  ".join(_missing_ai)
    )

a = Analysis(
    [str(ROOT / "desktop" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "IPython",
        "jupyter",
        "notebook",
        "pytest",
        "ultralytics",
        "tensorboard",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="RoboScoutAI-app",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    # Console build: the manager launches it with CREATE_NO_WINDOW and captures
    # stdout/stderr into %LOCALAPPDATA%\RoboScoutAI\app.log.
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
)
