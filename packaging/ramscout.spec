# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for RoboScoutAI desktop builds (Windows .exe / macOS binary).

Keeps the freeze lean by excluding torch/ultralytics; neural depth ships via
onnxruntime (Depth Anything V2 Small weights download on first use). Demo mode, overlay OCR,
YouTube ingest, and the git-remote in-app updater all work. Drop a robot .pt next to the
app and install ultralytics in a source tree for full tracking.
"""

from __future__ import annotations

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
    "yt_dlp",
    "ramscout",
    "ramscout.trackers",
    "desktop",
    "desktop.main",
    "desktop.app_window",
    "app",
    "webview",
]
hiddenimports += collect_submodules("uvicorn")
hiddenimports += collect_submodules("ramscout")
hiddenimports += collect_submodules("webview")

binaries = []
tmp_datas, tmp_binaries, tmp_hidden = collect_all("cv2")
datas += tmp_datas
binaries += tmp_binaries
hiddenimports += tmp_hidden
for _pkg in ("onnxruntime", "scipy"):
    try:
        tmp_datas, tmp_binaries, tmp_hidden = collect_all(_pkg)
    except Exception:  # noqa: BLE001 — optional in the freeze
        continue
    datas += tmp_datas
    binaries += tmp_binaries
    hiddenimports += tmp_hidden

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
        "torch",
        "torchvision",
        "torchaudio",
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
    name="RoboScoutAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
)
