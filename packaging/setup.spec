# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ``RoboScoutAI-Setup.exe`` — the one-time installer.

It is the manager code (``roboscout_manager``) with a ``payload/`` folder
bundled in: ``RoboScoutAI-app-windows-x64.exe``, ``RoboScoutAI.exe`` (manager)
and ``manifest.json``. Running it installs into %LOCALAPPDATA%\\RoboScoutAI (no
admin), registers the uninstall entry, creates shortcuts and launches the app.

Payload directory: ``ROBOSCOUT_SETUP_PAYLOAD`` (default ``dist/release``). When a
payload file is missing, Setup still works: the manager downloads the app from
the update channel on first launch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path(SPECPATH).resolve().parent
block_cipher = None

if sys.platform == "win32":
    APP_ICON = str(ROOT / "web" / "icons" / "app.ico")
elif sys.platform == "darwin":
    APP_ICON = str(ROOT / "web" / "icons" / "app.icns")
else:
    APP_ICON = str(ROOT / "web" / "icons" / "app.png")
if not Path(APP_ICON).is_file():
    APP_ICON = None

PAYLOAD_DIR = Path(os.environ.get("ROBOSCOUT_SETUP_PAYLOAD") or (ROOT / "dist" / "release"))
PAYLOAD_FILES = ("RoboScoutAI-app-windows-x64.exe", "RoboScoutAI.exe", "manifest.json")

datas = []
for name in PAYLOAD_FILES:
    candidate = PAYLOAD_DIR / name
    if candidate.is_file():
        datas.append((str(candidate), "payload"))
    else:
        print(f"[setup.spec] payload file missing (Setup will download it): {candidate}")
_channel = ROOT / "release-artifacts" / "update-channel.txt"
if _channel.is_file():
    datas.append((str(_channel), "."))

hiddenimports = ["roboscout_manager", "tkinter", "tkinter.messagebox"]
hiddenimports += collect_submodules("roboscout_manager")

a = Analysis(
    [str(ROOT / "roboscout_manager" / "setup_entry.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "ramscout",
        "desktop",
        "app",
        "fastapi",
        "uvicorn",
        "starlette",
        "pydantic",
        "numpy",
        "cv2",
        "PIL",
        "yt_dlp",
        "httpx",
        "webview",
        "matplotlib",
        "IPython",
        "pytest",
        "torch",
        "ultralytics",
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
    name="RoboScoutAI-Setup",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
)
