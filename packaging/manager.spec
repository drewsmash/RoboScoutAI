# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for ``RoboScoutAI.exe`` — the thin launcher / manager.

Pure standard library (+ tkinter for the splash and Setup dialog), so it is a
few MB and rarely needs to change. It installs, verifies, launches, updates and
rolls back the side-by-side app builds under %LOCALAPPDATA%\\RoboScoutAI\\app\\.
"""

from __future__ import annotations

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

datas = []
_channel = ROOT / "release-artifacts" / "update-channel.txt"
if _channel.is_file():
    datas.append((str(_channel), "."))

hiddenimports = ["roboscout_manager", "tkinter", "tkinter.messagebox"]
hiddenimports += collect_submodules("roboscout_manager")

a = Analysis(
    [str(ROOT / "roboscout_manager" / "__main__.py")],
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
    name="RoboScoutAI",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # No UPX: packed launchers trip antivirus heuristics far more often.
    upx=False,
    runtime_tmpdir=None,
    # Windowed: no console flashes when users double-click the shortcut.
    # CLI commands still work; use --json --out FILE for machine-readable output.
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=APP_ICON,
)
