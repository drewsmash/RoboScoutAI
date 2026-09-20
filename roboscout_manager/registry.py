"""HKCU uninstall entry so RoboScoutAI shows up in Windows "Installed apps"."""

from __future__ import annotations

import logging
from pathlib import Path
from types import ModuleType
from typing import Any

from roboscout_manager import APP_NAME, MANAGER_VERSION, PUBLISHER
from roboscout_manager.state import app_versions_dir, manager_exe_path

log = logging.getLogger("roboscout.manager")

UNINSTALL_KEY = rf"Software\Microsoft\Windows\CurrentVersion\Uninstall\{APP_NAME}"
APP_PATHS_KEY = rf"Software\Microsoft\Windows\CurrentVersion\App Paths\{APP_NAME}.exe"


def _winreg() -> ModuleType | None:
    try:
        import winreg  # type: ignore

        return winreg
    except ImportError:
        return None


def estimated_size_kb(root: Path) -> int:
    total = 0
    try:
        for path in Path(root).rglob("*"):
            if "data" in path.relative_to(root).parts[:1]:
                continue
            if path.is_file():
                total += path.stat().st_size
    except OSError:
        pass
    return max(1, total // 1024)


def uninstall_entry(root: Path, version: str, *, manager_version: str = MANAGER_VERSION, size_kb: int | None = None) -> dict[str, tuple[str, Any]]:
    """Registry values as ``name -> (type, value)`` — pure data, easy to test."""
    root = Path(root)
    exe = manager_exe_path(root)
    kb = size_kb if size_kb is not None else estimated_size_kb(root)
    return {
        "DisplayName": ("REG_SZ", APP_NAME),
        "DisplayVersion": ("REG_SZ", version or manager_version),
        "Publisher": ("REG_SZ", PUBLISHER),
        "InstallLocation": ("REG_SZ", str(root)),
        "DisplayIcon": ("REG_SZ", f"{exe},0"),
        "UninstallString": ("REG_SZ", f'"{exe}" --uninstall'),
        "QuietUninstallString": ("REG_SZ", f'"{exe}" --uninstall --silent'),
        "ModifyPath": ("REG_SZ", f'"{exe}" --repair'),
        "URLInfoAbout": ("REG_SZ", "https://github.com/drewsmash/RoboScoutAI"),
        "Comments": ("REG_SZ", f"{APP_NAME} desktop (manager {manager_version})"),
        "NoModify": ("REG_DWORD", 0),
        "NoRepair": ("REG_DWORD", 0),
        "EstimatedSize": ("REG_DWORD", int(kb)),
        "ManagerVersion": ("REG_SZ", manager_version),
        "AppVersionsDir": ("REG_SZ", str(app_versions_dir(root))),
    }


def register_uninstall(root: Path, version: str, *, winreg: ModuleType | None = None) -> bool:
    reg = winreg or _winreg()
    if reg is None:
        log.info("winreg unavailable; skipping uninstall registration")
        return False
    values = uninstall_entry(root, version)
    key = reg.CreateKeyEx(reg.HKEY_CURRENT_USER, UNINSTALL_KEY, 0, reg.KEY_SET_VALUE)
    try:
        for name, (kind, value) in values.items():
            reg_type = reg.REG_DWORD if kind == "REG_DWORD" else reg.REG_SZ
            reg.SetValueEx(key, name, 0, reg_type, value)
    finally:
        reg.CloseKey(key)
    # App Paths lets `RoboScoutAI.exe` resolve from Run / Start typed search.
    try:
        apk = reg.CreateKeyEx(reg.HKEY_CURRENT_USER, APP_PATHS_KEY, 0, reg.KEY_SET_VALUE)
        try:
            reg.SetValueEx(apk, None, 0, reg.REG_SZ, str(manager_exe_path(root)))
            reg.SetValueEx(apk, "Path", 0, reg.REG_SZ, str(root))
        finally:
            reg.CloseKey(apk)
    except OSError as exc:
        log.debug("App Paths registration skipped: %s", exc)
    return True


def unregister_uninstall(*, winreg: ModuleType | None = None) -> bool:
    reg = winreg or _winreg()
    if reg is None:
        return False
    removed = False
    for path in (UNINSTALL_KEY, APP_PATHS_KEY):
        try:
            reg.DeleteKey(reg.HKEY_CURRENT_USER, path)
            removed = True
        except OSError:
            continue
    return removed


def registered_version(*, winreg: ModuleType | None = None) -> str | None:
    reg = winreg or _winreg()
    if reg is None:
        return None
    try:
        key = reg.OpenKey(reg.HKEY_CURRENT_USER, UNINSTALL_KEY, 0, reg.KEY_READ)
    except OSError:
        return None
    try:
        value, _kind = reg.QueryValueEx(key, "DisplayVersion")
        return str(value)
    except OSError:
        return None
    finally:
        reg.CloseKey(key)
