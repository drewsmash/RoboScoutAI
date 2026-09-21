"""Start Menu / Desktop ``.lnk`` shortcuts via a generated PowerShell script."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Callable

from roboscout_manager import APP_NAME
from roboscout_manager.state import manager_exe_path

log = logging.getLogger("roboscout.manager")

SHORTCUT_NAME = f"{APP_NAME}.lnk"
DESCRIPTION = f"{APP_NAME} — FRC match auto-scout"


def start_menu_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs"


def desktop_dir() -> Path:
    profile = os.environ.get("USERPROFILE")
    base = Path(profile) if profile else Path.home()
    return base / "Desktop"


def shortcut_paths(*, desktop: bool, start_menu: bool = True) -> list[Path]:
    paths: list[Path] = []
    if start_menu:
        paths.append(start_menu_dir() / SHORTCUT_NAME)
    if desktop:
        paths.append(desktop_dir() / SHORTCUT_NAME)
    return paths


def _ps_quote(text: str) -> str:
    return "'" + str(text).replace("'", "''") + "'"


def shortcut_script(
    target: Path,
    links: list[Path],
    *,
    working_dir: Path | None = None,
    icon: Path | None = None,
    description: str = DESCRIPTION,
    arguments: str = "",
) -> str:
    """PowerShell that creates every ``.lnk`` in ``links`` pointing at ``target``."""
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "$shell = New-Object -ComObject WScript.Shell",
        f"$target = {_ps_quote(target)}",
    ]
    for link in links:
        lines += [
            f"$dir = Split-Path -Parent {_ps_quote(link)}",
            "if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }",
            f"$lnk = $shell.CreateShortcut({_ps_quote(link)})",
            "$lnk.TargetPath = $target",
            f"$lnk.WorkingDirectory = {_ps_quote(working_dir or Path(target).parent)}",
            f"$lnk.Description = {_ps_quote(description)}",
            f"$lnk.IconLocation = {_ps_quote(str(icon or target) + ',0')}",
        ]
        if arguments:
            lines.append(f"$lnk.Arguments = {_ps_quote(arguments)}")
        lines.append("$lnk.Save()")
    lines.append("Write-Output 'SHORTCUTS_OK'")
    return "\n".join(lines) + "\n"


def remove_script(links: list[Path]) -> str:
    lines = ["$ErrorActionPreference = 'SilentlyContinue'"]
    for link in links:
        lines.append(f"Remove-Item -LiteralPath {_ps_quote(link)} -Force")
    lines.append("Write-Output 'SHORTCUTS_REMOVED'")
    return "\n".join(lines) + "\n"


def run_powershell(script: str, *, timeout: float = 60.0) -> str:
    if not sys.platform.startswith("win"):
        raise OSError("PowerShell shortcuts are only created on Windows")
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False, encoding="utf-8-sig") as handle:
        handle.write(script)
        path = handle.name
    try:
        proc = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=0x08000000,  # CREATE_NO_WINDOW
        )
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    if proc.returncode != 0:
        raise OSError((proc.stderr or proc.stdout or f"powershell exited {proc.returncode}").strip())
    return proc.stdout


def create_shortcuts(
    root: Path,
    *,
    desktop: bool = False,
    start_menu: bool = True,
    runner: Callable[[str], str] | None = None,
) -> list[Path]:
    links = shortcut_paths(desktop=desktop, start_menu=start_menu)
    if not links:
        return []
    script = shortcut_script(manager_exe_path(root), links)
    (runner or run_powershell)(script)
    return links


def remove_shortcuts(*, runner: Callable[[str], str] | None = None) -> list[Path]:
    links = shortcut_paths(desktop=True, start_menu=True)
    if runner is None and not sys.platform.startswith("win"):
        removed = []
        for link in links:
            try:
                if link.exists():
                    link.unlink()
                    removed.append(link)
            except OSError:
                pass
        return removed
    (runner or run_powershell)(remove_script(links))
    return links
