"""Check GitHub Releases for newer RoboScoutAI desktop builds and apply them.

Accepts both RoboScoutAI-* and legacy RamScoutAI-* release assets so updates
keep working across the rebrand. Picks the newest semver release with a
matching desktop asset — not only GitHub's marked \"latest\" (which can lag).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from ramscout import __version__
from ramscout.brand import APP_NAME, BINARY_NAME, DEFAULT_GITHUB_REPO, LEGACY_APP_NAME, LEGACY_BINARY_NAME
from ramscout.paths import app_dir, is_frozen

log = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_CHECK: dict[str, Any] | None = None


@dataclass
class UpdateInfo:
    available: bool
    current_version: str
    latest_version: str = ""
    release_url: str = ""
    asset_name: str = ""
    asset_url: str = ""
    body: str = ""
    frozen: bool = False
    platform: str = ""
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def github_repo() -> str:
    for key in ("ROBOSCOUT_GITHUB_REPO", "RAMSCOUT_GITHUB_REPO"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return DEFAULT_GITHUB_REPO


def github_token() -> str:
    for key in (
        "ROBOSCOUT_GITHUB_TOKEN",
        "RAMSCOUT_GITHUB_TOKEN",
        "GITHUB_TOKEN",
        "GH_TOKEN",
    ):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos-arm64" if machine in {"arm64", "aarch64"} else "macos-x64"
    return f"linux-{machine or 'x64'}"


def preferred_asset_names() -> list[str]:
    """Ordered candidates: new brand first, then legacy RamScoutAI names."""
    key = platform_key()
    brands = (BINARY_NAME, LEGACY_BINARY_NAME)
    if key == "windows":
        names: list[str] = []
        for brand in brands:
            names.extend(
                [
                    f"{brand}-windows-x64.exe",
                    f"{brand}-windows-x64-signed.exe",
                    f"{brand}-windows.exe",
                    f"{brand}.exe",
                ]
            )
        return names
    if key.startswith("macos"):
        names = []
        for brand in brands:
            names.extend(
                [
                    f"{brand}-{key}.zip",
                    f"{brand}-macos-arm64.zip",
                    f"{brand}-macos.zip",
                    f"{brand}-macos-universal.zip",
                    f"{brand}-{key}.7z",
                    f"{brand}-macos-arm64.7z",
                ]
            )
            if key == "macos-x64":
                names.append(f"{brand}-macos-x64.zip")
        return names
    names = []
    for brand in brands:
        names.extend([f"{brand}-{key}.tar.gz", f"{brand}-linux.tar.gz"])
    return names


def normalize_version(tag: str) -> str:
    text = (tag or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    return text


def version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in re.split(r"[^\d]+", normalize_version(version)):
        if chunk.isdigit():
            parts.append(int(chunk))
    return tuple(parts or [0])


def is_newer(latest: str, current: str) -> bool:
    return version_tuple(latest) > version_tuple(current)


def _api_headers(version: str, *, download: bool = False) -> dict[str, str]:
    headers = {
        "User-Agent": f"{APP_NAME}/{version}",
        "Accept": "application/octet-stream" if download else "application/vnd.github+json",
    }
    token = github_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def check_for_update(current: str | None = None, timeout: float = 15.0) -> UpdateInfo:
    current_version = normalize_version(current or __version__)
    info = UpdateInfo(
        available=False,
        current_version=current_version,
        frozen=is_frozen(),
        platform=platform_key(),
    )
    repo = github_repo()
    url = f"https://api.github.com/repos/{repo}/releases"
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=_api_headers(current_version)) as client:
            res = client.get(url, params={"per_page": 30})
            if res.status_code == 404:
                if github_token():
                    info.error = (
                        "No desktop release is published yet. "
                        f"Open https://github.com/{repo}/releases after the first tagged build finishes."
                    )
                else:
                    info.error = (
                        f"Could not read releases for {repo}. "
                        "If the GitHub repo is private, download while signed in at "
                        f"https://github.com/{repo}/releases "
                        "or set ROBOSCOUT_GITHUB_TOKEN / RAMSCOUT_GITHUB_TOKEN for in-app updates."
                    )
                info.release_url = f"https://github.com/{repo}/releases"
                return _store(info)
            res.raise_for_status()
            payload = res.json()
            if not isinstance(payload, list):
                # Unexpected shape — try classic /latest as a last resort.
                latest_res = client.get(f"https://api.github.com/repos/{repo}/releases/latest")
                latest_res.raise_for_status()
                payload = [latest_res.json()]
    except Exception as exc:  # noqa: BLE001
        info.error = f"Could not check GitHub for updates: {exc}"
        return _store(info)

    release = _pick_newest_release(payload)
    if release is None:
        info.error = (
            f"No published desktop releases found for {repo}. "
            f"Check https://github.com/{repo}/releases."
        )
        info.release_url = f"https://github.com/{repo}/releases"
        return _store(info)

    latest = normalize_version(release.get("tag_name") or release.get("name") or "")
    info.latest_version = latest
    info.release_url = release.get("html_url") or f"https://github.com/{repo}/releases"
    info.body = (release.get("body") or "")[:2000]
    if not latest or not is_newer(latest, current_version):
        return _store(info)

    asset = _pick_asset(release.get("assets") or [])
    if asset is None:
        info.error = (
            f"Release {latest} exists, but the {info.platform} desktop file is not attached yet. "
            f"Check https://github.com/{repo}/releases for Windows/macOS assets."
        )
        info.available = False
        return _store(info)

    info.available = True
    info.asset_name = asset.get("name") or ""
    # Private repos need the API asset URL + token; public repos can use the browser URL.
    if github_token() and asset.get("url"):
        info.asset_url = str(asset.get("url"))
    else:
        info.asset_url = str(asset.get("browser_download_url") or asset.get("url") or "")
    return _store(info)


def last_check() -> dict[str, Any] | None:
    with _STATE_LOCK:
        return dict(_LAST_CHECK) if _LAST_CHECK else None


def download_update(info: UpdateInfo | None = None, dest_dir: Path | None = None) -> Path:
    """Download the release asset for this platform into dest_dir."""
    update = info or UpdateInfo(**(last_check() or {}))
    if not update.available or not update.asset_url:
        raise RuntimeError(update.error or "No update asset is available to download.")
    target_dir = dest_dir or (app_dir() / "updates")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / (update.asset_name or f"{BINARY_NAME}-update.bin")
    headers = _api_headers(__version__, download=True)
    with httpx.stream("GET", update.asset_url, headers=headers, follow_redirects=True, timeout=120.0) as res:
        res.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in res.iter_bytes(1024 * 256):
                fh.write(chunk)
    return dest


def apply_downloaded_update(package: Path) -> str:
    """Replace the running frozen binary and relaunch. Returns a status message."""
    if not is_frozen():
        raise RuntimeError("Auto-apply only works from the Windows/macOS desktop build.")
    package = package.resolve()
    if not package.is_file():
        raise RuntimeError(f"Update package missing: {package}")

    system = platform.system().lower()
    if system.startswith("win"):
        return _apply_windows(package)
    if system == "darwin":
        return _apply_macos(package)
    raise RuntimeError("Auto-apply is only supported on Windows and macOS desktop builds.")


def _pick_newest_release(releases: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Choose the highest semver among published (non-draft) releases."""
    scored: list[tuple[tuple[int, ...], dict[str, Any]]] = []
    for release in releases or []:
        if release.get("draft"):
            continue
        # Prefer non-prerelease; keep prereleases only if nothing else exists.
        tag = normalize_version(str(release.get("tag_name") or release.get("name") or ""))
        if not tag:
            continue
        scored.append((version_tuple(tag), release))
    if not scored:
        return None
    scored.sort(key=lambda item: item[0], reverse=True)
    # First pass: newest non-prerelease
    for _ver, release in scored:
        if not release.get("prerelease"):
            return release
    return scored[0][1]


def _pick_asset(assets: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_name = {str(a.get("name") or ""): a for a in assets}
    for name in preferred_asset_names():
        if name in by_name:
            return by_name[name]
    key = platform_key().split("-")[0]
    brand_tokens = (BINARY_NAME.lower(), LEGACY_BINARY_NAME.lower(), "roboscout", "ramscout")
    for name, asset in by_name.items():
        lower = name.lower()
        if key not in lower:
            continue
        if not any(token in lower for token in brand_tokens):
            continue
        if lower.endswith((".exe", ".zip", ".7z", ".tar.gz")):
            return asset
    # Last resort: platform match only (unsigned / oddly named builds).
    for name, asset in by_name.items():
        lower = name.lower()
        if key in lower and lower.endswith((".exe", ".zip", ".7z", ".tar.gz")):
            return asset
    return None


def _store(info: UpdateInfo) -> UpdateInfo:
    with _STATE_LOCK:
        global _LAST_CHECK
        _LAST_CHECK = info.as_dict()
    return info


def _apply_windows(package: Path) -> str:
    exe = Path(sys.executable).resolve()
    target = _windows_install_target(exe)
    staging = target.with_suffix(target.suffix + ".new")
    shutil.copy2(package, staging)
    script = Path(tempfile.gettempdir()) / "roboscout_update.bat"
    lines = [
        "@echo off",
        "timeout /t 2 /nobreak >nul",
        f'move /Y "{staging}" "{target}"',
    ]
    if target.resolve() != exe.resolve():
        lines.append(f'del /F /Q "{exe}"')
    lines.extend(
        [
            f'start "" "{target}"',
            'del "%~f0"',
            "",
        ]
    )
    script.write_text("\r\n".join(lines), encoding="utf-8")
    subprocess.Popen(["cmd", "/c", str(script)], close_fds=True)
    return f"Update staged. {APP_NAME} will restart momentarily."


def _windows_install_target(exe: Path) -> Path:
    """Prefer RoboScoutAI.exe when upgrading a legacy RamScoutAI.exe install."""
    lower_name = exe.name.lower()
    if LEGACY_BINARY_NAME.lower() in lower_name and BINARY_NAME.lower() not in lower_name:
        return exe.with_name(f"{BINARY_NAME}.exe")
    return exe


def _apply_macos(package: Path) -> str:
    exe = Path(sys.executable).resolve()
    extract_dir = Path(tempfile.mkdtemp(prefix="roboscout-update-"))
    suffix = package.suffix.lower()
    if suffix in {".zip", ".7z"}:
        if suffix == ".7z":
            # Prefer 7z CLI when present; zipfile cannot read 7z.
            seven = shutil.which("7z") or shutil.which("7zz")
            if seven:
                subprocess.run([seven, "x", str(package), f"-o{extract_dir}", "-y"], check=True)
            else:
                raise RuntimeError(
                    "macOS update is a .7z archive but 7z is not installed. "
                    f"Download the .zip from the release page, or install p7zip."
                )
        else:
            shutil.unpack_archive(package, extract_dir)
    else:
        shutil.copy2(package, extract_dir / package.name)

    replacement = _find_macos_binary(extract_dir)
    if replacement is None:
        raise RuntimeError(f"Could not find {APP_NAME} binary inside the macOS update archive.")

    target = exe
    if LEGACY_BINARY_NAME in exe.name and BINARY_NAME not in exe.name:
        target = exe.with_name(BINARY_NAME)

    staging = target.with_name(target.name + ".new")
    shutil.copy2(replacement, staging)
    os.chmod(staging, 0o755)
    script = Path(tempfile.gettempdir()) / "roboscout_update.sh"
    remove_old = f'rm -f "{exe}"' if target.resolve() != exe.resolve() else "true"
    script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                "sleep 2",
                f'mv -f "{staging}" "{target}"',
                f'chmod +x "{target}"',
                remove_old,
                f'"{target}" >/dev/null 2>&1 &',
                f'rm -f "{script}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    subprocess.Popen(["/bin/bash", str(script)], start_new_session=True)
    return f"Update staged. {APP_NAME} will restart momentarily."


def _find_macos_binary(root: Path) -> Path | None:
    for name in (BINARY_NAME, LEGACY_BINARY_NAME):
        candidates = sorted(root.rglob(name))
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return path
    for path in root.rglob("*"):
        if path.is_file() and path.suffix == "" and ("RoboScout" in path.name or "RamScout" in path.name):
            return path
    return None


def write_update_state(path: Path | None = None) -> Path:
    """Persist last check JSON for the desktop launcher / debugging."""
    target = path or (app_dir() / "update-state.json")
    payload = last_check() or check_for_update().as_dict()
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target
