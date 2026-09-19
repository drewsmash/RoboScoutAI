"""Resolve resource and writable paths for source and frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from ramscout.brand import APP_NAME


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    """Read-only assets shipped with the app (web UI, game JSON, icons)."""
    if is_frozen():
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def app_dir() -> Path:
    """Directory next to the executable (or project root) for user data."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_root() -> Path:
    """Always-writable per-user folder (avoids Program Files Access Denied)."""
    override = (
        (os.environ.get("ROBOSCOUT_DATA") or "").strip()
        or (os.environ.get("RAMSCOUT_DATA") or "").strip()
    )
    if override:
        return Path(override)
    system = sys.platform
    if system.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / APP_NAME
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def update_cache_dir() -> Path:
    """Where release assets are downloaded before apply."""
    path = user_data_root() / "updates"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_dir() -> Path:
    override = (
        (os.environ.get("ROBOSCOUT_DATA") or "").strip()
        or (os.environ.get("RAMSCOUT_DATA") or "").strip()
    )
    if override:
        path = Path(override)
    elif is_frozen():
        path = user_data_root() / "data"
        legacy = Path.home() / "RamScoutAI" / "data"
        # Prefer new brand folder; fall back so existing installs keep their jobs.
        if not path.exists() and legacy.exists():
            path = legacy
    else:
        path = app_dir() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def jobs_dir() -> Path:
    path = data_dir() / "jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def web_dir() -> Path:
    return bundle_root() / "web"


def models_dirs() -> list[Path]:
    return [app_dir(), app_dir() / "models", data_dir() / "models", Path.cwd()]


def ensure_writable_dir(path: Path) -> Path:
    """Create path and verify we can write files there."""
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise PermissionError(f"Cannot write to {path}: {exc}") from exc
    return path
