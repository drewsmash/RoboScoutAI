"""Install-root layout plus the two small JSON files that drive the manager.

``current.json`` says which side-by-side app version is active; ``manager.json``
stores the update channel and remote. Both are written atomically (temp file +
``os.replace``) so a crash mid-write can never leave a half-written pointer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from roboscout_manager import (
    APP_DIR_NAME,
    APP_NAME,
    CACHE_DIR_NAME,
    CURRENT_JSON,
    DEFAULT_CHANNEL,
    DEFAULT_GIT_REMOTE,
    DEFAULT_GITHUB_REPO,
    INSTALLED_APP_EXE_NAME,
    IPC_JSON,
    MANAGER_EXE_NAME,
    MANAGER_JSON,
    MANAGER_VERSION,
    STATUS_JSON,
    UPDATE_CHANNEL_FILE,
    UPDATE_LOG,
)

log = logging.getLogger("roboscout.manager")


# --- layout ---------------------------------------------------------------------


def default_install_root() -> Path:
    override = (os.environ.get("ROBOSCOUT_INSTALL_ROOT") or "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
        return base / APP_NAME
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return Path.home() / ".local" / "share" / APP_NAME


def manager_exe_path(root: Path) -> Path:
    return Path(root) / MANAGER_EXE_NAME


def app_versions_dir(root: Path) -> Path:
    return Path(root) / APP_DIR_NAME


def app_version_dir(root: Path, version: str) -> Path:
    return app_versions_dir(root) / version


def app_exe_path(root: Path, version: str) -> Path:
    return app_version_dir(root, version) / INSTALLED_APP_EXE_NAME


def current_json_path(root: Path) -> Path:
    return Path(root) / CURRENT_JSON


def manager_json_path(root: Path) -> Path:
    return Path(root) / MANAGER_JSON


def update_log_path(root: Path) -> Path:
    return Path(root) / UPDATE_LOG


def status_json_path(root: Path) -> Path:
    return Path(root) / STATUS_JSON


def ipc_json_path(root: Path) -> Path:
    return Path(root) / IPC_JSON


def cache_dir(root: Path) -> Path:
    return Path(root) / CACHE_DIR_NAME


def bundle_root() -> Path:
    """Read-only resources shipped inside the frozen manager (PyInstaller)."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass)
    return Path(__file__).resolve().parent.parent


def running_exe() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve()
    return Path(__file__).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- atomic json ----------------------------------------------------------------


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return path


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


# --- current.json ---------------------------------------------------------------


@dataclass
class CurrentState:
    version: str
    exe: str  # relative to the install root, forward slashes
    sha256: str = ""
    size: int = 0
    installed_at: str = ""
    previous_version: str = ""
    previous_sha256: str = ""
    channel: str = ""
    source: str = ""  # git | releases | setup | local

    def exe_path(self, root: Path) -> Path:
        return Path(root) / Path(self.exe)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_current(root: Path) -> CurrentState | None:
    data = read_json(current_json_path(root))
    if not data or not data.get("version") or not data.get("exe"):
        return None
    return CurrentState(
        version=str(data.get("version")),
        exe=str(data.get("exe")),
        sha256=str(data.get("sha256") or ""),
        size=int(data.get("size") or 0),
        installed_at=str(data.get("installed_at") or ""),
        previous_version=str(data.get("previous_version") or ""),
        previous_sha256=str(data.get("previous_sha256") or ""),
        channel=str(data.get("channel") or ""),
        source=str(data.get("source") or ""),
    )


def write_current(root: Path, state: CurrentState) -> Path:
    if not state.installed_at:
        state.installed_at = utc_now()
    return write_json_atomic(current_json_path(root), state.to_dict())


def relative_app_exe(version: str) -> str:
    return f"{APP_DIR_NAME}/{version}/{INSTALLED_APP_EXE_NAME}"


# --- manager.json ---------------------------------------------------------------


def bundled_channel() -> str | None:
    """update-channel.txt frozen next to / inside the manager, if any."""
    candidates = [
        running_exe().parent / UPDATE_CHANNEL_FILE,
        bundle_root() / UPDATE_CHANNEL_FILE,
        bundle_root() / "release-artifacts" / UPDATE_CHANNEL_FILE,
    ]
    for path in candidates:
        try:
            if path.is_file():
                for line in path.read_text(encoding="utf-8").splitlines():
                    text = line.strip()
                    if text and not text.startswith("#"):
                        return text
        except OSError:
            continue
    return None


@dataclass
class ManagerConfig:
    channel: str = DEFAULT_CHANNEL
    remote: str = DEFAULT_GIT_REMOTE
    github_repo: str = DEFAULT_GITHUB_REPO
    releases_fallback: bool = True
    auto_check: bool = True
    manager_version: str = MANAGER_VERSION
    installed_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        extra = data.pop("extra", {}) or {}
        data.update(extra)
        return data


def default_config() -> ManagerConfig:
    cfg = ManagerConfig()
    env_channel = (os.environ.get("ROBOSCOUT_CHANNEL") or os.environ.get("RAMSCOUT_GIT_BRANCH") or "").strip()
    bundled = bundled_channel()
    cfg.channel = env_channel or bundled or DEFAULT_CHANNEL
    env_remote = (os.environ.get("ROBOSCOUT_GIT_REMOTE") or os.environ.get("RAMSCOUT_GIT_REMOTE") or "").strip()
    if env_remote:
        cfg.remote = env_remote
    env_repo = (os.environ.get("ROBOSCOUT_GITHUB_REPO") or os.environ.get("RAMSCOUT_GITHUB_REPO") or "").strip()
    if env_repo and "/" in env_repo and "://" not in env_repo:
        cfg.github_repo = env_repo
        if not env_remote:
            cfg.remote = f"https://github.com/{env_repo}.git"
    return cfg


def read_config(root: Path) -> ManagerConfig:
    data = read_json(manager_json_path(root))
    cfg = default_config()
    if not data:
        return cfg
    known = {"channel", "remote", "github_repo", "releases_fallback", "auto_check", "manager_version", "installed_at"}
    if data.get("channel"):
        cfg.channel = str(data["channel"])
    if data.get("remote"):
        cfg.remote = str(data["remote"])
    if data.get("github_repo"):
        cfg.github_repo = str(data["github_repo"])
    if "releases_fallback" in data:
        cfg.releases_fallback = bool(data["releases_fallback"])
    if "auto_check" in data:
        cfg.auto_check = bool(data["auto_check"])
    if data.get("manager_version"):
        cfg.manager_version = str(data["manager_version"])
    if data.get("installed_at"):
        cfg.installed_at = str(data["installed_at"])
    cfg.extra = {k: v for k, v in data.items() if k not in known}
    # Environment always wins for one-off testing of another channel/remote.
    env_channel = (os.environ.get("ROBOSCOUT_CHANNEL") or "").strip()
    if env_channel:
        cfg.channel = env_channel
    return cfg


def write_config(root: Path, cfg: ManagerConfig) -> Path:
    if not cfg.installed_at:
        cfg.installed_at = utc_now()
    return write_json_atomic(manager_json_path(root), cfg.to_dict())


def set_channel(root: Path, channel: str) -> ManagerConfig:
    text = (channel or "").strip()
    if not text:
        raise ValueError("channel must be a branch or tag name")
    if any(ch in text for ch in " \t\n~^:?*[\\") or text.startswith("-"):
        raise ValueError(f"invalid channel name: {channel!r}")
    cfg = read_config(root)
    cfg.channel = text
    write_config(root, cfg)
    return cfg


# --- update.log -----------------------------------------------------------------


def append_log(root: Path, message: str) -> None:
    try:
        path = update_log_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(f"[{utc_now()}] {message}\n")
    except OSError:
        pass
    log.info("%s", message)
