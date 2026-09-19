"""Internal updater: pull updates from a git remote (not GitHub Releases)."""

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
from urllib.parse import quote, urlparse, urlunparse

from ramscout import __version__
from ramscout.brand import APP_NAME, BINARY_NAME, DEFAULT_GITHUB_REPO
from ramscout.paths import app_dir, data_dir, is_frozen

log = logging.getLogger(__name__)

DEFAULT_GIT_REMOTE = f"https://github.com/{DEFAULT_GITHUB_REPO}.git"
DEFAULT_GIT_BRANCH = "main"
_STATE_LOCK = threading.Lock()
_LAST_CHECK: dict[str, Any] | None = None


def _bundled_update_channel() -> str | None:
    """Optional update-channel.txt next to the EXE / in the freeze bundle."""
    from ramscout.paths import bundle_root

    candidates = [
        app_dir() / "update-channel.txt",
        bundle_root() / "update-channel.txt",
    ]
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file():
                text = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
                if text and not text.startswith("#"):
                    return text
        except OSError:
            continue
    return None


def git_branch() -> str:
    env = (os.environ.get("RAMSCOUT_GIT_BRANCH") or "").strip()
    if env:
        return env
    bundled = _bundled_update_channel()
    if bundled:
        return bundled
    return DEFAULT_GIT_BRANCH


# Paths inside the repo that may hold desktop binaries (tracked via git for the updater).
# Prefer release-artifacts/ — desktop-downloads/ is gitignored for local staging only.
_ARTIFACT_REL_PATHS = (
    "release-artifacts/RoboScoutAI-windows-x64.exe",
    "release-artifacts/RoboScoutAI-windows-x64-signed.exe",
    "release-artifacts/RoboScoutAI-macos-arm64.zip",
    "release-artifacts/RoboScoutAI-macos-x64.zip",
    "release-artifacts/RoboScoutAI-linux-x86_64.tar.gz",
    "desktop-downloads/RoboScoutAI-windows-x64.exe",
    "desktop-downloads/RoboScoutAI-macos-arm64.zip",
    "desktop-downloads/RoboScoutAI-macos-x64.zip",
    "desktop-downloads/RoboScoutAI-linux-x86_64.tar.gz",
    "dist/release/RoboScoutAI-windows-x64.exe",
    "dist/release/RoboScoutAI-macos-arm64.zip",
)


@dataclass
class UpdateInfo:
    available: bool
    current_version: str
    latest_version: str = ""
    remote: str = ""
    branch: str = ""
    local_sha: str = ""
    remote_sha: str = ""
    mode: str = ""  # source | frozen | unknown
    message: str = ""
    release_url: str = ""  # UI compat: points at the git remote / browse URL
    asset_name: str = ""
    asset_url: str = ""  # local file path or empty until download
    body: str = ""
    frozen: bool = False
    platform: str = ""
    can_apply: bool = False
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def git_remote() -> str:
    env = (os.environ.get("RAMSCOUT_GIT_REMOTE") or "").strip()
    if env:
        return env
    # Legacy alias — still treat as a git remote URL, not Releases API.
    legacy = (os.environ.get("RAMSCOUT_GITHUB_REPO") or "").strip()
    if legacy:
        if "://" in legacy or legacy.endswith(".git"):
            return legacy
        return f"https://github.com/{legacy}.git"
    discovered = _discover_origin_url()
    return discovered or DEFAULT_GIT_REMOTE


def github_repo() -> str:
    """Best-effort owner/repo label for UI; derived from the git remote."""
    remote = git_remote()
    parsed = urlparse(remote)
    path = (parsed.path or "").strip("/")
    if path.endswith(".git"):
        path = path[:-4]
    if path.count("/") >= 1:
        parts = path.split("/")
        return f"{parts[-2]}/{parts[-1]}"
    return path or "drewsmash/RoboScoutAI"


def update_cache_dir() -> Path:
    override = (
        (os.environ.get("ROBOSCOUT_UPDATE_CACHE") or "").strip()
        or (os.environ.get("RAMSCOUT_UPDATE_CACHE") or "").strip()
    )
    if override:
        path = Path(override).expanduser()
    elif is_frozen():
        # Always use a per-user writable folder — never Program Files.
        local = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if local:
            path = Path(local) / APP_NAME / "update-cache"
        else:
            path = Path.home() / APP_NAME / "update-cache"
    else:
        path = app_dir() / ".roboscout-update-cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def platform_key() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system.startswith("win"):
        return "windows"
    if system == "darwin":
        return "macos-arm64" if machine in {"arm64", "aarch64"} else "macos-x64"
    return f"linux-{machine or 'x64'}"


def preferred_asset_names() -> list[str]:
    """Ordered RoboScoutAI desktop artifact names for this platform."""
    key = platform_key()
    brands = (BINARY_NAME,)
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
                    f"{brand}-macos-arm64.7z",
                ]
            )
            if key == "macos-x64":
                names.append(f"{brand}-macos-x64.zip")
        return names
    names = []
    for brand in brands:
        names.extend(
            [
                f"{brand}-{key}.tar.gz",
                f"{brand}-linux.tar.gz",
                f"{brand}-linux-x86_64.tar.gz",
            ]
        )
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


def is_older(candidate: str, current: str) -> bool:
    return version_tuple(candidate) < version_tuple(current)


def classify_update(
    *,
    current_version: str,
    remote_version: str,
    local_sha: str,
    remote_sha: str,
) -> str:
    """Return up_to_date | available | ahead (local newer than remote channel).

    Semver wins over SHA. A pinned SHA must never hide a newer remote version
    (that was the 'closed then up to date' failure mode after a botched apply).
    """
    remote_ver = normalize_version(remote_version or "")
    current_ver = normalize_version(current_version or "")
    if remote_ver and re.match(r"^\d", remote_ver) and current_ver and re.match(r"^\d", current_ver):
        if is_newer(remote_ver, current_ver):
            return "available"
        if is_older(remote_ver, current_ver):
            return "ahead"
        # Same semver: only offer when we know the installed SHA and it differs (hotfix).
        if local_sha and remote_sha and local_sha != remote_sha:
            return "available"
        return "up_to_date"

    if local_sha and remote_sha and local_sha == remote_sha:
        return "up_to_date"

    # No usable semver — require a known local SHA drift (never empty→anything).
    if local_sha and remote_sha and local_sha != remote_sha:
        return "available"
    return "up_to_date"


def source_git_root() -> Path | None:
    """Return the working tree root when running from a git checkout."""
    if is_frozen():
        return None
    start = app_dir()
    cur = start
    for _ in range(6):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None


def check_for_update(current: str | None = None, timeout: float = 60.0) -> UpdateInfo:
    current_version = normalize_version(current or __version__)
    info = UpdateInfo(
        available=False,
        current_version=current_version,
        frozen=is_frozen(),
        platform=platform_key(),
        remote=git_remote(),
        branch=git_branch(),
        release_url=_browse_url(git_remote()),
    )
    if not shutil.which("git"):
        info.error = f"git is not installed, so {APP_NAME} cannot check for updates from the remote."
        info.message = "git remote unreachable"
        return _store(info)

    root = source_git_root()
    if root is not None:
        return _store(_check_source(info, root, timeout=timeout))
    return _store(_check_frozen(info, timeout=timeout))


def last_check() -> dict[str, Any] | None:
    with _STATE_LOCK:
        return dict(_LAST_CHECK) if _LAST_CHECK else None


def download_update(info: UpdateInfo | None = None, dest_dir: Path | None = None) -> Path:
    """Fetch the update payload via git into dest_dir (or the update cache)."""
    update = info or UpdateInfo(**(last_check() or check_for_update().as_dict()))
    if not update.available and not update.remote_sha:
        raise RuntimeError(update.error or "No git update is available.")

    root = source_git_root()
    if root is not None:
        # Source mode: pull/reset; return the repo root as the "package".
        _apply_source_pull(root, update.branch)
        marker = root / ".ramscout-updated"
        marker.write_text(update.remote_sha or "ok", encoding="utf-8")
        return root

    package = _fetch_frozen_artifact(update, dest_dir=dest_dir)
    update.asset_url = str(package)
    update.asset_name = package.name
    _store(update)
    return package


def apply_downloaded_update(package: Path) -> str:
    """Apply a previously downloaded package / source tree update."""
    package = Path(package)
    root = source_git_root()
    if root is not None and package.resolve() == root.resolve():
        return f"Source tree updated from git. Restart {APP_NAME} to load the new code."

    if not is_frozen():
        raise RuntimeError("Auto-apply of a binary package only works from the desktop build.")
    if not package.is_file():
        raise RuntimeError(f"Update package missing: {package}")

    system = platform.system().lower()
    if system.startswith("win"):
        # Windows bat writes the SHA only after a successful copy.
        return _apply_windows(package)
    if system == "darwin":
        message = _apply_macos(package)
    else:
        raise RuntimeError("Auto-apply of binaries is only supported on Windows and macOS.")

    sha = (last_check() or {}).get("remote_sha") or ""
    if sha:
        _write_installed_sha(sha)
    return message


def manual_download_url(*, version: str = "", asset_name: str = "") -> str:
    """Direct GitHub Releases URL for when auto-apply fails (e.g. Access Denied)."""
    repo = github_repo()
    tag = normalize_version(version)
    name = asset_name or preferred_asset_names()[0]
    if tag:
        return f"https://github.com/{repo}/releases/download/v{tag}/{name}"
    return f"https://github.com/{repo}/releases/latest"


def apply_update_now() -> dict[str, Any]:
    """Check, fetch via git, and apply. Used by /api/updates/download."""
    info = check_for_update()
    asset = info.asset_name if info.asset_name and not str(info.asset_name).startswith("git:") else ""
    manual = manual_download_url(
        version=info.latest_version or info.current_version,
        asset_name=asset,
    )
    if info.error and not info.available:
        return {
            "ok": False,
            "message": info.error,
            "open_url": info.release_url or manual,
            "update": info.as_dict(),
        }
    if not info.available:
        return {
            "ok": True,
            "message": info.message or f"{APP_NAME} {info.current_version} is up to date.",
            "update": info.as_dict(),
            "restarting": False,
        }
    try:
        package = download_update(info)
        message = apply_downloaded_update(package)
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)
        return {
            "ok": False,
            "message": (
                f"Auto-update failed ({detail}). "
                f"Download {BINARY_NAME}-windows-x64.exe from the release page and run it "
                "(More info → Run anyway if SmartScreen blocks the unsigned build)."
            ),
            "open_url": info.release_url or manual,
            "error": detail,
            "update": info.as_dict(),
            "restarting": False,
        }
    restarting = is_frozen() and package.is_file()
    return {
        "ok": True,
        "message": message,
        "open_url": info.release_url or manual,
        "update": (last_check() or info.as_dict()),
        "restarting": restarting,
    }


def write_update_state(path: Path | None = None) -> Path:
    target = path or (app_dir() / "update-state.json")
    payload = last_check() or check_for_update().as_dict()
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


# --- internals -----------------------------------------------------------------


def _store(info: UpdateInfo) -> UpdateInfo:
    with _STATE_LOCK:
        global _LAST_CHECK
        _LAST_CHECK = info.as_dict()
    return info


def _discover_origin_url() -> str | None:
    root = source_git_root()
    if root is None:
        return None
    try:
        out = _run_git(["config", "--get", "remote.origin.url"], cwd=root, timeout=10.0)
        return out.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _browse_url(remote: str) -> str:
    text = (remote or "").strip()
    if text.startswith("git@"):
        # git@github.com:owner/repo.git → https://github.com/owner/repo
        match = re.match(r"git@([^:]+):(.+?)(?:\.git)?$", text)
        if match:
            return f"https://{match.group(1)}/{match.group(2)}"
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        path = (parsed.path or "").removesuffix(".git")
        return urlunparse(("https", parsed.netloc, path, "", "", ""))
    return text


def _auth_remote(remote: str) -> str:
    """Inject a token into https remotes when RAMSCOUT_GITHUB_TOKEN is set."""
    token = ""
    for key in ("RAMSCOUT_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN", "RAMSCOUT_GIT_TOKEN"):
        token = (os.environ.get(key) or "").strip()
        if token:
            break
    if not token:
        return remote
    parsed = urlparse(remote)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return remote
    if "@" in parsed.netloc:
        return remote
    # Prefer x-access-token for GitHub; generic user:token otherwise.
    host = parsed.netloc.lower()
    user = "x-access-token" if "github" in host else "git"
    netloc = f"{user}:{quote(token, safe='')}@{parsed.netloc}"
    return urlunparse(parsed._replace(netloc=netloc))


def _run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 60.0,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> str:
    cmd = ["git", *args]
    merged = os.environ.copy()
    # Avoid interactive prompts hanging the desktop app.
    merged.setdefault("GIT_TERMINAL_PROMPT", "0")
    merged.setdefault("GCM_INTERACTIVE", "never")
    if env:
        merged.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=merged,
    )
    if check and proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip() or f"git exited {proc.returncode}"
        raise RuntimeError(err)
    return (proc.stdout or "").strip()


def _check_source(info: UpdateInfo, root: Path, *, timeout: float) -> UpdateInfo:
    info.mode = "source"
    branch = info.branch
    remote = _auth_remote(info.remote)
    try:
        info.local_sha = _run_git(["rev-parse", "HEAD"], cwd=root, timeout=timeout)
        # Fetch by URL so tokens never land in .git/config.
        _run_git(
            ["fetch", "--quiet", remote, f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
            cwd=root,
            timeout=timeout,
        )
        info.remote_sha = _run_git(["rev-parse", f"origin/{branch}"], cwd=root, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        info.error = f"git remote unreachable: {exc}"
        info.message = "git remote unreachable"
        return info

    remote_version = _remote_version_via_show(root, branch) or ""
    info.latest_version = normalize_version(remote_version) if remote_version else info.remote_sha[:7]
    info.body = f"Local {info.local_sha[:7]} → origin/{branch} {info.remote_sha[:7]}"

    status = classify_update(
        current_version=info.current_version,
        remote_version=remote_version or info.latest_version,
        local_sha=info.local_sha,
        remote_sha=info.remote_sha,
    )
    if status == "up_to_date":
        info.available = False
        info.message = "up to date"
        info.can_apply = False
        return info
    if status == "ahead":
        info.available = False
        info.message = "up to date"
        info.can_apply = False
        info.body = (
            f"Local {info.current_version} is newer than origin/{branch} "
            f"({info.latest_version or info.remote_sha[:7]})."
        )
        return info

    info.available = True
    info.message = "update available from git"
    info.can_apply = True
    info.asset_name = f"git:{branch}@{info.remote_sha[:7]}"
    return info


def _check_frozen(info: UpdateInfo, *, timeout: float) -> UpdateInfo:
    info.mode = "frozen"
    cache = update_cache_dir()
    mirror = cache / "mirror"
    remote = _auth_remote(info.remote)
    branch = info.branch
    try:
        _ensure_mirror(mirror, remote, branch, timeout=timeout)
        info.remote_sha = _run_git(
            ["rev-parse", f"refs/remotes/origin/{branch}"], cwd=mirror, timeout=timeout
        )
        if not info.remote_sha:
            info.remote_sha = _run_git(["rev-parse", "HEAD"], cwd=mirror, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        info.error = f"git remote unreachable: {exc}"
        info.message = "git remote unreachable"
        return info

    info.local_sha = _read_installed_sha()
    remote_version = _remote_version_via_show(mirror, branch) or ""
    info.latest_version = normalize_version(remote_version) if remote_version else info.remote_sha[:7]
    info.body = f"Installed {info.local_sha[:7] or info.current_version} → {branch} {info.remote_sha[:7]}"

    status = classify_update(
        current_version=info.current_version,
        remote_version=remote_version or info.latest_version,
        local_sha=info.local_sha,
        remote_sha=info.remote_sha,
    )
    if status in {"up_to_date", "ahead"}:
        info.available = False
        info.message = "up to date"
        info.can_apply = False
        if status == "ahead":
            info.body = (
                f"Installed {info.current_version} is newer than {branch} "
                f"({info.latest_version or info.remote_sha[:7]})."
            )
        # Only pin SHA when versions already match (or no semver). Never pin when
        # a newer remote version exists — that hid failed Windows applies.
        pending = _read_pending_update()
        if pending and pending.get("sha") and pending.get("sha") == info.remote_sha:
            # Last apply never finished — keep offering / expose manual download.
            info.available = True
            info.can_apply = True
            info.message = "previous update did not finish installing"
            info.error = (
                "A previous update closed the app but did not relaunch. "
                "Click Update again, or run the exe from "
                f"%LOCALAPPDATA%\\{APP_NAME}\\ (unsigned builds need More info → Run anyway)."
            )
            if info.latest_version and info.latest_version[0].isdigit():
                info.release_url = manual_download_url(version=info.latest_version)
            return info
        if info.remote_sha and (
            not info.local_sha
            or (
                status == "up_to_date"
                and info.local_sha != info.remote_sha
                and not is_newer(info.latest_version, info.current_version)
            )
        ):
            _write_installed_sha(info.remote_sha)
            info.local_sha = info.remote_sha
        return info

    asset_rel = _find_remote_artifact(mirror, branch)
    if asset_rel is None:
        # Newer remote without a binary is NOT an applyable update — don't nag.
        info.available = False
        info.message = "up to date"
        info.can_apply = False
        info.error = (
            f"Git remote {branch}@{info.remote_sha[:7]} has no "
            f"{platform_key()} desktop binary under release-artifacts/, "
            "so nothing to apply automatically."
        )
        # Still expose a Releases URL so the UI can offer a manual download.
        if info.latest_version and info.latest_version[0].isdigit():
            info.release_url = manual_download_url(version=info.latest_version)
        return info

    info.available = True
    info.message = "update available from git"
    info.can_apply = True
    info.asset_name = Path(asset_rel).name
    info.asset_url = f"git:{asset_rel}"
    if info.latest_version and info.latest_version[0].isdigit():
        info.release_url = manual_download_url(
            version=info.latest_version, asset_name=info.asset_name
        )
    return info


def _ensure_mirror(mirror: Path, remote: str, branch: str, *, timeout: float) -> None:
    if (mirror / ".git").exists():
        # Fetch by URL (may include ephemeral token) without rewriting remote.origin.url.
        _run_git(
            ["fetch", "--depth", "1", remote, f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
            cwd=mirror,
            timeout=max(timeout, 120.0),
        )
        return

    mirror.parent.mkdir(parents=True, exist_ok=True)
    if mirror.exists():
        shutil.rmtree(mirror, ignore_errors=True)
    # Shallow clone; sparse-checkout of artifact dirs happens on download.
    _run_git(
        [
            "clone",
            "--depth",
            "1",
            "--branch",
            branch,
            "--single-branch",
            remote,
            str(mirror),
        ],
        timeout=max(timeout, 180.0),
    )


def _remote_version_via_show(repo: Path, branch: str) -> str | None:
    for spec in (f"origin/{branch}:ramscout/__init__.py", f"{branch}:ramscout/__init__.py", "HEAD:ramscout/__init__.py"):
        try:
            text = _run_git(["show", spec], cwd=repo, timeout=15.0)
        except Exception:  # noqa: BLE001
            continue
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        if match:
            return match.group(1)
    return None


def _find_remote_artifact(mirror: Path, branch: str) -> str | None:
    names = preferred_asset_names()
    # Prefer known relative paths, then scan desktop-downloads via ls-tree.
    candidates: list[str] = []
    for rel in _ARTIFACT_REL_PATHS:
        if Path(rel).name in names or any(Path(rel).name == n for n in names):
            candidates.append(rel)
    try:
        listing = _run_git(
            ["ls-tree", "-r", "--name-only", f"origin/{branch}"],
            cwd=mirror,
            timeout=30.0,
            check=False,
        )
        if not listing:
            listing = _run_git(["ls-tree", "-r", "--name-only", "HEAD"], cwd=mirror, timeout=30.0, check=False)
        for line in listing.splitlines():
            name = Path(line).name
            if name in names or name in {Path(p).name for p in _ARTIFACT_REL_PATHS}:
                if line not in candidates:
                    candidates.append(line)
    except Exception:  # noqa: BLE001
        pass

    for rel in candidates:
        name = Path(rel).name
        if name not in names and not any(n in name for n in names[:1]):
            # Still allow if it's in preferred list loosely
            if name not in names:
                continue
        try:
            _run_git(["cat-file", "-e", f"origin/{branch}:{rel}"], cwd=mirror, timeout=15.0)
            return rel
        except Exception:  # noqa: BLE001
            try:
                _run_git(["cat-file", "-e", f"HEAD:{rel}"], cwd=mirror, timeout=15.0)
                return rel
            except Exception:  # noqa: BLE001
                continue

    # Fallback: any preferred filename present in the working tree after clone.
    for name in names:
        hits = list(mirror.rglob(name))
        if hits:
            try:
                return str(hits[0].relative_to(mirror)).replace("\\", "/")
            except ValueError:
                return str(hits[0])
    return None


def _fetch_frozen_artifact(update: UpdateInfo, dest_dir: Path | None = None) -> Path:
    cache = update_cache_dir()
    mirror = cache / "mirror"
    remote = _auth_remote(update.remote or git_remote())
    branch = update.branch or git_branch()
    _ensure_mirror(mirror, remote, branch, timeout=180.0)

    rel = ""
    if update.asset_url.startswith("git:"):
        rel = update.asset_url[4:]
    if not rel:
        rel = _find_remote_artifact(mirror, branch) or ""
    if not rel:
        raise RuntimeError(
            "No desktop binary found on the git remote. "
            "Commit an artifact under release-artifacts/ or rebuild from source."
        )

    # Sparse checkout of the artifact path (and parent) then checkout blob.
    try:
        _run_git(["sparse-checkout", "init", "--cone"], cwd=mirror, timeout=30.0, check=False)
        parent = str(Path(rel).parent).replace("\\", "/")
        _run_git(["sparse-checkout", "set", parent if parent != "." else rel], cwd=mirror, timeout=60.0, check=False)
        _run_git(["checkout", "-f", f"origin/{branch}"], cwd=mirror, timeout=120.0, check=False)
    except Exception as exc:  # noqa: BLE001
        log.info("sparse-checkout fallback: %s", exc)

    source = mirror / rel
    if not source.is_file():
        # git show blob → file
        target_dir = dest_dir or (cache / "packages")
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / Path(rel).name
        blob = None
        for spec in (f"origin/{branch}:{rel}", f"HEAD:{rel}"):
            try:
                proc = subprocess.run(
                    ["git", "show", spec],
                    cwd=str(mirror),
                    capture_output=True,
                    timeout=180.0,
                    check=False,
                    env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
                )
                if proc.returncode == 0 and proc.stdout:
                    blob = proc.stdout
                    break
            except Exception:  # noqa: BLE001
                continue
        if not blob:
            raise RuntimeError(f"Could not read {rel} from git remote.")
        dest.write_bytes(blob)
        return dest

    target_dir = dest_dir or (cache / "packages")
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / source.name
    shutil.copy2(source, dest)
    return dest


def _apply_source_pull(root: Path, branch: str) -> None:
    remote = _auth_remote(git_remote())
    _run_git(
        ["fetch", remote, f"+refs/heads/{branch}:refs/remotes/origin/{branch}"],
        cwd=root,
        timeout=120.0,
    )
    # Prefer fast-forward; fall back to hard reset to origin (dev updater).
    try:
        _run_git(["merge", "--ff-only", f"origin/{branch}"], cwd=root, timeout=60.0)
    except Exception:
        _run_git(["reset", "--hard", f"origin/{branch}"], cwd=root, timeout=60.0)


def _installed_sha_path() -> Path:
    """Prefer always-writable LocalAppData — Program Files often denies writes."""
    system = platform.system().lower()
    if system.startswith("win"):
        return _windows_user_install_dir() / "update-git-sha.txt"
    if system == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME / "update-git-sha.txt"
    return app_dir() / "update-git-sha.txt"


def _pending_update_path() -> Path:
    return update_cache_dir() / "pending-update.json"


def _write_pending_update(*, sha: str, package: Path, target: Path) -> None:
    payload = {
        "sha": (sha or "").strip(),
        "package": str(package),
        "target": str(target),
    }
    path = _pending_update_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _clear_pending_update() -> None:
    try:
        _pending_update_path().unlink(missing_ok=True)
    except OSError:
        pass


def _read_pending_update() -> dict[str, Any] | None:
    path = _pending_update_path()
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        pass
    return None


def _read_installed_sha() -> str:
    path = _installed_sha_path()
    try:
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    # Legacy: older builds wrote next to the EXE.
    legacy = app_dir() / "update-git-sha.txt"
    try:
        if legacy.is_file() and legacy.resolve() != path.resolve():
            return legacy.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


def _write_installed_sha(sha: str) -> None:
    try:
        path = _installed_sha_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(sha.strip() + "\n", encoding="utf-8")
        _clear_pending_update()
    except OSError as exc:
        log.warning("Could not persist installed git sha: %s", exc)


def _apply_windows(package: Path) -> str:
    r"""Install into %LOCALAPPDATA%\RoboScoutAI, wait for PID exit, then relaunch.

    Does NOT mark the SHA installed yet — the bat writes that after a successful
    copy so a failed relaunch cannot falsely report "up to date".
    """
    exe = Path(sys.executable).resolve()
    pid = os.getpid()
    install_dir = _windows_user_install_dir()
    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / f"{BINARY_NAME}.exe"
    portable_target = _windows_install_target(exe)
    sha = (last_check() or {}).get("remote_sha") or ""
    sha_file = _installed_sha_path()
    log_file = install_dir / "update.log"
    _write_pending_update(sha=sha, package=package, target=target)

    script = Path(tempfile.gettempdir()) / "roboscout_update.bat"
    src = str(package.resolve())
    dst = str(target.resolve())
    portable = str(portable_target.resolve())
    sha_path = str(sha_file.resolve())
    log_path = str(log_file.resolve())
    lines = [
        "@echo off",
        "setlocal EnableExtensions",
        f"set PID={pid}",
        f'set "SRC={src}"',
        f'set "DST={dst}"',
        f'set "PORTABLE={portable}"',
        f'set "SHAFILE={sha_path}"',
        f'set "LOG={log_path}"',
        f'set "SHA={sha}"',
        f'set "INSTALLDIR={install_dir}"',
        'echo [%DATE% %TIME%] update start pid=%PID% > "%LOG%"',
        'echo SRC=%SRC%>> "%LOG%"',
        'echo DST=%DST%>> "%LOG%"',
        ":waitloop",
        'tasklist /FI "PID eq %PID%" 2>nul | findstr /I /C:"%PID%" >nul',
        "if not errorlevel 1 (",
        "  rem Prefer ping over timeout — timeout fails when stdin is redirected.",
        "  ping -n 2 127.0.0.1 >nul",
        "  goto waitloop",
        ")",
        'echo [%DATE% %TIME%] process exited >> "%LOG%"',
        'if not exist "%INSTALLDIR%" mkdir "%INSTALLDIR%"',
        'copy /Y "%SRC%" "%DST%" >> "%LOG%" 2>&1',
        "if errorlevel 1 (",
        '  echo COPY_FAILED >> "%LOG%"',
        '  start "" explorer.exe /select,"%SRC%"',
        "  exit /b 1",
        ")",
        'powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Unblock-File -LiteralPath $env:DST -ErrorAction SilentlyContinue } catch {}" >> "%LOG%" 2>&1',
        'if /I not "%PORTABLE%"=="%DST%" (',
        '  copy /Y "%SRC%" "%PORTABLE%" >> "%LOG%" 2>&1',
        ")",
        'if defined SHA if not "%SHA%"=="" (',
        '  >"%SHAFILE%" echo %SHA%',
        '  echo wrote sha %SHA% >> "%LOG%"',
        ")",
        'echo launching "%DST%" >> "%LOG%"',
        'start "" /D "%INSTALLDIR%" "%DST%"',
        "if errorlevel 1 (",
        '  echo START_FAILED >> "%LOG%"',
        '  start "" explorer.exe /select,"%DST%"',
        "  exit /b 1",
        ")",
        'echo OK >> "%LOG%"',
        'del "%~f0" >nul 2>nul',
        "endlocal",
        "",
    ]
    script.write_text("\r\n".join(lines), encoding="utf-8")
    # CREATE_NO_WINDOW keeps the console from flashing/closing in the user's face.
    creation = _windows_detach_flags() | 0x08000000  # CREATE_NO_WINDOW
    subprocess.Popen(
        ["cmd.exe", "/c", str(script)],
        close_fds=True,
        creationflags=creation,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    threading.Timer(1.25, lambda: os._exit(0)).start()
    return (
        f"Update downloaded to %LOCALAPPDATA%\\{APP_NAME}\\{BINARY_NAME}.exe. "
        "The app will close and relaunch. Builds are currently unsigned — if Windows "
        "SmartScreen appears, click More info → Run anyway. "
        f"If nothing opens, run that exe from File Explorer (log: %LOCALAPPDATA%\\{APP_NAME}\\update.log)."
    )


def _windows_detach_flags() -> int:
    # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS — keep bat alive after we exit.
    return 0x00000200 | 0x00000008


def _windows_user_install_dir() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    return local / APP_NAME


def _windows_install_target(exe: Path) -> Path:
    """Install / refresh target as RoboScoutAI.exe next to the running binary."""
    if exe.suffix.lower() == ".exe":
        return exe.with_name(f"{BINARY_NAME}.exe")
    return exe


def _apply_macos(package: Path) -> str:
    extract_dir = Path(tempfile.mkdtemp(prefix="roboscout-update-"))
    suffix = package.suffix.lower()
    if suffix in {".zip", ".7z"}:
        if suffix == ".7z":
            seven = shutil.which("7z") or shutil.which("7zz")
            if seven:
                subprocess.run([seven, "x", str(package), f"-o{extract_dir}", "-y"], check=True)
            else:
                raise RuntimeError(
                    "macOS update is a .7z archive but 7z is not installed. "
                    "Download the .zip from the release page, or install p7zip."
                )
        else:
            shutil.unpack_archive(package, extract_dir)
    else:
        shutil.copy2(package, extract_dir / package.name)

    replacement = _find_macos_binary(extract_dir)
    if replacement is None:
        raise RuntimeError(f"Could not find {APP_NAME} binary inside the macOS update package.")

    install_dir = Path.home() / "Library" / "Application Support" / APP_NAME
    install_dir.mkdir(parents=True, exist_ok=True)
    target = install_dir / BINARY_NAME
    shutil.copy2(replacement, target)
    os.chmod(target, 0o755)

    pid = os.getpid()
    script = Path(tempfile.gettempdir()) / "roboscout_update.sh"
    script.write_text(
        "\n".join(
            [
                "#!/bin/bash",
                f"while kill -0 {pid} 2>/dev/null; do sleep 0.5; done",
                f'chmod +x "{target}"',
                f'"{target}" >/dev/null 2>&1 &',
                f'rm -f "{script}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    os.chmod(script, 0o755)
    subprocess.Popen(["/bin/bash", str(script)], start_new_session=True)
    threading.Timer(0.75, lambda: os._exit(0)).start()
    return f"Update downloaded. {APP_NAME} will restart from {target}."


def _find_macos_binary(root: Path) -> Path | None:
    for name in (BINARY_NAME,):
        candidates = sorted(root.rglob(name))
        for path in candidates:
            if path.is_file() and os.access(path, os.X_OK):
                return path
    for path in root.rglob("*"):
        if path.is_file() and path.suffix == "" and "RoboScout" in path.name:
            return path
    return None
