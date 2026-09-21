"""Update channels for the manager: git branch/tag (primary) and GitHub Releases (fallback).

Both channels expose the same tiny interface -- read ``manifest.json`` for the
channel head and download a named artifact into a file -- so the install logic
does not care where bytes come from. Downloads always land in a fresh
``app\\<version>\\`` folder; the running app is never overwritten.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from roboscout_manager import (
    APP_NAME,
    MANAGER_EXE_NAME,
    MANAGER_VERSION,
    MANIFEST_NAME,
    RELEASE_ARTIFACTS_DIR,
)
from roboscout_manager.install import install_app_payload, prune_versions
from roboscout_manager.ipc import write_status
from roboscout_manager.manifest import (
    Artifact,
    Manifest,
    ManifestError,
    VerificationError,
    compare_versions,
    is_newer,
    looks_like_version,
    manager_satisfies,
    normalize_version,
    parse_manifest,
    verify_file,
    version_tuple,
)
from roboscout_manager.state import (
    ManagerConfig,
    app_version_dir,
    append_log,
    cache_dir,
    manager_exe_path,
    read_config,
    read_current,
    running_exe,
)

log = logging.getLogger("roboscout.manager")

ProgressFn = Callable[[str, int, str], None]
USER_AGENT = f"{APP_NAME}-manager/{MANAGER_VERSION}"
_TOKEN_ENVS = ("ROBOSCOUT_GITHUB_TOKEN", "RAMSCOUT_GITHUB_TOKEN", "GITHUB_TOKEN", "GH_TOKEN", "RAMSCOUT_GIT_TOKEN")


class ChannelError(RuntimeError):
    pass


def github_token() -> str:
    for key in _TOKEN_ENVS:
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return ""


def _noop_progress(_state: str, _pct: int, _msg: str) -> None:
    return None


# --- git ------------------------------------------------------------------------


def run_git(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: float = 120.0,
    check: bool = True,
    stdout_file: Path | None = None,
) -> str:
    """Run git non-interactively; optionally stream stdout into ``stdout_file``."""
    git = shutil.which("git")
    if not git:
        raise ChannelError("git is not installed")
    env = os.environ.copy()
    env.setdefault("GIT_TERMINAL_PROMPT", "0")
    env.setdefault("GCM_INTERACTIVE", "never")
    cmd = [git]
    token = github_token()
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        cmd += ["-c", f"http.extraHeader=Authorization: Basic {basic}"]
    cmd += args
    creation = 0x08000000 if sys.platform.startswith("win") else 0
    if stdout_file is not None:
        stdout_file.parent.mkdir(parents=True, exist_ok=True)
        with open(stdout_file, "wb") as handle:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=handle,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,
                env=env,
                creationflags=creation,
            )
        if check and proc.returncode != 0:
            raise ChannelError((proc.stderr or b"").decode("utf-8", "replace").strip() or f"git exited {proc.returncode}")
        return ""
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=env,
        creationflags=creation,
    )
    if check and proc.returncode != 0:
        raise ChannelError((proc.stderr or proc.stdout or "").strip() or f"git exited {proc.returncode}")
    return (proc.stdout or "").strip()


class GitChannel:
    """Blob-less shallow mirror of one branch/tag; artifacts are fetched lazily."""

    name = "git"

    def __init__(self, remote: str, ref: str, mirror: Path, *, git: Callable[..., str] = run_git) -> None:
        self.remote = remote
        self.ref = ref
        self.mirror = Path(mirror)
        self._git = git
        self.sha = ""

    @property
    def local_ref(self) -> str:
        return f"refs/channel/{self.ref}"

    def fetch(self, *, timeout: float = 180.0) -> str:
        mirror = self.mirror
        if not (mirror / ".git").is_dir() and not (mirror / "HEAD").is_file():
            mirror.mkdir(parents=True, exist_ok=True)
            self._git(["init", "--quiet", str(mirror)], timeout=30.0)
            self._git(["remote", "add", "origin", self.remote], cwd=mirror, timeout=30.0, check=False)
        else:
            self._git(["remote", "set-url", "origin", self.remote], cwd=mirror, timeout=30.0, check=False)
        errors: list[str] = []
        for kind in ("heads", "tags"):
            refspec = f"+refs/{kind}/{self.ref}:{self.local_ref}"
            try:
                self._git(
                    ["fetch", "--quiet", "--depth", "1", "--filter=blob:none", "origin", refspec],
                    cwd=mirror,
                    timeout=timeout,
                )
                break
            except ChannelError as exc:
                errors.append(str(exc))
                # Older servers without partial-clone support: retry without the filter.
                try:
                    self._git(["fetch", "--quiet", "--depth", "1", "origin", refspec], cwd=mirror, timeout=timeout)
                    break
                except ChannelError as exc2:
                    errors.append(str(exc2))
        else:
            raise ChannelError(f"could not fetch channel {self.ref!r} from {self.remote}: {errors[-1] if errors else 'unknown'}")
        self.sha = self._git(["rev-parse", self.local_ref], cwd=mirror, timeout=30.0)
        return self.sha

    def read_text(self, rel_path: str) -> str | None:
        try:
            return self._git(["cat-file", "-p", f"{self.local_ref}:{rel_path}"], cwd=self.mirror, timeout=120.0)
        except ChannelError:
            return None

    def list_files(self, rel_dir: str = RELEASE_ARTIFACTS_DIR) -> list[str]:
        try:
            out = self._git(["ls-tree", "--name-only", f"{self.local_ref}:{rel_dir}"], cwd=self.mirror, timeout=30.0)
        except ChannelError:
            return []
        return [line.strip() for line in out.splitlines() if line.strip()]

    def read_manifest(self) -> Manifest | None:
        text = self.read_text(f"{RELEASE_ARTIFACTS_DIR}/{MANIFEST_NAME}")
        if text is None:
            return None
        return parse_manifest(text)

    def read_source_version(self) -> str:
        text = self.read_text("ramscout/__init__.py") or ""
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', text)
        return normalize_version(match.group(1)) if match else ""

    def download(self, artifact_name: str, dest: Path, progress: ProgressFn = _noop_progress, *, expected_size: int = 0) -> Path:
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        stop = threading.Event()

        def watch() -> None:
            while not stop.wait(0.5):
                try:
                    size = dest.stat().st_size
                except OSError:
                    continue
                if expected_size:
                    pct = 20 + int(60 * min(1.0, size / expected_size))
                    progress("downloading", pct, f"{size // (1024 * 1024)} / {expected_size // (1024 * 1024)} MB")

        watcher = threading.Thread(target=watch, daemon=True)
        watcher.start()
        try:
            self._git(
                ["cat-file", "blob", f"{self.local_ref}:{RELEASE_ARTIFACTS_DIR}/{artifact_name}"],
                cwd=self.mirror,
                timeout=1800.0,
                stdout_file=dest,
            )
        finally:
            stop.set()
            watcher.join(timeout=2.0)
        return dest

    def browse_url(self) -> str:
        text = self.remote
        if text.endswith(".git"):
            text = text[:-4]
        return f"{text}/tree/{self.ref}"


# --- GitHub Releases ----------------------------------------------------------------


def http_get(url: str, *, headers: dict[str, str] | None = None, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return resp.read()
    except urllib.error.HTTPError as exc:
        raise ChannelError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChannelError(f"network error for {url}: {exc}") from exc


def http_download(
    url: str,
    dest: Path,
    *,
    headers: dict[str, str] | None = None,
    progress: ProgressFn = _noop_progress,
    timeout: float = 60.0,
) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as handle:  # noqa: S310
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            last = -1
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                done += len(chunk)
                if total:
                    pct = 20 + int(60 * min(1.0, done / total))
                    if pct != last:
                        last = pct
                        progress("downloading", pct, f"{done // (1024 * 1024)} / {total // (1024 * 1024)} MB")
    except urllib.error.HTTPError as exc:
        raise ChannelError(f"HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ChannelError(f"network error for {url}: {exc}") from exc
    return dest


class ReleasesChannel:
    """GitHub Releases: newest semver tag (or an exact tag) carrying manifest.json."""

    name = "releases"

    def __init__(
        self,
        repo: str,
        *,
        tag: str | None = None,
        get: Callable[..., bytes] = http_get,
        download: Callable[..., Path] = http_download,
        token: str | None = None,
    ) -> None:
        self.repo = repo
        self.tag = tag
        self._get = get
        self._download = download
        self._token = github_token() if token is None else token
        self.release: dict[str, Any] | None = None
        self.sha = ""

    def _headers(self, accept: str = "application/vnd.github+json") -> dict[str, str]:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def fetch(self, *, timeout: float = 30.0) -> str:
        url = f"https://api.github.com/repos/{self.repo}/releases?per_page=30"
        data = json.loads(self._get(url, headers=self._headers(), timeout=timeout) or b"[]")
        if not isinstance(data, list):
            raise ChannelError("unexpected GitHub releases payload")
        candidates = []
        for rel in data:
            if not isinstance(rel, dict) or rel.get("draft"):
                continue
            tag = str(rel.get("tag_name") or "")
            names = {str(a.get("name")) for a in rel.get("assets") or [] if isinstance(a, dict)}
            if MANIFEST_NAME not in names:
                continue
            if self.tag:
                if normalize_version(tag) == normalize_version(self.tag):
                    candidates.append(rel)
                continue
            if rel.get("prerelease"):
                continue
            if looks_like_version(tag):
                candidates.append(rel)
        if not candidates:
            raise ChannelError(
                f"no GitHub release of {self.repo} carries {MANIFEST_NAME}" + (f" for tag {self.tag}" if self.tag else "")
            )
        candidates.sort(key=lambda r: version_tuple(str(r.get("tag_name") or "")), reverse=True)
        self.release = candidates[0]
        self.sha = str(self.release.get("target_commitish") or "")
        return self.sha

    def _asset(self, name: str) -> dict[str, Any]:
        if self.release is None:
            self.fetch()
        for asset in (self.release or {}).get("assets") or []:
            if isinstance(asset, dict) and str(asset.get("name")) == name:
                return asset
        raise ChannelError(f"release {self.release.get('tag_name') if self.release else '?'} has no asset {name}")

    def _asset_url(self, asset: dict[str, Any]) -> tuple[str, dict[str, str]]:
        if self._token and asset.get("url"):
            return str(asset["url"]), self._headers("application/octet-stream")
        return str(asset.get("browser_download_url") or asset.get("url")), {"Accept": "application/octet-stream"}

    def read_manifest(self) -> Manifest | None:
        asset = self._asset(MANIFEST_NAME)
        url, headers = self._asset_url(asset)
        return parse_manifest(self._get(url, headers=headers, timeout=60.0))

    def download(self, artifact_name: str, dest: Path, progress: ProgressFn = _noop_progress, *, expected_size: int = 0) -> Path:
        asset = self._asset(artifact_name)
        url, headers = self._asset_url(asset)
        return self._download(url, dest, headers=headers, progress=progress)

    def browse_url(self) -> str:
        if self.release and self.release.get("html_url"):
            return str(self.release["html_url"])
        return f"https://github.com/{self.repo}/releases"


# --- check ------------------------------------------------------------------------


@dataclass
class UpdateCheck:
    current_version: str
    manager_version: str = MANAGER_VERSION
    latest_version: str = ""
    available: bool = False
    needs_manager_update: bool = False
    channel: str = ""
    source: str = ""  # git | releases
    remote: str = ""
    remote_sha: str = ""
    message: str = ""
    error: str = ""
    errors: list[str] = field(default_factory=list)
    release_url: str = ""
    manifest: Manifest | None = None
    channel_obj: Any = None

    @property
    def can_apply(self) -> bool:
        return bool(self.available and self.manifest is not None and self.manifest.app_artifact() is not None)

    def as_dict(self) -> dict[str, Any]:
        """UI-compatible payload (mirrors the legacy ramscout.updater UpdateInfo keys)."""
        art = self.manifest.app_artifact() if self.manifest else None
        return {
            "available": self.available,
            "can_apply": self.can_apply,
            "needs_manager_update": self.needs_manager_update,
            "current_version": self.current_version,
            "latest_version": self.latest_version,
            "manager_version": self.manager_version,
            "min_manager_version": self.manifest.min_manager_version if self.manifest else "",
            "channel": self.channel,
            "branch": self.channel,
            "source": self.source,
            "remote": self.remote,
            "remote_sha": self.remote_sha,
            "local_sha": "",
            "mode": "managed",
            "frozen": bool(getattr(sys, "frozen", False)),
            "platform": "windows" if sys.platform.startswith("win") else sys.platform,
            "message": self.message,
            "error": self.error or None,
            "errors": list(self.errors),
            "release_url": self.release_url,
            "asset_name": art.name if art else "",
            "asset_url": "",
            "asset_size": art.size if art else 0,
            "body": (
                f"Channel {self.channel} ({self.source}) -> {self.latest_version}"
                if self.latest_version
                else ""
            ),
            "manifest": self.manifest.to_dict() if self.manifest else None,
        }


def _channel_is_tag(channel: str) -> bool:
    return looks_like_version(channel)


def download_artifact(
    channel: Any,
    artifact: Artifact,
    dest: Path,
    *,
    progress: ProgressFn = _noop_progress,
    github_repo: str = "",
    release_tag: str = "",
    releases_factory: Callable[[str, str], ReleasesChannel] | None = None,
) -> Path:
    """Download ``artifact`` via the channel, then ``artifact.url``, then GitHub Releases.

    Large app builds no longer fit in the git update channel (GitHub's 100 MB
    blob limit). The channel still carries ``manifest.json`` + the thin manager;
    the Windows app is fetched from the matching GitHub Release asset.
    """
    errors: list[str] = []
    try:
        return channel.download(artifact.name, dest, progress, expected_size=artifact.size)
    except ChannelError as exc:
        errors.append(f"{getattr(channel, 'name', 'channel')}: {exc}")

    if artifact.url:
        try:
            return http_download(artifact.url, dest, progress=progress)
        except ChannelError as exc:
            errors.append(f"url: {exc}")

    repo = (github_repo or "").strip()
    tag = normalize_version(release_tag or "")
    if repo and tag:
        release_ref = tag if tag.startswith("v") else f"v{tag}"
        try:
            if releases_factory is not None:
                rel = releases_factory(repo, release_ref)
            else:
                rel = ReleasesChannel(repo, tag=release_ref)
            rel.fetch()
            return rel.download(artifact.name, dest, progress, expected_size=artifact.size)
        except (ChannelError, ManifestError, json.JSONDecodeError) as exc:
            errors.append(f"releases: {exc}")

    raise ChannelError("; ".join(errors) or f"could not download {artifact.name}")


def check_for_update(
    root: Path,
    *,
    cfg: ManagerConfig | None = None,
    current_version: str | None = None,
    manager_version: str = MANAGER_VERSION,
    git_factory: Callable[[ManagerConfig, Path], GitChannel] | None = None,
    releases_factory: Callable[[ManagerConfig], ReleasesChannel] | None = None,
    progress: ProgressFn = _noop_progress,
    write_status_file: bool = False,
) -> UpdateCheck:
    root = Path(root)
    cfg = cfg or read_config(root)
    current = read_current(root)
    cur_version = normalize_version(current_version if current_version is not None else (current.version if current else ""))
    check = UpdateCheck(current_version=cur_version, manager_version=manager_version, channel=cfg.channel, remote=cfg.remote)
    if write_status_file:
        write_status(root, "checking", progress=5, message=f"Checking {cfg.channel}…", current_version=cur_version)
    progress("checking", 5, f"checking channel {cfg.channel}")

    manifest: Manifest | None = None
    channel_obj: Any = None

    def make_git(c: ManagerConfig, r: Path) -> GitChannel:
        return GitChannel(c.remote, c.channel, cache_dir(r) / "mirror")

    def make_releases(c: ManagerConfig) -> ReleasesChannel:
        return ReleasesChannel(c.github_repo, tag=c.channel if _channel_is_tag(c.channel) else None)

    try:
        git_channel = (git_factory or make_git)(cfg, root)
        check.remote_sha = git_channel.fetch()
        manifest = git_channel.read_manifest()
        if manifest is None:
            src_version = git_channel.read_source_version()
            raise ChannelError(
                f"channel {cfg.channel} @ {check.remote_sha[:7]} has no {RELEASE_ARTIFACTS_DIR}/{MANIFEST_NAME}"
                + (f" (source version {src_version})" if src_version else "")
            )
        channel_obj = git_channel
        check.source = "git"
        check.release_url = git_channel.browse_url()
    except (ChannelError, ManifestError) as exc:
        check.errors.append(f"git: {exc}")
        log.info("git channel unavailable: %s", exc)

    if manifest is None and cfg.releases_fallback:
        try:
            rel = (releases_factory or make_releases)(cfg)
            sha = rel.fetch()
            manifest = rel.read_manifest()
            channel_obj = rel
            check.source = "releases"
            check.remote_sha = sha or check.remote_sha
            check.release_url = rel.browse_url()
        except (ChannelError, ManifestError, json.JSONDecodeError) as exc:
            check.errors.append(f"releases: {exc}")
            log.info("releases channel unavailable: %s", exc)

    if manifest is None:
        check.error = "; ".join(check.errors) or "no update channel reachable"
        check.message = "update channel unreachable"
        if write_status_file:
            write_status(root, "error", message=check.message, error=check.error, current_version=cur_version)
        return check

    check.manifest = manifest
    check.channel_obj = channel_obj
    check.latest_version = manifest.version
    check.needs_manager_update = not manager_satisfies(manifest, manager_version)
    if not cur_version:
        check.available = True
        check.message = f"{APP_NAME} {manifest.version} is ready to install"
    elif is_newer(manifest.version, cur_version):
        check.available = True
        check.message = f"update available: {cur_version} → {manifest.version}"
    elif compare_versions(manifest.version, cur_version) < 0:
        check.message = f"installed {cur_version} is newer than channel {cfg.channel} ({manifest.version})"
    else:
        check.message = "up to date"
    if check.available and manifest.app_artifact() is None:
        check.available = False
        check.error = f"channel {cfg.channel} manifest {manifest.version} lists no Windows app artifact"
    if check.needs_manager_update and check.available:
        check.message += f" (manager {manager_version} → {manifest.min_manager_version} first)"
    if write_status_file:
        write_status(
            root,
            "idle",
            progress=0,
            message=check.message,
            version=check.latest_version,
            current_version=cur_version,
            error=check.error,
        )
    return check


# --- apply ----------------------------------------------------------------------


@dataclass
class UpdateResult:
    ok: bool
    message: str
    installed_version: str = ""
    manager_update_scheduled: bool = False
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "message": self.message,
            "installed_version": self.installed_version,
            "manager_update_scheduled": self.manager_update_scheduled,
            "error": self.error or None,
        }


def perform_update(
    root: Path,
    check: UpdateCheck | None = None,
    *,
    cfg: ManagerConfig | None = None,
    progress: ProgressFn = _noop_progress,
    force: bool = False,
    relaunch_after_manager_swap: list[str] | None = None,
    spawn_swap: bool = True,
) -> UpdateResult:
    """Download + verify + install into a new side-by-side folder, then flip current.json."""
    root = Path(root)
    cfg = cfg or read_config(root)
    check = check or check_for_update(root, cfg=cfg, progress=progress)

    def report(state: str, pct: int, message: str, **extra: Any) -> None:
        write_status(
            root,
            state,
            progress=pct,
            message=message,
            version=check.latest_version,
            current_version=check.current_version,
            error=str(extra.get("error") or ""),
        )
        progress(state, pct, message)

    if check.manifest is None or check.channel_obj is None:
        report("error", 0, check.message or "update channel unreachable", error=check.error)
        return UpdateResult(ok=False, message=check.message or "update channel unreachable", error=check.error)
    if not check.available and not force:
        report("done", 100, check.message or "up to date")
        return UpdateResult(ok=True, message=check.message or "up to date", installed_version=check.current_version)

    manifest = check.manifest
    channel = check.channel_obj

    if check.needs_manager_update:
        return self_update_manager(
            root,
            manifest,
            channel,
            progress=progress,
            relaunch_args=relaunch_after_manager_swap,
            spawn_swap=spawn_swap,
            current_version=check.current_version,
        )

    art = manifest.app_artifact()
    if art is None:
        report("error", 0, "no app artifact in manifest", error="manifest lists no app build")
        return UpdateResult(ok=False, message="manifest lists no app build", error="no app artifact")

    dest_dir = app_version_dir(root, manifest.version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest_dir / ".download.part"
    try:
        report("downloading", 20, f"Downloading {APP_NAME} {manifest.version} ({art.size // (1024 * 1024)} MB)…")
        download_artifact(
            channel,
            art,
            tmp,
            progress=lambda s, p, m: report(s, p, m),
            github_repo=cfg.github_repo,
            release_tag=manifest.version,
        )
        report("verifying", 85, "Verifying download…")
        verify_file(tmp, art)
        report("installing", 92, f"Installing {manifest.version}…")
        state = install_app_payload(
            root,
            tmp,
            manifest.version,
            sha256=art.sha256,
            size=art.size,
            channel=cfg.channel,
            source=check.source,
            move=True,
        )
        removed = prune_versions(root)
        if removed:
            append_log(root, f"pruned {', '.join(removed)}")
        report("done", 100, f"{APP_NAME} {state.version} installed. Restart to use it.")
        return UpdateResult(ok=True, message=f"{APP_NAME} {state.version} installed", installed_version=state.version)
    except (ChannelError, VerificationError, OSError, RuntimeError) as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        append_log(root, f"update to {manifest.version} failed: {exc}")
        report("error", 0, f"Update failed: {exc}", error=str(exc))
        return UpdateResult(ok=False, message=f"Update failed: {exc}", error=str(exc))


# --- manager self-update --------------------------------------------------------------


def manager_update_bat_lines(*, pid: int, staged: str, target: str, log_path: str, relaunch_args: list[str]) -> list[str]:
    """Wait for the manager PID, replace RoboScoutAI.exe with the staged .new, relaunch."""
    args = " ".join(f'"{a}"' if " " in a else a for a in relaunch_args)
    return [
        "@echo off",
        "setlocal EnableExtensions",
        f"set PID={pid}",
        f'set "SRC={staged}"',
        f'set "DST={target}"',
        f'set "LOG={log_path}"',
        'echo [%DATE% %TIME%] manager swap start pid=%PID% >> "%LOG%"',
        'powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Wait-Process -Id ([int]$env:PID) -Timeout 180 -ErrorAction Stop } catch { }" >> "%LOG%" 2>&1',
        "ping -n 2 127.0.0.1 >nul",
        "set TRIES=0",
        ":retry",
        'copy /Y "%SRC%" "%DST%" >> "%LOG%" 2>&1',
        "if errorlevel 1 (",
        "  set /a TRIES+=1",
        "  if %TRIES% LSS 10 (",
        "    ping -n 2 127.0.0.1 >nul",
        "    goto retry",
        "  )",
        '  echo MANAGER_SWAP_FAILED >> "%LOG%"',
        '  start "" explorer.exe /select,"%SRC%"',
        "  exit /b 1",
        ")",
        'del /F /Q "%SRC%" >nul 2>nul',
        'echo manager swapped >> "%LOG%"',
        f'start "" "%DST%" {args}'.rstrip(),
        "endlocal",
        '(goto) 2>nul & del "%~f0"',
        "",
    ]


def self_update_manager(
    root: Path,
    manifest: Manifest,
    channel: Any,
    *,
    progress: ProgressFn = _noop_progress,
    relaunch_args: list[str] | None = None,
    spawn_swap: bool = True,
    current_version: str = "",
) -> UpdateResult:
    """Stage a newer manager binary; swap it in after this process exits."""
    root = Path(root)
    art: Artifact | None = manifest.manager_artifact()

    def report(state: str, pct: int, message: str, error: str = "") -> None:
        write_status(root, state, progress=pct, message=message, version=manifest.version, current_version=current_version, error=error)
        progress(state, pct, message)

    if art is None:
        msg = (
            f"{APP_NAME} {manifest.version} needs manager {manifest.min_manager_version}+ but the channel has no "
            f"{MANAGER_EXE_NAME}; download {APP_NAME}-Setup.exe from GitHub Releases."
        )
        report("error", 0, msg, error="manager artifact missing")
        return UpdateResult(ok=False, message=msg, error="manager artifact missing")

    target = manager_exe_path(root)
    staged = root / f"{MANAGER_EXE_NAME}.new"
    try:
        report("manager-update", 10, f"Updating manager to {manifest.min_manager_version}+…")
        cfg = read_config(root)
        download_artifact(
            channel,
            art,
            staged,
            progress=lambda s, p, m: report("manager-update", p, m),
            github_repo=cfg.github_repo,
            release_tag=manifest.version,
        )
        verify_file(staged, art)
    except (ChannelError, VerificationError, OSError) as exc:
        try:
            staged.unlink(missing_ok=True)
        except OSError:
            pass
        report("error", 0, f"Manager update failed: {exc}", error=str(exc))
        return UpdateResult(ok=False, message=f"Manager update failed: {exc}", error=str(exc))

    running_here = False
    try:
        running_here = target.is_file() and running_exe().resolve() == target.resolve()
    except OSError:
        running_here = False
    if not running_here:
        # Not locked (dev run, or Setup running from Downloads): swap immediately.
        os.replace(staged, target)
        append_log(root, f"manager updated in place to satisfy {manifest.min_manager_version}")
        report("manager-update", 100, "Manager updated; continuing…")
        return UpdateResult(ok=True, message="manager updated", manager_update_scheduled=False)

    args = relaunch_args if relaunch_args is not None else ["--update", "--launch-after"]
    schedule_manager_swap(root, staged, relaunch_args=args, spawn=spawn_swap)
    report("manager-update", 100, "Manager will restart to finish the update…")
    return UpdateResult(ok=True, message="manager update scheduled; restarting", manager_update_scheduled=True)


def schedule_manager_swap(root: Path, staged: Path, *, relaunch_args: list[str], spawn: bool = True) -> Path:
    """Write (and on Windows spawn) the .bat that swaps RoboScoutAI.exe after we exit."""
    root = Path(root)
    target = manager_exe_path(root)
    script = Path(tempfile.gettempdir()) / "RoboScoutAI-manager-update.bat"
    lines = manager_update_bat_lines(
        pid=os.getpid(),
        staged=str(staged),
        target=str(target),
        log_path=str(root / "update.log"),
        relaunch_args=relaunch_args,
    )
    script.write_text("\r\n".join(lines), encoding="utf-8")
    append_log(root, f"manager swap scheduled via {script}")
    if spawn and sys.platform.startswith("win"):
        creation = 0x00000200 | 0x00000008 | 0x08000000
        subprocess.Popen(
            ["cmd.exe", "/c", str(script)],
            close_fds=True,
            creationflags=creation,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return script


def repair(root: Path, *, cfg: ManagerConfig | None = None, progress: ProgressFn = _noop_progress) -> UpdateResult:
    """Re-download the channel's current build when the installed one is damaged."""
    root = Path(root)
    cfg = cfg or read_config(root)
    check = check_for_update(root, cfg=cfg, current_version="", progress=progress)
    if check.manifest is None:
        return UpdateResult(ok=False, message=check.message, error=check.error)
    check.available = True
    return perform_update(root, check, cfg=cfg, progress=progress, force=True)


def wait_for_status_pid_exit(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.1)
