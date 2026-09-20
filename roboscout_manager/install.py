"""Side-by-side app installs, pruning and the Setup/uninstall flows.

Every app build is copied into ``app\\<version>\\RoboScoutAI-app.exe``; nothing is
ever written over a running binary. Switching versions is a single atomic
``current.json`` write, which is what makes rollback trivial.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from roboscout_manager import (
    APP_ASSET_NAME,
    APP_NAME,
    INSTALLED_APP_EXE_NAME,
    KEEP_VERSIONS,
    MANAGER_ASSET_NAME,
    MANAGER_EXE_NAME,
    MANAGER_VERSION,
    MANIFEST_NAME,
)
from roboscout_manager.manifest import (
    Manifest,
    VerificationError,
    compare_versions,
    load_manifest,
    looks_like_version,
    sha256_file,
    verify_file,
)
from roboscout_manager.state import (
    CurrentState,
    ManagerConfig,
    app_version_dir,
    app_versions_dir,
    append_log,
    current_json_path,
    ipc_json_path,
    manager_exe_path,
    manager_json_path,
    read_config,
    read_current,
    relative_app_exe,
    running_exe,
    status_json_path,
    update_log_path,
    write_config,
    write_current,
)

log = logging.getLogger("roboscout.manager")


class InstallError(RuntimeError):
    pass


# --- side-by-side app versions ---------------------------------------------------


def install_app_payload(
    root: Path,
    payload: Path,
    version: str,
    *,
    sha256: str = "",
    size: int | None = None,
    channel: str = "",
    source: str = "local",
    move: bool = False,
) -> CurrentState:
    """Place ``payload`` at app/<version>/RoboScoutAI-app.exe and switch current.json.

    The payload is verified (when a digest is known), copied under a temporary
    name inside the version folder, renamed into place, and only then does
    ``current.json`` flip. A previous active version is remembered for rollback.
    """
    root = Path(root)
    payload = Path(payload)
    version = version.strip()
    if not looks_like_version(version):
        raise InstallError(f"refusing to install unversioned build {version!r}")
    if not payload.is_file():
        raise InstallError(f"payload missing: {payload}")
    digest = verify_file(payload, sha256=sha256, size=size)

    dest_dir = app_version_dir(root, version)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / INSTALLED_APP_EXE_NAME
    if dest.is_file() and payload.resolve() == dest.resolve():
        pass  # already in place (repair/rollback path)
    elif dest.is_file() and sha256_file(dest) == digest:
        if move:
            payload.unlink(missing_ok=True)
    else:
        tmp = dest_dir / f".{INSTALLED_APP_EXE_NAME}.part"
        if move:
            shutil.move(str(payload), str(tmp))
        else:
            shutil.copy2(payload, tmp)
        os.replace(tmp, dest)
    try:
        os.chmod(dest, 0o755)
    except OSError:
        pass

    previous = read_current(root)
    state = CurrentState(
        version=version,
        exe=relative_app_exe(version),
        sha256=digest,
        size=dest.stat().st_size,
        channel=channel,
        source=source,
    )
    if previous is not None and previous.version != version:
        state.previous_version = previous.version
        state.previous_sha256 = previous.sha256
    elif previous is not None:
        state.previous_version = previous.previous_version
        state.previous_sha256 = previous.previous_sha256
    write_current(root, state)
    append_log(root, f"installed app {version} from {source} ({digest[:12]})")
    return state


def installed_versions(root: Path) -> list[str]:
    """Versions that have an app exe on disk, newest first."""
    base = app_versions_dir(root)
    if not base.is_dir():
        return []
    found = [p.name for p in base.iterdir() if p.is_dir() and (p / INSTALLED_APP_EXE_NAME).is_file() and looks_like_version(p.name)]
    return sorted(found, key=lambda v: tuple(int(x) for x in _ver_parts(v)), reverse=True)


def _ver_parts(version: str) -> list[str]:
    import re

    return [c for c in re.split(r"[^\d]+", version) if c.isdigit()] or ["0"]


def prune_versions(root: Path, keep: int = KEEP_VERSIONS) -> list[str]:
    """Delete old side-by-side versions, never the active or the rollback target."""
    current = read_current(root)
    protected: set[str] = set()
    if current is not None:
        protected.add(current.version)
        if current.previous_version:
            protected.add(current.previous_version)
    versions = installed_versions(root)
    keepers: list[str] = []
    for ver in versions:
        if ver in protected or len(keepers) < max(keep, len(protected)):
            keepers.append(ver)
    removed: list[str] = []
    for ver in versions:
        if ver in keepers or ver in protected:
            continue
        target = app_version_dir(root, ver)
        try:
            shutil.rmtree(target)
            removed.append(ver)
            append_log(root, f"pruned app {ver}")
        except OSError as exc:
            log.warning("could not prune %s: %s", target, exc)
    # Stray partial downloads.
    for stray in app_versions_dir(root).glob("*/*.part") if app_versions_dir(root).is_dir() else []:
        try:
            stray.unlink()
        except OSError:
            pass
    return removed


def verify_current(root: Path) -> tuple[CurrentState | None, str]:
    """Return (state, problem). problem == "" when the active app is launchable."""
    state = read_current(root)
    if state is None:
        return None, "current.json missing or invalid"
    exe = state.exe_path(root)
    if not exe.is_file():
        return state, f"app exe missing: {exe}"
    if state.sha256:
        try:
            verify_file(exe, sha256=state.sha256, size=state.size or None)
        except VerificationError as exc:
            return state, str(exc)
    return state, ""


# --- manager binary ----------------------------------------------------------------


def install_manager_binary(root: Path, source: Path | None = None) -> Path:
    """Copy the manager exe into the install root (skips when already running there)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    target = manager_exe_path(root)
    src = Path(source) if source else running_exe()
    if not src.is_file():
        raise InstallError(f"manager binary missing: {src}")
    try:
        if target.is_file() and target.resolve() == src.resolve():
            return target
    except OSError:
        pass
    if target.is_file() and sha256_file(target) == sha256_file(src):
        return target
    tmp = root / f".{MANAGER_EXE_NAME}.part"
    shutil.copy2(src, tmp)
    try:
        os.replace(tmp, target)
    except PermissionError:
        # Running manager is locked: stage .new; the launcher swaps it on next start.
        staged = root / f"{MANAGER_EXE_NAME}.new"
        os.replace(tmp, staged)
        append_log(root, f"manager binary locked; staged {staged.name}")
        return staged
    try:
        os.chmod(target, 0o755)
    except OSError:
        pass
    return target


def swap_staged_manager(root: Path) -> bool:
    """Apply RoboScoutAI.exe.new when we are *not* running from the target."""
    root = Path(root)
    staged = root / f"{MANAGER_EXE_NAME}.new"
    target = manager_exe_path(root)
    if not staged.is_file():
        return False
    try:
        if running_exe().resolve() == target.resolve():
            return False
    except OSError:
        return False
    try:
        os.replace(staged, target)
        append_log(root, "applied staged manager binary")
        return True
    except OSError as exc:
        log.warning("could not swap staged manager: %s", exc)
        return False


# --- setup ---------------------------------------------------------------------


@dataclass
class SetupOptions:
    root: Path
    payload_dir: Path | None = None
    desktop_shortcut: bool = False
    start_menu_shortcut: bool = True
    register: bool = True
    launch: bool = True
    channel: str = ""
    silent: bool = False
    manager_source: Path | None = None


@dataclass
class SetupResult:
    root: Path
    version: str = ""
    manager_exe: Path | None = None
    app_exe: Path | None = None
    shortcuts: list[Path] = field(default_factory=list)
    registered: bool = False
    needs_download: bool = False
    messages: list[str] = field(default_factory=list)


def find_payload(payload_dir: Path | None) -> tuple[Path | None, Path | None, Manifest | None]:
    """Locate (app exe, manager exe, manifest) inside a Setup payload folder."""
    if payload_dir is None or not Path(payload_dir).is_dir():
        return None, None, None
    payload_dir = Path(payload_dir)
    manifest = None
    mpath = payload_dir / MANIFEST_NAME
    if mpath.is_file():
        try:
            manifest = load_manifest(mpath)
        except Exception as exc:  # noqa: BLE001
            log.warning("payload manifest unreadable: %s", exc)
    app = payload_dir / APP_ASSET_NAME
    manager = payload_dir / MANAGER_ASSET_NAME
    return (app if app.is_file() else None), (manager if manager.is_file() else None), manifest


def run_setup(
    opts: SetupOptions,
    *,
    version_fallback: str = MANAGER_VERSION,
    shortcut_fn: Callable[..., list[Path]] | None = None,
    register_fn: Callable[..., bool] | None = None,
) -> SetupResult:
    """Install manager + bundled app payload, config, shortcuts and uninstall entry."""
    root = Path(opts.root)
    root.mkdir(parents=True, exist_ok=True)
    result = SetupResult(root=root)

    app_payload, manager_payload, manifest = find_payload(opts.payload_dir)
    manager_src = opts.manager_source or manager_payload
    result.manager_exe = install_manager_binary(root, manager_src)
    result.messages.append(f"manager -> {result.manager_exe}")

    cfg: ManagerConfig = read_config(root)
    if opts.channel:
        cfg.channel = opts.channel
    elif manifest is not None and manifest.channel and not manager_json_path(root).is_file():
        cfg.channel = manifest.channel
    cfg.manager_version = MANAGER_VERSION
    write_config(root, cfg)

    if app_payload is not None:
        version = manifest.version if manifest is not None else version_fallback
        art = manifest.app_artifact() if manifest is not None else None
        state = install_app_payload(
            root,
            app_payload,
            version,
            sha256=art.sha256 if art else "",
            size=art.size if art else None,
            channel=cfg.channel,
            source="setup",
        )
        result.version = state.version
        result.app_exe = state.exe_path(root)
        result.messages.append(f"app {state.version} -> {result.app_exe}")
        prune_versions(root)
    else:
        current = read_current(root)
        if current is not None and current.exe_path(root).is_file():
            result.version = current.version
            result.app_exe = current.exe_path(root)
            result.messages.append(f"kept installed app {current.version}")
        else:
            result.needs_download = True
            result.messages.append("no bundled app payload; the manager will download the app on first launch")

    if opts.start_menu_shortcut or opts.desktop_shortcut:
        from roboscout_manager import shortcuts

        fn = shortcut_fn or shortcuts.create_shortcuts
        try:
            result.shortcuts = fn(root, desktop=opts.desktop_shortcut, start_menu=opts.start_menu_shortcut)
        except Exception as exc:  # noqa: BLE001
            result.messages.append(f"shortcuts skipped: {exc}")
    if opts.register:
        from roboscout_manager import registry

        fn_reg = register_fn or registry.register_uninstall
        try:
            result.registered = bool(fn_reg(root, result.version or version_fallback))
        except Exception as exc:  # noqa: BLE001
            result.messages.append(f"registry skipped: {exc}")
    append_log(root, f"setup complete: app={result.version or 'pending'} manager={MANAGER_VERSION}")
    return result


def bootstrap_install(root: Path, *, frozen: bool | None = None) -> list[str]:
    """First launch of a bare manager exe (no manager.json yet): make it a real install.

    Covers two paths: a user double-clicked ``RoboScoutAI.exe`` from Downloads, or a
    legacy 0.5.x in-app updater dropped the manager at %LOCALAPPDATA%\\RoboScoutAI\\.
    Copies the manager into the root, writes manager.json, and (frozen only) creates
    the Start Menu shortcut and uninstall entry. Idempotent; never fatal.
    """
    root = Path(root)
    messages: list[str] = []
    if manager_json_path(root).is_file():
        return messages
    is_frozen = bool(getattr(sys, "frozen", False)) if frozen is None else frozen
    root.mkdir(parents=True, exist_ok=True)
    cfg = read_config(root)
    cfg.manager_version = MANAGER_VERSION
    write_config(root, cfg)
    messages.append(f"wrote manager.json (channel {cfg.channel})")
    if is_frozen and running_exe().suffix.lower() == ".exe":
        try:
            placed = install_manager_binary(root)
            messages.append(f"manager -> {placed}")
        except (InstallError, OSError) as exc:
            messages.append(f"manager copy skipped: {exc}")
        from roboscout_manager import registry, shortcuts

        try:
            shortcuts.create_shortcuts(root, desktop=False, start_menu=True)
            messages.append("start menu shortcut created")
        except Exception as exc:  # noqa: BLE001
            messages.append(f"shortcut skipped: {exc}")
        try:
            current = read_current(root)
            if registry.register_uninstall(root, current.version if current else MANAGER_VERSION):
                messages.append("uninstall entry registered")
        except Exception as exc:  # noqa: BLE001
            messages.append(f"registry skipped: {exc}")
    append_log(root, "bootstrap: " + "; ".join(messages))
    return messages


# --- uninstall -----------------------------------------------------------------


def uninstall_bat_lines(*, pid: int, root: str, purge: bool, log_path: str) -> list[str]:
    """cmd script that waits for the manager PID, then removes install files.

    User data (``data\\``) is kept unless ``purge``. The final line deletes the
    script itself via ``(goto) 2>nul & del`` so cmd does not complain.
    """
    lines = [
        "@echo off",
        "setlocal EnableExtensions",
        f"set PID={pid}",
        f'set "ROOT={root}"',
        f'set "LOG={log_path}"',
        'echo [%DATE% %TIME%] uninstall start pid=%PID% > "%LOG%"',
        'powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Wait-Process -Id ([int]$env:PID) -Timeout 120 -ErrorAction Stop } catch { }" >> "%LOG%" 2>&1',
        "ping -n 2 127.0.0.1 >nul",
        f'if exist "%ROOT%\\{MANAGER_EXE_NAME}" del /F /Q "%ROOT%\\{MANAGER_EXE_NAME}" >> "%LOG%" 2>&1',
        f'if exist "%ROOT%\\{MANAGER_EXE_NAME}.new" del /F /Q "%ROOT%\\{MANAGER_EXE_NAME}.new" >> "%LOG%" 2>&1',
        'if exist "%ROOT%\\app" rmdir /S /Q "%ROOT%\\app" >> "%LOG%" 2>&1',
        'if exist "%ROOT%\\update-cache" rmdir /S /Q "%ROOT%\\update-cache" >> "%LOG%" 2>&1',
    ]
    for name in ("current.json", "manager.json", "update-status.json", "ipc.json", "update-git-sha.txt", "update-channel.txt"):
        lines.append(f'if exist "%ROOT%\\{name}" del /F /Q "%ROOT%\\{name}" >> "%LOG%" 2>&1')
    if purge:
        lines.append('if exist "%ROOT%\\data" rmdir /S /Q "%ROOT%\\data" >> "%LOG%" 2>&1')
        lines.append('echo purged user data >> "%LOG%"')
        lines.append('rmdir "%ROOT%" >nul 2>nul')
    lines += [
        'echo OK >> "%LOG%"',
        "endlocal",
        '(goto) 2>nul & del "%~f0"',
        "",
    ]
    return lines


def uninstall(
    root: Path,
    *,
    purge: bool = False,
    unregister_fn: Callable[..., bool] | None = None,
    remove_shortcuts_fn: Callable[..., list[Path]] | None = None,
    spawn: bool = True,
) -> Path:
    """Remove shortcuts + registry now; schedule file removal after we exit."""
    from roboscout_manager import registry, shortcuts

    root = Path(root)
    (unregister_fn or registry.unregister_uninstall)()
    try:
        (remove_shortcuts_fn or shortcuts.remove_shortcuts)()
    except Exception as exc:  # noqa: BLE001
        log.warning("shortcut removal failed: %s", exc)
    append_log(root, f"uninstall requested (purge={purge})")
    log_path = Path(tempfile.gettempdir()) / "RoboScoutAI-uninstall.log"
    script = Path(tempfile.gettempdir()) / "RoboScoutAI-uninstall.bat"
    lines = uninstall_bat_lines(pid=os.getpid(), root=str(root), purge=purge, log_path=str(log_path))
    script.write_text("\r\n".join(lines), encoding="utf-8")
    if spawn and sys.platform.startswith("win"):
        creation = 0x00000200 | 0x00000008 | 0x08000000  # new group | detached | no window
        subprocess.Popen(
            ["cmd.exe", "/c", str(script)],
            close_fds=True,
            creationflags=creation,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    elif spawn:
        # Non-Windows dev: remove synchronously (nothing is locked).
        for path in (manager_exe_path(root), current_json_path(root), manager_json_path(root), status_json_path(root), ipc_json_path(root)):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        shutil.rmtree(app_versions_dir(root), ignore_errors=True)
        if purge:
            shutil.rmtree(root / "data", ignore_errors=True)
    return script


def newest_installed_version(root: Path) -> str | None:
    versions = installed_versions(root)
    return versions[0] if versions else None


def is_downgrade(candidate: str, current: str | None) -> bool:
    return bool(current) and compare_versions(candidate, current or "0") < 0


__all__ = [
    "APP_NAME",
    "InstallError",
    "SetupOptions",
    "SetupResult",
    "bootstrap_install",
    "find_payload",
    "install_app_payload",
    "install_manager_binary",
    "installed_versions",
    "prune_versions",
    "run_setup",
    "swap_staged_manager",
    "uninstall",
    "uninstall_bat_lines",
    "update_log_path",
    "verify_current",
]
