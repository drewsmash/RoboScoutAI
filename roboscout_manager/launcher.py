"""Thin launcher: verify the active app build, start it as a child, relaunch on request."""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Sequence

from roboscout_manager import APP_NAME, MANAGER_VERSION, RELAUNCH_EXIT_CODE
from roboscout_manager.install import swap_staged_manager, verify_current
from roboscout_manager.ipc import clear_status, issue_token, read_status, write_status
from roboscout_manager.rollback import RollbackError, rollback
from roboscout_manager.state import (
    CurrentState,
    ManagerConfig,
    append_log,
    manager_exe_path,
    read_config,
    read_current,
    running_exe,
)
from roboscout_manager.ui import Splash, message_box
from roboscout_manager.updater import check_for_update, perform_update, repair

log = logging.getLogger("roboscout.manager")


class LaunchError(RuntimeError):
    pass


def free_port(preferred: int = 8000) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def build_app_command(
    exe: Path,
    *,
    port: int,
    token: str,
    root: Path,
    manager_exe: Path | None = None,
    extra: Sequence[str] = (),
) -> list[str]:
    cmd = [
        str(exe),
        "--managed",
        "--port",
        str(int(port)),
        "--manager-token",
        token,
        "--manager-root",
        str(root),
        "--manager-exe",
        str(manager_exe or manager_exe_path(root)),
        "--skip-update-check",
    ]
    cmd.extend(str(a) for a in extra)
    return cmd


def wait_for_health(port: int, *, timeout: float = 60.0, child: subprocess.Popen[Any] | None = None, pump=None) -> bool:
    url = f"http://127.0.0.1:{int(port)}/api/health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if child is not None and child.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:  # noqa: S310
                if 200 <= int(getattr(resp, "status", 200) or 200) < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        if pump is not None:
            pump()
        time.sleep(0.2)
    return False


def ensure_launchable(root: Path, *, cfg: ManagerConfig | None = None, progress=None) -> CurrentState:
    """Return a verified active build, rolling back or repairing when damaged."""
    root = Path(root)
    state, problem = verify_current(root)
    if state is not None and not problem:
        return state
    append_log(root, f"active build not launchable: {problem}")
    if state is not None:
        try:
            rolled = rollback(root)
            append_log(root, f"rolled back to {rolled.version} after damage")
            return rolled
        except RollbackError as exc:
            append_log(root, f"rollback impossible: {exc}")
    result = repair(root, cfg=cfg, progress=progress or (lambda *_a: None))
    if not result.ok:
        raise LaunchError(result.message or result.error or "repair failed")
    state, problem = verify_current(root)
    if state is None or problem:
        raise LaunchError(problem or "repair did not produce a launchable build")
    return state


def _child_flags() -> int:
    if sys.platform.startswith("win"):
        return 0x08000000  # CREATE_NO_WINDOW: the app is a console build; hide it.
    return 0


def _background_check(root: Path, cfg: ManagerConfig) -> None:
    try:
        check_for_update(root, cfg=cfg, write_status_file=True)
    except Exception as exc:  # noqa: BLE001
        log.debug("background update check failed: %s", exc)


def run_launcher(
    root: Path,
    *,
    port: int | None = None,
    extra_args: Sequence[str] = (),
    splash: bool = True,
    background_check: bool = True,
    popen=subprocess.Popen,
) -> int:
    """Launch the active app and keep relaunching it while it asks to (exit 75)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    swap_staged_manager(root)
    cfg = read_config(root)
    ui = Splash(enabled=splash)

    def progress(state: str, pct: int, message: str) -> None:
        ui.set_message(message, progress=pct)

    try:
        if read_current(root) is None:
            ui.set_message(f"Downloading {APP_NAME} from channel {cfg.channel}…", progress=5)
            result = perform_update(root, cfg=cfg, progress=progress, force=True)
            if result.manager_update_scheduled:
                ui.close()
                return 0
            if not result.ok:
                raise LaunchError(result.message)
        state = ensure_launchable(root, cfg=cfg, progress=progress)
    except LaunchError as exc:
        ui.close()
        append_log(root, f"launch failed: {exc}")
        message_box(
            f"{APP_NAME} cannot start",
            f"{exc}\n\nRun RoboScoutAI.exe --repair after checking your connection, or reinstall with RoboScoutAI-Setup.exe.",
            kind="error",
        )
        return 2

    manager_exe = running_exe() if running_exe().suffix.lower() == ".exe" else manager_exe_path(root)
    clear_status(root)
    code = 0
    launches = 0
    preferred_port = port or 8000
    while True:
        launches += 1
        state = read_current(root) or state
        exe = state.exe_path(root)
        # Keep the same port across relaunches so an open UI window can simply reload.
        chosen_port = free_port(preferred_port)
        preferred_port = chosen_port
        token = issue_token(root, port=chosen_port)
        cmd = build_app_command(exe, port=chosen_port, token=token, root=root, manager_exe=manager_exe, extra=extra_args)
        log_file = root / "app.log"
        append_log(root, f"launching app {state.version} on port {chosen_port}")
        try:
            handle = open(log_file, "ab")
        except OSError:
            handle = None
        env = os.environ.copy()
        env["ROBOSCOUT_MANAGED"] = "1"
        env["ROBOSCOUT_MANAGER_ROOT"] = str(root)
        env["ROBOSCOUT_MANAGER_EXE"] = str(manager_exe)
        env["ROBOSCOUT_MANAGER_TOKEN"] = token
        env["ROBOSCOUT_MANAGER_VERSION"] = MANAGER_VERSION
        env["ROBOSCOUT_UPDATE_CHANNEL"] = cfg.channel
        try:
            child = popen(
                cmd,
                cwd=str(exe.parent),
                stdin=subprocess.DEVNULL,
                stdout=handle or subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                creationflags=_child_flags(),
                env=env,
                close_fds=True,
            )
        except OSError as exc:
            ui.close()
            if handle:
                handle.close()
            append_log(root, f"could not start {exe}: {exc}")
            message_box(f"{APP_NAME} cannot start", f"{exe}\n{exc}", kind="error")
            return 2
        ui.set_message(f"Starting {APP_NAME} {state.version}…", progress=None)
        healthy = wait_for_health(chosen_port, timeout=90.0, child=child, pump=ui.pump)
        ui.close()
        if background_check and launches == 1 and cfg.auto_check and healthy:
            threading.Thread(target=_background_check, args=(root, cfg), daemon=True).start()
        try:
            code = child.wait()
        except KeyboardInterrupt:
            child.terminate()
            code = 130
        finally:
            if handle:
                handle.close()
        append_log(root, f"app {state.version} exited with {code}")
        if code == RELAUNCH_EXIT_CODE:
            status = read_status(root)
            if status.state == "manager-update":
                # A manager swap is pending; the .bat relaunches us after we exit.
                return 0
            ui = Splash(enabled=splash)
            try:
                state = ensure_launchable(root, cfg=cfg, progress=progress)
            except LaunchError as exc:
                ui.close()
                message_box(f"{APP_NAME} cannot restart", str(exc), kind="error")
                return 2
            continue
        break
    return int(code or 0)
