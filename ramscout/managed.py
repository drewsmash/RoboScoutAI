"""Bridge from the FastAPI app to the RoboScoutAI manager (``RoboScoutAI.exe``).

When the app is launched by the manager (``--managed``), it never touches
binaries itself. Update checks and installs are delegated to the manager
process; progress is read from the manager's status file; and a relaunch is
requested by exiting with ``RELAUNCH_EXIT_CODE`` so the manager starts the
freshly installed side-by-side version.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from ramscout import __version__
from ramscout.brand import APP_NAME

log = logging.getLogger(__name__)

_STATE_LOCK = threading.Lock()
_LAST_CHECK: dict[str, Any] | None = None
_RELAUNCH_REQUESTED = False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_managed() -> bool:
    return _truthy(os.environ.get("ROBOSCOUT_MANAGED"))


def manager_root() -> Path | None:
    raw = (os.environ.get("ROBOSCOUT_MANAGER_ROOT") or "").strip()
    return Path(raw) if raw else None


def manager_exe() -> Path | None:
    raw = (os.environ.get("ROBOSCOUT_MANAGER_EXE") or "").strip()
    return Path(raw) if raw else None


def manager_token() -> str:
    return (os.environ.get("ROBOSCOUT_MANAGER_TOKEN") or "").strip()


def manager_version() -> str:
    return (os.environ.get("ROBOSCOUT_MANAGER_VERSION") or "").strip()


def update_channel() -> str:
    return (os.environ.get("ROBOSCOUT_UPDATE_CHANNEL") or "").strip()


def relaunch_exit_code() -> int:
    try:
        from roboscout_manager import RELAUNCH_EXIT_CODE

        return int(RELAUNCH_EXIT_CODE)
    except Exception:  # noqa: BLE001
        return 75


def info() -> dict[str, Any]:
    return {
        "managed": is_managed(),
        "manager_version": manager_version(),
        "manager_exe": str(manager_exe() or ""),
        "root": str(manager_root() or ""),
        "channel": update_channel(),
        "relaunch_requested": _RELAUNCH_REQUESTED,
    }


def _manager_command(args: list[str]) -> list[str]:
    exe = manager_exe()
    root = manager_root()
    base: list[str]
    if exe is not None and exe.suffix.lower() == ".exe" and exe.is_file():
        base = [str(exe)]
    else:
        # Source / dev run: use the in-repo manager package with this interpreter.
        base = [sys.executable, "-m", "roboscout_manager"]
    cmd = base + list(args)
    if root is not None:
        cmd += ["--root", str(root)]
    return cmd


def _no_window_flags() -> int:
    return 0x08000000 if sys.platform.startswith("win") else 0


def run_manager(args: list[str], *, timeout: float = 300.0) -> dict[str, Any]:
    """Run a manager command and return its JSON result (via --out for windowed exes)."""
    out = Path(tempfile.gettempdir()) / f"roboscout-manager-{uuid.uuid4().hex}.json"
    cmd = _manager_command(list(args) + ["--json", "--out", str(out)])
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            creationflags=_no_window_flags(),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"ok": False, "error": f"manager unavailable: {exc}", "available": False, "message": "manager unavailable"}
    payload: dict[str, Any] | None = None
    try:
        if out.is_file():
            payload = json.loads(out.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = None
    finally:
        try:
            out.unlink(missing_ok=True)
        except OSError:
            pass
    if payload is None and proc.stdout.strip():
        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError:
            payload = None
    if payload is None:
        err = (proc.stderr or proc.stdout or "").strip()[-800:]
        return {"ok": False, "error": err or f"manager exited {proc.returncode}", "available": False, "message": "manager error"}
    payload.setdefault("exit_code", proc.returncode)
    return payload


def _store(check: dict[str, Any]) -> dict[str, Any]:
    global _LAST_CHECK
    with _STATE_LOCK:
        _LAST_CHECK = dict(check)
    return check


def last_check() -> dict[str, Any] | None:
    with _STATE_LOCK:
        return dict(_LAST_CHECK) if _LAST_CHECK else None


def check_for_update() -> dict[str, Any]:
    """UpdateInfo-compatible dict produced by ``RoboScoutAI.exe --check``."""
    result = run_manager(["--check"], timeout=240.0)
    result.setdefault("current_version", __version__)
    result.setdefault("mode", "managed")
    result.setdefault("frozen", bool(getattr(sys, "frozen", False)))
    result.setdefault("managed", True)
    result.setdefault("can_apply", bool(result.get("available")))
    result.setdefault("branch", update_channel())
    if result.get("error") and not result.get("message"):
        result["message"] = "update channel unreachable"
    return _store(result)


def status() -> dict[str, Any]:
    root = manager_root()
    payload: dict[str, Any] = {"managed": is_managed(), "state": "idle", "progress": 0, "message": "", "version": "", "error": ""}
    if root is None:
        return payload
    try:
        from roboscout_manager.ipc import read_status

        payload.update(read_status(root).to_dict())
    except Exception as exc:  # noqa: BLE001
        payload["error"] = f"status unavailable: {exc}"
    payload["relaunch_requested"] = _RELAUNCH_REQUESTED
    return payload


def start_update() -> dict[str, Any]:
    """Kick off ``RoboScoutAI.exe --update`` detached; UI polls /api/updates/status."""
    root = manager_root()
    if root is None:
        return {"ok": False, "message": "manager root unknown; start RoboScoutAI from the Start Menu.", "restarting": False}
    try:
        from roboscout_manager.ipc import write_status

        write_status(root, "checking", progress=1, message="Starting update…", current_version=__version__)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not pre-write status: %s", exc)
    args = ["--update", "--silent"]
    token = manager_token()
    if token:
        args += ["--token", token]
    cmd = _manager_command(args)
    kwargs: dict[str, Any] = {}
    flags = _no_window_flags()
    if sys.platform.startswith("win"):
        flags |= 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    try:
        subprocess.Popen(
            cmd,
            close_fds=True,
            creationflags=flags,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **kwargs,
        )
    except OSError as exc:
        return {"ok": False, "message": f"Could not start the {APP_NAME} manager: {exc}", "error": str(exc), "restarting": False}
    return {
        "ok": True,
        "managed": True,
        "restarting": False,
        "message": (
            f"{APP_NAME} is downloading the update side-by-side. "
            "Keep working; you will be asked to restart when it is installed."
        ),
        "status_url": "/api/updates/status",
    }


def request_relaunch(delay: float = 0.8) -> dict[str, Any]:
    """Close the UI window and exit with the relaunch code; the manager restarts us."""
    global _RELAUNCH_REQUESTED
    if not is_managed():
        return {"ok": False, "message": "Relaunch is only available when started by RoboScoutAI.exe."}
    _RELAUNCH_REQUESTED = True
    code = relaunch_exit_code()

    def _go() -> None:
        try:
            from desktop.app_window import close_app_window

            close_app_window()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
        os._exit(code)

    threading.Timer(delay, _go).start()
    return {"ok": True, "message": f"Restarting {APP_NAME}…", "restarting": True, "exit_code": code}
