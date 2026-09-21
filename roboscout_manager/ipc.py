"""Manager <-> app IPC: a per-launch token plus a JSON status file.

The managed app never touches binaries itself. It invokes the manager exe with
``--update --token <token>`` (token issued by the manager at launch), polls the
status file for progress, and finally exits with ``RELAUNCH_EXIT_CODE`` so the
manager relaunches the freshly installed version.
"""

from __future__ import annotations

import hmac
import os
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from roboscout_manager import RELAUNCH_EXIT_CODE
from roboscout_manager.state import ipc_json_path, read_json, status_json_path, utc_now, write_json_atomic

STATES = ("idle", "checking", "downloading", "verifying", "installing", "done", "error", "manager-update")


class IpcError(RuntimeError):
    pass


def new_token() -> str:
    return secrets.token_urlsafe(32)


def issue_token(root: Path, *, manager_pid: int | None = None, port: int | None = None) -> str:
    token = new_token()
    payload: dict[str, Any] = {
        "token": token,
        "manager_pid": int(manager_pid if manager_pid is not None else os.getpid()),
        "port": int(port or 0),
        "issued_at": utc_now(),
    }
    path = write_json_atomic(ipc_json_path(root), payload)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def read_ipc(root: Path) -> dict[str, Any] | None:
    return read_json(ipc_json_path(root))


def read_token(root: Path) -> str:
    data = read_ipc(root) or {}
    return str(data.get("token") or "")


def validate_token(root: Path, token: str | None) -> bool:
    expected = read_token(root)
    if not expected or not token:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), str(token).encode("utf-8"))


def require_token(root: Path, token: str | None) -> None:
    if not validate_token(root, token):
        raise IpcError("manager token missing or invalid")


@dataclass
class UpdateStatus:
    state: str = "idle"
    progress: int = 0
    message: str = ""
    version: str = ""
    current_version: str = ""
    error: str = ""
    updated_at: str = ""
    pid: int = 0
    relaunch_exit_code: int = RELAUNCH_EXIT_CODE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def finished(self) -> bool:
        return self.state in {"done", "error"}


def write_status(
    root: Path,
    state: str,
    *,
    progress: int | float = 0,
    message: str = "",
    version: str = "",
    current_version: str = "",
    error: str = "",
    pid: int | None = None,
) -> UpdateStatus:
    if state not in STATES:
        raise ValueError(f"unknown status state {state!r}")
    status = UpdateStatus(
        state=state,
        progress=max(0, min(100, int(progress))),
        message=message,
        version=version,
        current_version=current_version,
        error=error,
        updated_at=utc_now(),
        pid=int(pid if pid is not None else os.getpid()),
    )
    write_json_atomic(status_json_path(root), status.to_dict())
    return status


def read_status(root: Path) -> UpdateStatus:
    data = read_json(status_json_path(root))
    if not data:
        return UpdateStatus()
    return UpdateStatus(
        state=str(data.get("state") or "idle"),
        progress=int(data.get("progress") or 0),
        message=str(data.get("message") or ""),
        version=str(data.get("version") or ""),
        current_version=str(data.get("current_version") or ""),
        error=str(data.get("error") or ""),
        updated_at=str(data.get("updated_at") or ""),
        pid=int(data.get("pid") or 0),
        relaunch_exit_code=int(data.get("relaunch_exit_code") or RELAUNCH_EXIT_CODE),
    )


def clear_status(root: Path) -> None:
    try:
        status_json_path(root).unlink(missing_ok=True)
    except OSError:
        pass
