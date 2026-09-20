"""Switch current.json back to the previously active side-by-side version."""

from __future__ import annotations

from pathlib import Path

from roboscout_manager.install import installed_versions
from roboscout_manager.manifest import VerificationError, verify_file
from roboscout_manager.state import (
    CurrentState,
    app_exe_path,
    append_log,
    read_current,
    relative_app_exe,
    write_current,
)


class RollbackError(RuntimeError):
    pass


def rollback_candidates(root: Path) -> list[str]:
    """Versions we could roll back to: recorded previous first, then any other installed."""
    current = read_current(root)
    ordered: list[str] = []
    if current is not None and current.previous_version and app_exe_path(root, current.previous_version).is_file():
        ordered.append(current.previous_version)
    for ver in installed_versions(root):
        if current is not None and ver == current.version:
            continue
        if ver not in ordered:
            ordered.append(ver)
    return ordered


def rollback(root: Path, target: str | None = None) -> CurrentState:
    """Make ``target`` (default: the recorded previous version) the active app."""
    root = Path(root)
    current = read_current(root)
    candidates = rollback_candidates(root)
    if target:
        if target not in candidates and not app_exe_path(root, target).is_file():
            raise RollbackError(f"version {target} is not installed")
        chosen = target
    elif candidates:
        chosen = candidates[0]
    else:
        raise RollbackError("no previous version is installed to roll back to")
    exe = app_exe_path(root, chosen)
    expected_sha = ""
    if current is not None and current.previous_version == chosen:
        expected_sha = current.previous_sha256
    try:
        digest = verify_file(exe, sha256=expected_sha)
    except VerificationError as exc:
        raise RollbackError(f"cannot roll back to {chosen}: {exc}") from exc
    state = CurrentState(
        version=chosen,
        exe=relative_app_exe(chosen),
        sha256=digest,
        size=exe.stat().st_size,
        previous_version=current.version if current is not None and current.version != chosen else "",
        previous_sha256=current.sha256 if current is not None and current.version != chosen else "",
        channel=current.channel if current is not None else "",
        source="rollback",
    )
    write_current(root, state)
    append_log(root, f"rolled back {current.version if current else '?'} -> {chosen}")
    return state
