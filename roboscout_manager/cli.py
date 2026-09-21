"""Command line for RoboScoutAI.exe (manager) and RoboScoutAI-Setup.exe.

No arguments → launch the active app. Everything else is a management command
that prints (or writes with ``--out``) JSON when ``--json`` is given, so the
managed FastAPI app can shell out to the manager and parse the result.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from roboscout_manager import APP_NAME, MANAGER_VERSION, RELAUNCH_EXIT_CODE
from roboscout_manager.install import (
    SetupOptions,
    find_payload,
    install_manager_binary,
    installed_versions,
    run_setup,
    uninstall,
    verify_current,
)
from roboscout_manager.ipc import IpcError, read_status, require_token, write_status
from roboscout_manager.rollback import RollbackError, rollback, rollback_candidates
from roboscout_manager.state import (
    append_log,
    bundle_root,
    default_install_root,
    manager_exe_path,
    read_config,
    read_current,
    running_exe,
    set_channel,
    update_log_path,
)
from roboscout_manager.updater import check_for_update, perform_update, repair

log = logging.getLogger("roboscout.manager")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="RoboScoutAI",
        description=f"{APP_NAME} desktop manager: install, launch, update, roll back.",
    )
    parser.add_argument("--version", action="store_true", help="print manager and installed app versions")
    parser.add_argument("--root", default="", help="install root (default %%LOCALAPPDATA%%\\RoboScoutAI)")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--out", default="", help="write the JSON result to this file (windowed exe has no stdout)")
    parser.add_argument("--silent", action="store_true", help="no dialogs")
    parser.add_argument("--token", default="", help="IPC token issued by the running manager")

    cmd = parser.add_argument_group("commands")
    cmd.add_argument("--check", action="store_true", help="check the update channel")
    cmd.add_argument("--update", action="store_true", help="download + install the newest build side-by-side")
    cmd.add_argument("--rollback", nargs="?", const="", default=None, metavar="VERSION", help="switch back to the previous (or given) version")
    cmd.add_argument("--repair", action="store_true", help="re-download the channel build and recreate shortcuts")
    cmd.add_argument("--uninstall", action="store_true", help="remove RoboScoutAI for this user")
    cmd.add_argument("--purge", action="store_true", help="with --uninstall: also delete user data")
    cmd.add_argument("--channel", default=None, metavar="BRANCH_OR_TAG", help="switch the update channel")
    cmd.add_argument("--status", action="store_true", help="print the current update status file")
    cmd.add_argument("--list", action="store_true", help="list installed versions")
    cmd.add_argument("--install", action="store_true", help="run the installer (Setup mode)")

    opt = parser.add_argument_group("options")
    opt.add_argument("--payload", default="", help="Setup: folder holding the app/manager exe + manifest.json")
    opt.add_argument("--desktop-shortcut", dest="desktop_shortcut", action="store_true", default=None)
    opt.add_argument("--no-desktop-shortcut", dest="desktop_shortcut", action="store_false")
    opt.add_argument("--no-launch", action="store_true", help="Setup/update: do not start the app afterwards")
    opt.add_argument("--launch-after", action="store_true", help="update: launch the app when done")
    opt.add_argument("--force", action="store_true", help="update: reinstall even when up to date")
    opt.add_argument("--port", type=int, default=None, help="preferred local port for the app")
    opt.add_argument("--no-splash", action="store_true")
    parser.add_argument("app_args", nargs=argparse.REMAINDER, help="arguments after -- are passed to the app")
    return parser


def emit(payload: dict[str, Any], *, as_json: bool, out: str = "", human: str = "") -> None:
    text = json.dumps(payload, indent=2)
    if out:
        try:
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(text, encoding="utf-8")
        except OSError as exc:
            log.warning("could not write %s: %s", out, exc)
    stream = sys.stdout
    if stream is None:
        return
    try:
        if as_json:
            print(text, file=stream)
        elif human:
            print(human, file=stream)
    except (OSError, ValueError):
        pass


def _configure_logging(root: Path) -> None:
    handlers: list[logging.Handler] = []
    try:
        root.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(root / "manager.log", encoding="utf-8"))
    except OSError:
        pass
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s", handlers=handlers)


def _spawn_detached(cmd: list[str], cwd: Path | None = None) -> None:
    flags = 0
    kwargs: dict[str, Any] = {}
    if sys.platform.startswith("win"):
        flags = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        close_fds=True,
        creationflags=flags,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        **kwargs,
    )


def _versions_payload(root: Path) -> dict[str, Any]:
    current = read_current(root)
    cfg = read_config(root)
    return {
        "manager_version": MANAGER_VERSION,
        "app_version": current.version if current else "",
        "installed_versions": installed_versions(root),
        "rollback_candidates": rollback_candidates(root),
        "channel": cfg.channel,
        "remote": cfg.remote,
        "github_repo": cfg.github_repo,
        "root": str(root),
        "manager_exe": str(manager_exe_path(root)),
        "running_exe": str(running_exe()),
        "update_log": str(update_log_path(root)),
    }


def cmd_install(args: argparse.Namespace, root: Path) -> int:
    from roboscout_manager.ui import message_box, setup_dialog

    payload_dir = Path(args.payload) if args.payload else (bundle_root() / "payload")
    if not payload_dir.is_dir() and not args.payload:
        # Setup running from source: fall back to dist/release if present.
        alt = Path.cwd() / "dist" / "release"
        payload_dir = alt if alt.is_dir() else payload_dir
    _app, _mgr, manifest = find_payload(payload_dir)
    version = manifest.version if manifest else MANAGER_VERSION
    existing = ""
    try:
        cur = read_current(root)
        if cur is not None and cur.exe_path(root).is_file():
            existing = cur.version or ""
    except Exception:  # noqa: BLE001
        existing = ""
    choice = setup_dialog(
        str(root),
        version=version,
        existing_version=existing,
        default_desktop=True if args.desktop_shortcut is None else args.desktop_shortcut,
        silent=args.silent,
    )
    if not choice.proceed:
        emit({"ok": False, "cancelled": True}, as_json=args.json, out=args.out, human="Setup cancelled.")
        return 1
    opts = SetupOptions(
        root=root,
        payload_dir=payload_dir if payload_dir.is_dir() else None,
        desktop_shortcut=bool(choice.desktop_shortcut),
        start_menu_shortcut=True,
        register=True,
        launch=not args.no_launch,
        channel=args.channel or "",
        silent=args.silent,
    )
    try:
        result = run_setup(opts)
    except Exception as exc:  # noqa: BLE001
        message_box(f"{APP_NAME} Setup failed", str(exc), kind="error", silent=args.silent)
        emit({"ok": False, "error": str(exc)}, as_json=args.json, out=args.out, human=f"Setup failed: {exc}")
        return 1
    payload = {
        "ok": True,
        "root": str(root),
        "version": result.version,
        "previous_version": existing,
        "mode": getattr(choice, "mode", "install"),
        "manager_exe": str(result.manager_exe) if result.manager_exe else "",
        "app_exe": str(result.app_exe) if result.app_exe else "",
        "shortcuts": [str(p) for p in result.shortcuts],
        "registered": result.registered,
        "needs_download": result.needs_download,
        "messages": result.messages,
    }
    if existing and result.version and existing != result.version:
        human = f"{APP_NAME} updated {existing} → {result.version} at {root}."
    elif existing and result.version == existing:
        human = f"{APP_NAME} {result.version} repaired at {root}."
    else:
        human = f"{APP_NAME} {result.version or '(download pending)'} installed to {root}."
    emit(payload, as_json=args.json, out=args.out, human=human)
    staged = result.manager_exe if result.manager_exe and result.manager_exe.name.endswith(".new") else None
    if staged is not None:
        # We are running *as* %LOCALAPPDATA%\RoboScoutAI\RoboScoutAI.exe (legacy updater dropped
        # Setup there). Swap in the real manager after we exit, then launch it.
        from roboscout_manager.updater import schedule_manager_swap

        relaunch = ["--update", "--launch-after"] if result.needs_download else []
        schedule_manager_swap(root, staged, relaunch_args=relaunch, spawn=opts.launch)
        return 0
    if opts.launch:
        exe = manager_exe_path(root)
        launch_cmd = [str(exe)]
        if result.needs_download:
            launch_cmd += ["--update", "--launch-after"]
        if exe.is_file():
            try:
                _spawn_detached(launch_cmd, cwd=root)
            except OSError as exc:
                message_box(APP_NAME, f"Installed, but could not start {exe}: {exc}", kind="warning", silent=args.silent)
    return 0


def cmd_check(args: argparse.Namespace, root: Path) -> int:
    check = check_for_update(root, write_status_file=True)
    human = check.message if not check.error else f"{check.message}: {check.error}"
    emit(check.as_dict(), as_json=args.json, out=args.out, human=human)
    return 0 if not check.error or check.manifest is not None else 1


def cmd_update(args: argparse.Namespace, root: Path) -> int:
    if args.token:
        try:
            require_token(root, args.token)
        except IpcError as exc:
            emit({"ok": False, "error": str(exc)}, as_json=args.json, out=args.out, human=str(exc))
            return 3
    status = read_status(root)
    if status.state in {"downloading", "verifying", "installing"} and status.pid and status.pid != os.getpid() and _pid_alive(status.pid):
        emit({"ok": False, "error": "an update is already running", "status": status.to_dict()}, as_json=args.json, out=args.out, human="Update already running.")
        return 4
    write_status(root, "checking", progress=2, message="Checking for updates…")
    check = check_for_update(root)
    relaunch = ["--update"] + (["--launch-after"] if args.launch_after else [])
    result = perform_update(root, check, force=args.force, relaunch_after_manager_swap=relaunch)
    payload = result.as_dict()
    payload["check"] = check.as_dict()
    payload["relaunch_exit_code"] = RELAUNCH_EXIT_CODE
    emit(payload, as_json=args.json, out=args.out, human=result.message)
    if result.manager_update_scheduled:
        return 0
    if result.ok and args.launch_after and not args.no_launch:
        try:
            _spawn_detached([str(manager_exe_path(root))], cwd=root)
        except OSError as exc:
            log.warning("could not relaunch after update: %s", exc)
    return 0 if result.ok else 1


def cmd_rollback(args: argparse.Namespace, root: Path) -> int:
    try:
        state = rollback(root, args.rollback or None)
    except RollbackError as exc:
        emit({"ok": False, "error": str(exc)}, as_json=args.json, out=args.out, human=f"Rollback failed: {exc}")
        return 1
    emit({"ok": True, "version": state.version, "current": state.to_dict()}, as_json=args.json, out=args.out, human=f"Rolled back to {state.version}. Restart {APP_NAME}.")
    return 0


def cmd_repair(args: argparse.Namespace, root: Path) -> int:
    from roboscout_manager import registry, shortcuts

    messages: list[str] = []
    state, problem = verify_current(root)
    if state is not None and not problem and not args.force:
        messages.append(f"app {state.version} verified OK")
    else:
        result = repair(root)
        messages.append(result.message)
        if not result.ok:
            emit({"ok": False, "error": result.error, "messages": messages}, as_json=args.json, out=args.out, human="\n".join(messages))
            return 1
    try:
        install_manager_binary(root)
        messages.append("manager binary refreshed")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"manager copy skipped: {exc}")
    try:
        shortcuts.create_shortcuts(root, desktop=bool(args.desktop_shortcut), start_menu=True)
        messages.append("shortcuts recreated")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"shortcuts skipped: {exc}")
    current = read_current(root)
    try:
        if registry.register_uninstall(root, current.version if current else MANAGER_VERSION):
            messages.append("uninstall entry refreshed")
    except Exception as exc:  # noqa: BLE001
        messages.append(f"registry skipped: {exc}")
    append_log(root, "repair: " + "; ".join(messages))
    emit({"ok": True, "messages": messages}, as_json=args.json, out=args.out, human="\n".join(messages))
    return 0


def cmd_uninstall(args: argparse.Namespace, root: Path) -> int:
    from roboscout_manager.ui import ask_yes_no

    if not ask_yes_no(
        f"Uninstall {APP_NAME}",
        f"Remove {APP_NAME} from {root}?" + ("\n\nUser data will also be deleted." if args.purge else "\n\nYour data folder is kept."),
        silent=args.silent,
    ):
        emit({"ok": False, "cancelled": True}, as_json=args.json, out=args.out, human="Uninstall cancelled.")
        return 1
    script = uninstall(root, purge=args.purge)
    emit({"ok": True, "script": str(script), "purge": args.purge}, as_json=args.json, out=args.out, human=f"{APP_NAME} is being removed from {root}.")
    return 0


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw)
    root = Path(args.root).expanduser() if args.root else default_install_root()
    _configure_logging(root)

    if args.version:
        payload = _versions_payload(root)
        emit(payload, as_json=args.json, out=args.out, human=f"{APP_NAME} manager {MANAGER_VERSION}; app {payload['app_version'] or 'not installed'}; channel {payload['channel']}")
        return 0
    if args.install:
        return cmd_install(args, root)
    if args.channel is not None and not (args.check or args.update or args.repair):
        try:
            cfg = set_channel(root, args.channel)
        except ValueError as exc:
            emit({"ok": False, "error": str(exc)}, as_json=args.json, out=args.out, human=str(exc))
            return 2
        append_log(root, f"channel set to {cfg.channel}")
        emit({"ok": True, "channel": cfg.channel}, as_json=args.json, out=args.out, human=f"Update channel is now {cfg.channel}.")
        return 0
    if args.channel is not None:
        set_channel(root, args.channel)
    if args.uninstall:
        return cmd_uninstall(args, root)
    if args.status:
        emit(read_status(root).to_dict(), as_json=True, out=args.out)
        return 0
    if args.list:
        payload = _versions_payload(root)
        emit(payload, as_json=args.json, out=args.out, human="\n".join(payload["installed_versions"]) or "(no versions installed)")
        return 0
    if args.check:
        return cmd_check(args, root)
    if args.update:
        return cmd_update(args, root)
    if args.rollback is not None:
        return cmd_rollback(args, root)
    if args.repair:
        return cmd_repair(args, root)

    from roboscout_manager.launcher import run_launcher

    extra = [a for a in args.app_args if a != "--"]
    return run_launcher(root, port=args.port, extra_args=extra, splash=not args.no_splash)


if __name__ == "__main__":
    sys.exit(main())
