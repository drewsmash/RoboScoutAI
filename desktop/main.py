#!/usr/bin/env python3
"""Desktop entrypoint: start the local RoboScoutAI server and open an app window."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
from pathlib import Path

# Ensure project root imports work when running from source.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from desktop.app_window import run_app_ui
from ramscout import __version__
from ramscout.paths import jobs_dir, web_dir
from ramscout.updater import apply_update_now, check_for_update

log = logging.getLogger("roboscout.desktop")


def _free_port(preferred: int) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _maybe_check_updates(auto_apply: bool) -> None:
    try:
        info = check_for_update()
    except Exception as exc:  # noqa: BLE001
        log.warning("Update check failed: %s", exc)
        return
    if info.error and not info.available:
        log.info("Update check: %s", info.error)
        return
    if not info.available:
        log.info("RamScoutAI %s is up to date (%s).", info.current_version, info.message or "git")
        return
    log.info(
        "Update available from git: %s → %s (%s @ %s)",
        info.current_version,
        info.latest_version or info.remote_sha[:7],
        info.branch,
        info.remote_sha[:7] or info.asset_name or "remote",
    )
    if not auto_apply or not info.can_apply:
        log.info("Use Check for updates / Update now in the app (remote %s).", info.remote)
        return
    try:
        result = apply_update_now()
        log.info("%s", result.get("message") or "Update applied.")
        if result.get("restarting"):
            time.sleep(1.0)
            os._exit(0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not auto-apply git update: %s", exc)


def _stop_server(server: object) -> None:
    should_exit = getattr(server, "should_exit", None)
    if should_exit is not None:
        try:
            server.should_exit = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass
    force = getattr(server, "force_exit", None)
    if force is not None:
        try:
            server.force_exit = True  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RoboScoutAI desktop launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument(
        "--browser",
        action="store_true",
        help="Open in a normal browser tab (with URL bar) instead of an app window",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open a UI window; only start the local server",
    )
    parser.add_argument(
        "--auto-update",
        action="store_true",
        help="Fetch and apply a newer build from the git remote on launch",
    )
    parser.add_argument("--skip-update-check", action="store_true")
    args = parser.parse_args(argv)

    _configure_logging()
    jobs_dir()  # ensure writable data path exists
    if not web_dir().is_dir():
        log.error("Web assets missing at %s", web_dir())
        return 1

    if not args.skip_update_check:
        thread = threading.Thread(target=_maybe_check_updates, args=(args.auto_update,), daemon=True)
        thread.start()

    port = _free_port(args.port)
    # Import after path bootstrap so frozen bundles resolve datas correctly.
    import uvicorn
    from app import app

    url = f"http://{args.host}:{port}/"
    log.info("RoboScoutAI %s starting on %s", __version__, url)
    config = uvicorn.Config(app, host=args.host, port=port, log_level="info", reload=False)
    server = uvicorn.Server(config)

    if args.no_browser:
        ui_mode = "none"
    elif args.browser:
        ui_mode = "browser"
    else:
        ui_mode = "app"

    # App / chrome-app windows block until closed; serve in a background thread.
    # Browser / none keep the server in the foreground until Ctrl+C.
    if ui_mode == "app":
        server_thread = threading.Thread(target=server.run, daemon=True, name="roboscout-uvicorn")
        server_thread.start()
        try:
            strategy = run_app_ui(url, mode="app")
            if strategy in {"webview", "chrome_app"}:
                log.info("UI closed via %s; shutting down server.", strategy)
            else:
                # System-browser fallback does not block on window close.
                log.info("App window unavailable; serving until Ctrl+C (%s).", strategy)
                while server_thread.is_alive():
                    server_thread.join(timeout=1.0)
        except KeyboardInterrupt:
            log.info("Shutting down.")
        finally:
            _stop_server(server)
            server_thread.join(timeout=3.0)
        return 0

    if ui_mode == "browser":
        def _open() -> None:
            run_app_ui(url, mode="browser")

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
