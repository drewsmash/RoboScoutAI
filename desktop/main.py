#!/usr/bin/env python3
"""Desktop entrypoint: start the local RamScoutAI server and open a browser."""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path

# Ensure project root imports work when running from source.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ramscout import __version__
from ramscout.paths import is_frozen, jobs_dir, web_dir
from ramscout.updater import apply_downloaded_update, check_for_update, download_update

log = logging.getLogger("ramscout.desktop")


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
        log.info("RamScoutAI %s is up to date.", info.current_version)
        return
    log.info(
        "Update available: %s → %s (%s)",
        info.current_version,
        info.latest_version,
        info.asset_name or "open release page",
    )
    if not auto_apply or not is_frozen() or not info.asset_url:
        log.info("Open %s or use Check for updates in the app.", info.release_url)
        return
    try:
        package = download_update(info)
        message = apply_downloaded_update(package)
        log.info("%s", message)
        time.sleep(1.0)
        os._exit(0)
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not auto-apply update: %s", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RamScoutAI desktop launcher")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-update", action="store_true", help="Download and apply a newer GitHub release on launch")
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

    log.info("RamScoutAI %s starting on http://%s:%s", __version__, args.host, port)
    config = uvicorn.Config(app, host=args.host, port=port, log_level="info", reload=False)
    server = uvicorn.Server(config)

    if not args.no_browser:
        def _open() -> None:
            time.sleep(0.8)
            webbrowser.open(f"http://{args.host}:{port}/")

        threading.Thread(target=_open, daemon=True).start()

    try:
        server.run()
    except KeyboardInterrupt:
        log.info("Shutting down.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
