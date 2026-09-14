"""Open RamScoutAI in a chrome-less desktop window (native app feel)."""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable
from pathlib import Path
from typing import Any

log = logging.getLogger("ramscout.desktop")

APP_TITLE = "RamScoutAI"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 840


def _chrome_user_data_dir() -> Path:
    """Isolated profile so --app stays its own process (not an existing Chrome session)."""
    base = Path(tempfile.gettempdir()) / "ramscout-app-chrome"
    base.mkdir(parents=True, exist_ok=True)
    return base


def wait_for_server(url: str, timeout: float = 30.0) -> bool:
    """Poll until the local HTTP server responds or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:
                if 200 <= int(getattr(resp, "status", 200) or 200) < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(0.15)
    return False


def _chrome_like_binaries() -> list[str]:
    names: list[str] = []
    if sys.platform == "win32":
        names = [
            "msedge",
            "chrome",
            "google-chrome",
            "chromium",
        ]
        # Common install paths when not on PATH
        program_files = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        ]
        found = [p for p in program_files if Path(p).is_file()]
        return found + [n for n in names if shutil.which(n)]
    if sys.platform == "darwin":
        mac_paths = [
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
            "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
            "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ]
        found = [p for p in mac_paths if Path(p).is_file()]
        return found + [
            n
            for n in ("google-chrome", "chrome", "chromium", "msedge")
            if shutil.which(n)
        ]
    # Linux
    names = [
        "google-chrome-stable",
        "google-chrome",
        "chromium",
        "chromium-browser",
        "microsoft-edge",
        "microsoft-edge-stable",
        "chrome",
    ]
    return [n for n in names if shutil.which(n)]


def open_chrome_app_window(url: str) -> subprocess.Popen[Any] | None:
    """Launch Chrome/Edge/Chromium in --app mode (no URL bar)."""
    profile = str(_chrome_user_data_dir())
    for binary in _chrome_like_binaries():
        cmd = [
            binary,
            f"--app={url}",
            f"--user-data-dir={profile}",
            "--no-first-run",
            "--no-default-browser-check",
            f"--class={APP_TITLE}",
        ]
        try:
            proc = subprocess.Popen(  # noqa: S603
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # If Chrome handed off to an existing session, the child exits immediately.
            time.sleep(0.6)
            code = proc.poll()
            if code is not None:
                log.debug("%s exited early (code %s); trying next binary.", binary, code)
                continue
            log.info("Opened chrome-less app window via %s", binary)
            return proc
        except OSError as exc:
            log.debug("Could not launch %s: %s", binary, exc)
    return None


def open_pywebview(url: str) -> bool:
    """Open a native WebView window. Blocks until the window is closed.

    Returns True if the window ran successfully, False if pywebview/GUI is unavailable.
    """
    try:
        import webview
    except ImportError:
        log.info("pywebview not installed; trying browser app mode.")
        return False

    # Avoid scary ERROR traces when GTK/Qt are missing on Linux; we fall back cleanly.
    logging.getLogger("pywebview").setLevel(logging.CRITICAL)

    try:
        webview.create_window(
            APP_TITLE,
            url,
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            min_size=(900, 600),
            confirm_close=False,
            text_select=True,
        )
        # gui=None lets pywebview pick Edge/WebView2 (Win), Cocoa (mac), GTK/Qt (Linux)
        webview.start()
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Native WebView unavailable (%s); trying browser app mode.", exc)
        return False


def open_system_browser(url: str) -> None:
    webbrowser.open(url)


def run_app_ui(
    url: str,
    *,
    mode: str = "app",
    on_ready: Callable[[], None] | None = None,
) -> str:
    """Open the UI and block until the app window exits when possible.

    mode:
      - "app": prefer pywebview, then Chrome --app, then system browser
      - "browser": system browser tab (with URL bar)
      - "none": do not open a window

    Returns which strategy was used: "webview" | "chrome_app" | "browser" | "none".
    """
    if mode == "none":
        return "none"

    if not wait_for_server(url):
        log.warning("Server did not become ready at %s; opening UI anyway.", url)

    if on_ready:
        on_ready()

    if mode == "browser":
        open_system_browser(url)
        return "browser"

    # Prefer a true native window (no address bar).
    if open_pywebview(url):
        return "webview"

    proc = open_chrome_app_window(url)
    if proc is not None:
        try:
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
        return "chrome_app"

    log.info("Falling back to the system browser (URL bar may be visible).")
    open_system_browser(url)
    return "browser"
