"""Download recorded YouTube match videos and read metadata.

YouTube aggressively bot-checks datacenter IPs. This module tries several
strategies in order:

1. yt-dlp with modern player clients + optional PO-token plugins / cookies
2. The same, through ``YTDLP_PROXY`` (or ``HTTPS_PROXY``)
3. An optional auto-proxy ladder when ``YTDLP_AUTO_PROXY`` is enabled (default)
4. Local files already on disk (``file://`` / absolute paths)

Prefer running RamScoutAI on a residential network, or set cookies / a proxy.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import unquote, urlparse

import httpx

from ramscout.titles import extract_youtube_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

_BOT_HINT = (
    "YouTube blocked this download (bot check / sign-in required). "
    "Fixes that usually work: run RamScoutAI on your own network, set "
    "YTDLP_COOKIES to a Netscape cookies.txt from a signed-in browser, "
    "set YTDLP_PROXY to a residential HTTP proxy, or upload the match video file."
)

# Prefer clients that still work with PO tokens / guest sessions when not IP-blocked.
_CLIENT_STRATEGIES: tuple[tuple[str, ...], ...] = (
    ("tv", "android_vr"),
    ("mweb", "web"),
    ("android", "ios"),
    ("web_embedded", "tv"),
)

_FORMAT = "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b[height<=720]/b"
_FORMAT_COMPACT = "18/b[height<=480]/b[height<=720]/b"
_AUTO_PROXY_ATTEMPTS = 30
_AUTO_PROXY_TIMEOUT_S = 90.0


def fetch_video_info(url: str) -> dict[str, Any]:
    """Read title/description. Falls back to oEmbed when yt-dlp is blocked."""
    local = _as_local_path(url)
    if local is not None:
        return {
            "id": local.stem,
            "title": local.stem,
            "description": "",
            "duration": None,
            "uploader": "",
            "webpage_url": str(local),
            "thumbnail": None,
            "is_live": False,
            "source": "local",
        }

    video_id = extract_youtube_id(url) or ""
    errors: list[str] = []
    for strategy in _iter_download_strategies(download=False):
        try:
            info = _ytdlp_extract(url, download=False, strategy=strategy)
            return {
                "id": info.get("id") or video_id,
                "title": info.get("title") or "",
                "description": info.get("description") or "",
                "duration": info.get("duration"),
                "uploader": info.get("uploader") or "",
                "webpage_url": info.get("webpage_url") or url,
                "thumbnail": info.get("thumbnail"),
                "is_live": bool(info.get("is_live") or info.get("live_status") == "is_live"),
                "source": f"yt-dlp:{strategy.label}",
            }
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{strategy.label}: {exc}")
            log.info("metadata strategy %s failed: %s", strategy.label, exc)

    oembed = _oembed(url)
    if oembed:
        return {
            "id": video_id,
            "title": oembed.get("title") or "",
            "description": "",
            "duration": None,
            "uploader": oembed.get("author_name") or "",
            "webpage_url": url,
            "thumbnail": oembed.get("thumbnail_url"),
            "is_live": False,
            "source": "oembed",
            "warning": _friendly_youtube_error(errors[-1] if errors else "blocked"),
        }
    raise RuntimeError(_friendly_youtube_error(errors[-1] if errors else "unknown error"))


def download_video(url: str, dest_dir: Path, on_progress: ProgressFn | None = None) -> Path:
    """Download a recorded VOD as mp4, or copy a local file into dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    local = _as_local_path(url)
    if local is not None:
        if on_progress:
            on_progress("Using local match video…", 50.0)
        target = dest_dir / local.name
        if local.resolve() != target.resolve():
            shutil.copy2(local, target)
        if on_progress:
            on_progress("Local video ready.", 100.0)
        return target

    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    errors: list[str] = []

    for strategy in _iter_download_strategies(download=True):
        if on_progress:
            on_progress(f"Downloading via {strategy.label}…", 5.0)
        try:
            if strategy.label.startswith("auto-proxy"):
                path = _download_via_auto_proxy(url, dest_dir, strategy, on_progress)
            else:
                info = _ytdlp_extract(
                    url,
                    download=False,
                    strategy=strategy,
                    compact=strategy.use_proxy,
                )
                if info.get("is_live") or info.get("live_status") == "is_live":
                    raise RuntimeError("This looks like a live stream. Paste a recorded match video instead.")
                info = _ytdlp_extract(
                    url,
                    download=True,
                    strategy=strategy,
                    outtmpl=outtmpl,
                    progress_hooks=[_progress_hook(on_progress)],
                    compact=strategy.use_proxy,
                )
                path = _locate_download(dest_dir, outtmpl, info)
            if on_progress:
                on_progress(f"Download complete ({strategy.label}).", 100.0)
            log.info("Downloaded %s via %s", path.name, strategy.label)
            return path
        except RuntimeError as exc:
            if "live stream" in str(exc).lower():
                raise
            msg = str(exc)
            errors.append(f"{strategy.label}: {msg}")
            log.warning("download strategy %s failed: %s", strategy.label, msg)
            for partial in dest_dir.glob("*.part"):
                try:
                    partial.unlink()
                except OSError:
                    pass
            continue
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            errors.append(f"{strategy.label}: {msg}")
            log.warning("download strategy %s failed: %s", strategy.label, msg)
            # Clear partials so the next strategy starts clean.
            for partial in dest_dir.glob("*.part"):
                try:
                    partial.unlink()
                except OSError:
                    pass
            continue

    raise RuntimeError(_friendly_youtube_error("; ".join(errors[-3:]) if errors else "unknown error"))


def _download_via_auto_proxy(
    url: str,
    dest_dir: Path,
    strategy: _Strategy,
    on_progress: ProgressFn | None,
) -> Path:
    """One-shot yt-dlp CLI through a proxy with a hard kill timeout."""
    proxy = strategy.proxy or ""
    if "://" not in proxy:
        proxy = f"http://{proxy}"
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-plugins",
        "--no-warnings",
        "--retries",
        "2",
        "--socket-timeout",
        "15",
        "--proxy",
        proxy,
        "--extractor-args",
        "youtube:player_client=tv,android_vr",
        "-f",
        _FORMAT_COMPACT,
        "-o",
        outtmpl,
        "--merge-output-format",
        "mp4",
        url,
    ]
    cookies = (os.environ.get("YTDLP_COOKIES") or "").strip()
    if cookies and Path(cookies).is_file():
        cmd[3:3] = ["--cookies", cookies]
    if on_progress:
        on_progress(f"Trying proxy {proxy}…", 8.0)
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=_AUTO_PROXY_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f"auto-proxy timed out after {_AUTO_PROXY_TIMEOUT_S:.0f}s") from exc
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "yt-dlp failed").strip()
        raise RuntimeError(err[-400:])
    for found in sorted(dest_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if found.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and found.is_file() and found.stat().st_size > 100_000:
            return found
    raise RuntimeError("auto-proxy finished but no usable video file was written")


def store_uploaded_video(upload_path: Path, dest_dir: Path, preferred_name: str | None = None) -> Path:
    """Move/copy an uploaded match video into the job folder."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = upload_path.suffix.lower() or ".mp4"
    name = preferred_name or upload_path.name or f"upload{suffix}"
    if not Path(name).suffix:
        name = f"{name}{suffix}"
    target = dest_dir / Path(name).name
    shutil.copy2(upload_path, target)
    return target


class _Strategy:
    __slots__ = ("label", "clients", "proxy", "use_proxy")

    def __init__(self, label: str, clients: tuple[str, ...], proxy: str | None = None):
        self.label = label
        self.clients = clients
        self.proxy = (proxy or "").strip() or None
        self.use_proxy = bool(self.proxy)


def _iter_download_strategies(download: bool) -> Iterable[_Strategy]:
    user_proxy = _configured_proxy()
    for clients in _CLIENT_STRATEGIES:
        label = "+".join(clients)
        if user_proxy:
            yield _Strategy(f"proxy/{label}", clients, user_proxy)
        yield _Strategy(f"direct/{label}", clients, None)

    if download and _auto_proxy_enabled():
        proxies = _discover_proxies()
        random.shuffle(proxies)
        for i, proxy in enumerate(proxies[:_AUTO_PROXY_ATTEMPTS]):
            # Compact progressive formats survive flaky free proxies better.
            yield _Strategy(f"auto-proxy#{i}/tv+android_vr", ("tv", "android_vr"), proxy)


def _configured_proxy() -> str | None:
    for key in ("YTDLP_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy"):
        value = (os.environ.get(key) or "").strip()
        if value:
            return value
    return None


def _auto_proxy_enabled() -> bool:
    raw = (os.environ.get("YTDLP_AUTO_PROXY") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _discover_proxies() -> list[str]:
    """Best-effort public proxy candidates for cloud / datacenter IPs."""
    urls = [
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=us&ssl=all&anonymity=all",
        "https://api.proxyscrape.com/v2/?request=displayproxies&protocol=http&timeout=5000&country=all&ssl=all&anonymity=elite",
        "https://raw.githubusercontent.com/TheSpeedX/PROXY-List/master/http.txt",
    ]
    found: list[str] = []
    seen: set[str] = set()
    for url in urls:
        try:
            res = httpx.get(url, timeout=12.0, follow_redirects=True)
            if res.status_code != 200:
                continue
            for line in res.text.splitlines():
                proxy = line.strip()
                if not proxy or proxy.startswith("#") or ":" not in proxy:
                    continue
                # Reject obvious non-proxy text.
                if " " in proxy or len(proxy) > 64:
                    continue
                if not re.match(r"^[\w\.\-]+:\d{2,5}$", proxy):
                    continue
                if proxy in seen:
                    continue
                seen.add(proxy)
                found.append(proxy)
        except Exception as exc:  # noqa: BLE001
            log.info("proxy list %s failed: %s", url, exc)
    log.info("Discovered %d proxy candidates for YouTube fallback", len(found))
    return found


def _ytdlp_opts(
    *,
    download: bool,
    strategy: _Strategy,
    outtmpl: str | None = None,
    progress_hooks: list | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 8,
        "fragment_retries": 8,
        "socket_timeout": 20 if strategy.proxy else 30,
        "extractor_args": {"youtube": {"player_client": list(strategy.clients)}},
        "js_runtimes": {"deno": {}},
    }
    if strategy.proxy:
        opts["proxy"] = strategy.proxy if "://" in strategy.proxy else f"http://{strategy.proxy}"

    cookies = (os.environ.get("YTDLP_COOKIES") or "").strip()
    if cookies and Path(cookies).is_file():
        opts["cookiefile"] = cookies
    browser = (os.environ.get("YTDLP_BROWSER") or "").strip()
    if browser and "cookiefile" not in opts:
        opts["cookiesfrombrowser"] = (browser,)

    chrome = _chrome_path()
    if chrome:
        opts.setdefault("extractor_args", {}).setdefault("youtubepot-wpc", {})["browser_path"] = chrome

    if not download:
        opts["skip_download"] = True
    else:
        opts.update(
            {
                "outtmpl": outtmpl,
                "merge_output_format": "mp4",
                "format": _FORMAT_COMPACT if compact else _FORMAT,
                "progress_hooks": progress_hooks or [],
            }
        )
    return opts


def _ytdlp_extract(
    url: str,
    *,
    download: bool,
    strategy: _Strategy,
    outtmpl: str | None = None,
    progress_hooks: list | None = None,
    compact: bool = False,
) -> dict[str, Any]:
    import yt_dlp

    with yt_dlp.YoutubeDL(
        _ytdlp_opts(
            download=download,
            strategy=strategy,
            outtmpl=outtmpl,
            progress_hooks=progress_hooks,
            compact=compact,
        )
    ) as ydl:
        return ydl.extract_info(url, download=download)


def _progress_hook(on_progress: ProgressFn | None):
    def hook(status: dict[str, Any]) -> None:
        if not on_progress:
            return
        if status.get("status") == "downloading":
            total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
            done = status.get("downloaded_bytes") or 0
            pct = (done / total * 100.0) if total else 0.0
            on_progress("Downloading match video…", pct)
        elif status.get("status") == "finished":
            on_progress("Download complete.", 100.0)

    return hook


def _locate_download(dest_dir: Path, outtmpl: str, info: dict[str, Any]) -> Path:
    import yt_dlp

    path = Path(yt_dlp.YoutubeDL({"outtmpl": outtmpl}).prepare_filename(info)).with_suffix(".mp4")
    if path.exists() and path.stat().st_size > 0:
        return path
    candidate = Path(yt_dlp.YoutubeDL({"outtmpl": outtmpl}).prepare_filename(info))
    if candidate.exists() and candidate.stat().st_size > 0:
        return candidate
    for found in sorted(dest_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if found.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and found.is_file() and found.stat().st_size > 0:
            return found
    raise RuntimeError("yt-dlp finished but the video file was not found.")


def _as_local_path(url: str) -> Path | None:
    text = (url or "").strip()
    if not text:
        return None
    if text.startswith("file://"):
        parsed = urlparse(text)
        path = Path(unquote(parsed.path))
        return path if path.is_file() else None
    path = Path(text)
    if path.is_file() and path.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".avi"}:
        return path
    return None


def _chrome_path() -> str | None:
    for candidate in (
        os.environ.get("CHROME_PATH"),
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable",
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
    ):
        if candidate and Path(candidate).is_file():
            return candidate
    return shutil.which("google-chrome") or shutil.which("chromium") or shutil.which("chromium-browser")


def _oembed(url: str) -> dict[str, Any] | None:
    try:
        res = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=15.0,
            headers={"User-Agent": "RamScoutAI/0.3"},
        )
        if res.status_code != 200:
            return None
        return res.json()
    except Exception as exc:  # noqa: BLE001
        log.info("oEmbed failed: %s", exc)
        return None


def _friendly_youtube_error(exc: Exception | str) -> str:
    text = str(exc)
    if re.search(r"sign in|not a bot|login_required|confirm you|bot check|blocked", text, re.I):
        return _BOT_HINT
    return f"Could not download the YouTube video: {text}"
