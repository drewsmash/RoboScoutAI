"""Download recorded YouTube match videos and read metadata.

YouTube aggressively bot-checks datacenter IPs. This module tries a ladder:

1. yt-dlp with multiple player clients + format fallbacks (progressive → muxed)
2. Same through ``YTDLP_PROXY`` / ``HTTPS_PROXY`` when set
3. Optional cookies (``YTDLP_COOKIES`` file or ``YTDLP_BROWSER``)
4. Subprocess yt-dlp CLI fallback (sometimes more resilient than the Python API)
5. Optional auto-proxy ladder when ``YTDLP_AUTO_PROXY`` is enabled (default on)
6. Local files / uploads (``file://`` / absolute paths) — always preferred when available

Prefer running RoboScoutAI on a residential network, or set cookies / a proxy.
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
from urllib.request import url2pathname

import httpx

from ramscout.paths import app_dir, data_dir
from ramscout.titles import extract_youtube_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

_BOT_HINT = (
    "YouTube blocked this download (bot check / sign-in required). "
    "Best fix: upload the MP4/MKV you already downloaded. "
    "Or place cookies.txt next to the EXE / in %APPDATA%\\RoboScoutAI\\, "
    "set YTDLP_COOKIES, use YTDLP_BROWSER=chrome|edge|firefox, "
    "or set YTDLP_PROXY to a residential HTTP proxy."
)

_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}

# Prefer clients that still work with guest sessions / PO tokens when not IP-blocked.
_CLIENT_STRATEGIES: tuple[tuple[str, ...], ...] = (
    ("android", "ios"),
    ("tv", "android_vr"),
    ("web_embedded", "mweb"),
    ("tv_simply", "mediaconnect"),
    ("web",),
)

# Format ladder: progressive first (no ffmpeg merge), then adaptive muxed 720p.
_FORMATS: tuple[str, ...] = (
    "18/22/best[height<=480][ext=mp4]/best[height<=480]/best[height<=360]",
    "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b[height<=720]/b",
    "bv*+ba/b",
    "worst",
)

_AUTO_PROXY_ATTEMPTS = 20
_AUTO_PROXY_TIMEOUT_S = 75.0


class _Strategy:
    __slots__ = ("label", "clients", "proxy", "use_proxy", "fmt", "via_cli")

    def __init__(
        self,
        label: str,
        clients: tuple[str, ...],
        proxy: str | None = None,
        fmt: str | None = None,
        via_cli: bool = False,
    ):
        self.label = label
        self.clients = clients
        self.proxy = (proxy or "").strip() or None
        self.use_proxy = bool(self.proxy)
        self.fmt = fmt
        self.via_cli = via_cli


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
            elif strategy.via_cli:
                path = _download_via_cli(url, dest_dir, strategy, on_progress)
            else:
                info = _ytdlp_extract(
                    url,
                    download=False,
                    strategy=strategy,
                )
                if info.get("is_live") or info.get("live_status") == "is_live":
                    raise RuntimeError("This looks like a live stream. Paste a recorded match video instead.")
                info = _ytdlp_extract(
                    url,
                    download=True,
                    strategy=strategy,
                    outtmpl=outtmpl,
                    progress_hooks=[_progress_hook(on_progress)],
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
            _clear_partials(dest_dir)
            continue
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            errors.append(f"{strategy.label}: {msg}")
            log.warning("download strategy %s failed: %s", strategy.label, msg)
            _clear_partials(dest_dir)
            continue

    raise RuntimeError(_friendly_youtube_error("; ".join(errors[-4:]) if errors else "unknown error"))


def _clear_partials(dest_dir: Path) -> None:
    for partial in dest_dir.glob("*.part"):
        try:
            partial.unlink()
        except OSError:
            pass
    for partial in dest_dir.glob("*.ytdl"):
        try:
            partial.unlink()
        except OSError:
            pass


def _download_via_cli(
    url: str,
    dest_dir: Path,
    strategy: _Strategy,
    on_progress: ProgressFn | None,
) -> Path:
    """CLI yt-dlp path — useful when the Python API + JS runtime misbehave."""
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")
    fmt = strategy.fmt or _FORMATS[0]
    clients = ",".join(strategy.clients)
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--socket-timeout",
        "25",
        "--extractor-args",
        f"youtube:player_client={clients}",
        "-f",
        fmt,
        "-o",
        outtmpl,
        "--merge-output-format",
        "mp4",
        url,
    ]
    cookie_file = resolve_cookies_path()
    if cookie_file is not None:
        cmd[3:3] = ["--cookies", str(cookie_file)]
    browser = (os.environ.get("YTDLP_BROWSER") or "").strip()
    if browser and cookie_file is None:
        cmd[3:3] = ["--cookies-from-browser", browser]
    if strategy.proxy:
        proxy = strategy.proxy if "://" in strategy.proxy else f"http://{strategy.proxy}"
        cmd[3:3] = ["--proxy", proxy]
    if on_progress:
        on_progress(f"CLI download ({strategy.label})…", 10.0)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "yt-dlp CLI failed").strip()
        raise RuntimeError(err[-500:])
    for found in sorted(dest_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if found.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and found.is_file() and found.stat().st_size > 50_000:
            return found
    raise RuntimeError("CLI yt-dlp finished but no usable video file was written")


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
    fmt = strategy.fmt or _FORMATS[0]
    cmd = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-plugin-dirs",
        "--no-warnings",
        "--retries",
        "2",
        "--socket-timeout",
        "15",
        "--proxy",
        proxy,
        "--extractor-args",
        "youtube:player_client=android,ios",
        "-f",
        fmt,
        "-o",
        outtmpl,
        "--merge-output-format",
        "mp4",
        url,
    ]
    cookie_file = resolve_cookies_path()
    if cookie_file is not None:
        cmd[3:3] = ["--cookies", str(cookie_file)]
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
    upload_path = Path(upload_path)
    if not upload_path.is_file():
        raise FileNotFoundError(
            f"Uploaded video is missing: {upload_path}. "
            "Re-select the MP4/MKV and try again."
        )
    if upload_path.stat().st_size <= 0:
        raise ValueError("Uploaded video file is empty.")
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = upload_path.suffix.lower() or ".mp4"
    if suffix not in _UPLOAD_SUFFIXES:
        suffix = ".mp4"
    name = preferred_name or upload_path.name or f"upload{suffix}"
    # Strip accidental UUID staging prefixes like "a1b2…_match.mp4".
    clean = Path(name).name
    if "_" in clean and len(clean.split("_", 1)[0]) == 32:
        maybe = clean.split("_", 1)[1]
        if maybe:
            clean = maybe
    if not Path(clean).suffix:
        clean = f"{clean}{suffix}"
    target = dest_dir / clean
    if upload_path.resolve() != target.resolve():
        shutil.copy2(upload_path, target)
    return target


def resolve_cookies_path() -> Path | None:
    """Locate Netscape cookies.txt for yt-dlp (env, app dir, or APPDATA)."""
    env = (os.environ.get("YTDLP_COOKIES") or "").strip()
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).expanduser())
    candidates.extend(
        [
            app_dir() / "cookies.txt",
            data_dir().parent / "cookies.txt",
            Path.home() / "RoboScoutAI" / "cookies.txt",
            Path.home() / "RamScoutAI" / "cookies.txt",  # silent old-folder fallback
        ]
    )
    appdata = (os.environ.get("APPDATA") or "").strip()
    if appdata:
        candidates.append(Path(appdata) / "RoboScoutAI" / "cookies.txt")
        candidates.append(Path(appdata) / "RamScoutAI" / "cookies.txt")
    local = (os.environ.get("LOCALAPPDATA") or "").strip()
    if local:
        candidates.append(Path(local) / "RoboScoutAI" / "cookies.txt")
    xdg = (os.environ.get("XDG_CONFIG_HOME") or "").strip()
    if xdg:
        candidates.append(Path(xdg) / "RoboScoutAI" / "cookies.txt")
    else:
        candidates.append(Path.home() / ".config" / "RoboScoutAI" / "cookies.txt")
    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            if path.is_file() and path.stat().st_size > 0:
                return path.resolve()
        except OSError:
            continue
    return None


def is_youtube_url(url: str) -> bool:
    text = (url or "").strip().lower()
    if not text or text.startswith("file://") or text.startswith("demo://"):
        return False
    if extract_youtube_id(url):
        return True
    return any(host in text for host in ("youtube.com", "youtu.be", "youtube-nocookie.com"))


def _iter_download_strategies(download: bool) -> Iterable[_Strategy]:
    user_proxy = _configured_proxy()
    formats = _format_ladder()

    # Fast progressive clients first (android/ios often bypass web bot checks).
    for clients in _CLIENT_STRATEGIES:
        for fmt in formats[:2] if download else formats[:1]:
            label_base = "+".join(clients)
            fmt_tag = "prog" if "18/" in fmt or "worst" in fmt else "mux"
            if user_proxy:
                yield _Strategy(f"proxy/{label_base}/{fmt_tag}", clients, user_proxy, fmt=fmt)
            yield _Strategy(f"direct/{label_base}/{fmt_tag}", clients, None, fmt=fmt)

    if download:
        # CLI fallback with progressive format — avoids Python JS-runtime pitfalls.
        yield _Strategy("cli/android+ios/prog", ("android", "ios"), None, fmt=formats[0], via_cli=True)
        if user_proxy:
            yield _Strategy(
                "cli-proxy/android+ios/prog",
                ("android", "ios"),
                user_proxy,
                fmt=formats[0],
                via_cli=True,
            )

    if download and _auto_proxy_enabled():
        proxies = _discover_proxies()
        random.shuffle(proxies)
        for i, proxy in enumerate(proxies[:_AUTO_PROXY_ATTEMPTS]):
            yield _Strategy(
                f"auto-proxy#{i}/android+ios",
                ("android", "ios"),
                proxy,
                fmt=formats[0],
            )


def _format_ladder() -> tuple[str, ...]:
    override = (os.environ.get("YTDLP_FORMAT") or "").strip()
    if override:
        return (override, *_FORMATS)
    return _FORMATS


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
) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 10,
        "fragment_retries": 10,
        "socket_timeout": 20 if strategy.proxy else 35,
        "extractor_args": {"youtube": {"player_client": list(strategy.clients)}},
    }
    # Only request a JS runtime when Deno/Node is actually present — otherwise
    # yt-dlp can fail early on web clients that need it.
    runtime = _js_runtime()
    if runtime:
        opts["js_runtimes"] = runtime

    if strategy.proxy:
        opts["proxy"] = strategy.proxy if "://" in strategy.proxy else f"http://{strategy.proxy}"

    cookie_file = resolve_cookies_path()
    if cookie_file is not None:
        opts["cookiefile"] = str(cookie_file)
    browser = (os.environ.get("YTDLP_BROWSER") or "").strip()
    if browser and "cookiefile" not in opts:
        opts["cookiesfrombrowser"] = (browser,)

    chrome = _chrome_path()
    if chrome:
        opts.setdefault("extractor_args", {}).setdefault("youtubepot-wpc", {})["browser_path"] = chrome

    if not download:
        opts["skip_download"] = True
    else:
        fmt = strategy.fmt or _FORMATS[0]
        opts.update(
            {
                "outtmpl": outtmpl,
                "merge_output_format": "mp4",
                "format": fmt,
                "progress_hooks": progress_hooks or [],
            }
        )
    return opts


def _js_runtime() -> dict[str, dict] | None:
    if shutil.which("deno"):
        return {"deno": {}}
    if shutil.which("node"):
        return {"node": {}}
    return None


def _ytdlp_extract(
    url: str,
    *,
    download: bool,
    strategy: _Strategy,
    outtmpl: str | None = None,
    progress_hooks: list | None = None,
    compact: bool = False,  # kept for older call sites / tests
) -> dict[str, Any]:
    import yt_dlp

    # compact flag historically forced progressive formats through proxies.
    if compact and strategy.fmt is None:
        strategy = _Strategy(strategy.label, strategy.clients, strategy.proxy, fmt=_FORMATS[0])

    with yt_dlp.YoutubeDL(
        _ytdlp_opts(
            download=download,
            strategy=strategy,
            outtmpl=outtmpl,
            progress_hooks=progress_hooks,
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
        # url2pathname handles Windows file:///C:/… → C:\…
        raw = url2pathname(unquote(parsed.path or ""))
        if parsed.netloc and parsed.netloc not in {"", "localhost"}:
            # UNC or host-qualified paths — uncommon for local uploads.
            raw = f"//{parsed.netloc}{raw}"
        path = Path(raw)
        if path.is_file() and path.suffix.lower() in _UPLOAD_SUFFIXES:
            return path
        return None
    path = Path(text).expanduser()
    try:
        if path.is_file() and path.suffix.lower() in _UPLOAD_SUFFIXES:
            return path.resolve()
    except OSError:
        return None
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
            headers={"User-Agent": "RoboScoutAI/0.5"},
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
    if re.search(r"format is not available|requested format", text, re.I):
        return (
            "Could not find a compatible YouTube format. "
            "Try uploading the match video file, or set YTDLP_FORMAT / YTDLP_COOKIES."
        )
    return f"Could not download the YouTube video: {text}"
