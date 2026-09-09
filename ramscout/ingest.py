"""Download recorded YouTube match videos and read metadata via yt-dlp."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

import httpx

from ramscout.titles import extract_youtube_id

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]

_BOT_HINT = (
    "YouTube blocked this cloud download (bot check / sign-in required). "
    "On your own machine, export cookies from a signed-in browser and set "
    "YTDLP_COOKIES=/path/to/cookies.txt, or run RamScoutAI locally where "
    "yt-dlp can use --cookies-from-browser."
)


def fetch_video_info(url: str) -> dict[str, Any]:
    """Read title/description. Falls back to oEmbed when yt-dlp is blocked."""
    video_id = extract_youtube_id(url) or ""
    try:
        info = _ytdlp_info(url, download=False)
        return {
            "id": info.get("id") or video_id,
            "title": info.get("title") or "",
            "description": info.get("description") or "",
            "duration": info.get("duration"),
            "uploader": info.get("uploader") or "",
            "webpage_url": info.get("webpage_url") or url,
            "thumbnail": info.get("thumbnail"),
            "is_live": bool(info.get("is_live") or info.get("live_status") == "is_live"),
            "source": "yt-dlp",
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("yt-dlp metadata failed (%s); trying oEmbed", exc)
        oembed = _oembed(url)
        if not oembed:
            raise RuntimeError(_friendly_youtube_error(exc)) from exc
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
            "warning": _friendly_youtube_error(exc),
        }


def download_video(url: str, dest_dir: Path, on_progress: ProgressFn | None = None) -> Path:
    """Download a recorded VOD as mp4. Live streams are rejected."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(dest_dir / "%(id)s.%(ext)s")

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

    try:
        info = _ytdlp_info(url, download=False)
        if info.get("is_live") or info.get("live_status") == "is_live":
            raise RuntimeError("This looks like a live stream. Paste a recorded match video instead.")
        info = _ytdlp_info(url, download=True, outtmpl=outtmpl, progress_hooks=[hook])
    except RuntimeError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(_friendly_youtube_error(exc)) from exc

    import yt_dlp

    path = Path(yt_dlp.YoutubeDL({"outtmpl": outtmpl}).prepare_filename(info)).with_suffix(".mp4")
    if path.exists():
        return path
    candidate = Path(yt_dlp.YoutubeDL({"outtmpl": outtmpl}).prepare_filename(info))
    if candidate.exists():
        return candidate
    # Prefer any downloaded media in the dest folder.
    for found in sorted(dest_dir.glob("*")):
        if found.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and found.is_file():
            return found
    raise RuntimeError("yt-dlp finished but the video file was not found.")


def _ytdlp_opts(download: bool, outtmpl: str | None = None, progress_hooks: list | None = None) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
    }
    cookies = (os.environ.get("YTDLP_COOKIES") or "").strip()
    if cookies and Path(cookies).is_file():
        opts["cookiefile"] = cookies
    browser = (os.environ.get("YTDLP_BROWSER") or "").strip()
    if browser and "cookiefile" not in opts:
        opts["cookiesfrombrowser"] = (browser,)
    if not download:
        opts["skip_download"] = True
    else:
        opts.update(
            {
                "outtmpl": outtmpl,
                "merge_output_format": "mp4",
                "format": "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b[height<=720]/b",
                "progress_hooks": progress_hooks or [],
            }
        )
    return opts


def _ytdlp_info(
    url: str,
    download: bool,
    outtmpl: str | None = None,
    progress_hooks: list | None = None,
) -> dict[str, Any]:
    import yt_dlp

    with yt_dlp.YoutubeDL(_ytdlp_opts(download, outtmpl, progress_hooks)) as ydl:
        return ydl.extract_info(url, download=download)


def _oembed(url: str) -> dict[str, Any] | None:
    try:
        res = httpx.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            timeout=15.0,
            headers={"User-Agent": "RamScoutAI/0.2"},
        )
        if res.status_code != 200:
            return None
        return res.json()
    except Exception as exc:  # noqa: BLE001
        log.info("oEmbed failed: %s", exc)
        return None


def _friendly_youtube_error(exc: Exception) -> str:
    text = str(exc)
    if re.search(r"sign in|not a bot|login_required|confirm you", text, re.I):
        return _BOT_HINT
    return f"Could not download the YouTube video: {text}"
