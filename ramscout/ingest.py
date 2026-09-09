"""Download recorded YouTube match videos and read metadata via yt-dlp."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

ProgressFn = Callable[[str, float], None]


def fetch_video_info(url: str) -> dict[str, Any]:
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "id": info.get("id"),
        "title": info.get("title") or "",
        "description": info.get("description") or "",
        "duration": info.get("duration"),
        "uploader": info.get("uploader") or "",
        "webpage_url": info.get("webpage_url") or url,
        "thumbnail": info.get("thumbnail"),
        "is_live": bool(info.get("is_live") or info.get("live_status") == "is_live"),
    }


def download_video(url: str, dest_dir: Path, on_progress: ProgressFn | None = None) -> Path:
    """Download a recorded VOD as mp4. Live streams are rejected."""
    import yt_dlp

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

    opts = {
        "quiet": True,
        "no_warnings": True,
        "outtmpl": outtmpl,
        "merge_output_format": "mp4",
        "format": "bv*[ext=mp4][height<=720]+ba[ext=m4a]/b[ext=mp4][height<=720]/b[height<=720]/b",
        "progress_hooks": [hook],
        "noprogress": True,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if info.get("is_live") or info.get("live_status") == "is_live":
            raise RuntimeError("This looks like a live stream. Paste a recorded match video instead.")
        info = ydl.extract_info(url, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if not path.exists():
            # yt-dlp may keep the original extension if already mp4.
            candidate = Path(ydl.prepare_filename(info))
            if candidate.exists():
                return candidate
            raise RuntimeError("yt-dlp finished but the video file was not found.")
        return path
