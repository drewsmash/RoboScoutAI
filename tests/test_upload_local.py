"""Upload / local-file ingest must never surface YouTube bot errors."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app import app
from ramscout.ingest import _as_local_path, is_youtube_url, resolve_cookies_path, store_uploaded_video
from ramscout.pipeline import STORE, start_job


client = TestClient(app)


def _tiny_mp4(path: Path, size: int = 4096) -> Path:
    path.write_bytes(b"\x00" * size)
    return path


def test_is_youtube_url_distinguishes_local():
    assert is_youtube_url("https://youtu.be/m9uLAGKtenM")
    assert is_youtube_url("https://www.youtube.com/watch?v=m9uLAGKtenM")
    assert not is_youtube_url("file:///tmp/match.mp4")
    assert not is_youtube_url("/tmp/match.mp4")
    assert not is_youtube_url("")


def test_as_local_path_file_uri(tmp_path):
    src = _tiny_mp4(tmp_path / "match.mp4")
    found = _as_local_path(src.resolve().as_uri())
    assert found is not None
    assert found.resolve() == src.resolve()


def test_store_uploaded_strips_uuid_prefix(tmp_path):
    src = _tiny_mp4(tmp_path / "aabbccddeeff00112233445566778899_qual65.mp4")
    out = store_uploaded_video(src, tmp_path / "job")
    assert out.name == "qual65.mp4"


def test_resolve_cookies_next_to_app(tmp_path, monkeypatch):
    cookies = tmp_path / "cookies.txt"
    cookies.write_text("# Netscape\n.", encoding="utf-8")
    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    monkeypatch.setattr("ramscout.ingest.app_dir", lambda: tmp_path)
    monkeypatch.setattr("ramscout.ingest.data_dir", lambda: tmp_path / "data")
    assert resolve_cookies_path() == cookies.resolve()


def test_upload_without_youtube_url_keeps_local_source(tmp_path):
    video = _tiny_mp4(tmp_path / "match.mp4", size=8192)
    with video.open("rb") as fh:
        res = client.post(
            "/api/jobs/upload",
            data={"crop_top": "0.10", "crop_bottom": "0.65", "tracker_mode": "motion"},
            files={"file": ("match.mp4", fh, "video/mp4")},
        )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["id"]
    assert body["has_video"] is True
    assert body.get("media_source") == "upload"
    assert not is_youtube_url(body["url"])
    # Job url must point at the stored copy, not a deleted staging temp.
    assert Path(body["url"]).is_file() or "match" in body["url"]

    job = None
    for _ in range(80):
        job = client.get(f"/api/jobs/{body['id']}").json()
        if job["status"] in {"ready", "error"}:
            break
        time.sleep(0.05)
    assert job is not None
    err = f"{job.get('error') or ''} {job.get('message') or ''}".lower()
    assert "bot" not in err
    assert "sign in" not in err
    assert "youtube blocked" not in err


def test_start_job_retargets_file_url_after_temp_delete(tmp_path):
    staging = tmp_path / "staging.mp4"
    _tiny_mp4(staging, size=2048)
    job = start_job(url=str(staging), local_video=staging, tracker_mode="motion")
    staging.unlink()
    assert job.video_path and Path(job.video_path).is_file()
    assert Path(job.url).is_file()
    # Background thread may still be running; wait briefly so it doesn't race suite teardown.
    for _ in range(40):
        current = STORE.get(job.id)
        if current and current.status in {"ready", "error"}:
            break
        time.sleep(0.05)


def test_upload_rejects_empty_file(tmp_path):
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    with empty.open("rb") as fh:
        res = client.post(
            "/api/jobs/upload",
            data={"crop_top": "0.10", "crop_bottom": "0.65"},
            files={"file": ("empty.mp4", fh, "video/mp4")},
        )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"].lower()
