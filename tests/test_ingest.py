from pathlib import Path

from ramscout.ingest import download_video, fetch_video_info, store_uploaded_video


def test_oembed_fallback_for_blocked_youtube(monkeypatch):
    def boom(*_args, **_kwargs):
        raise RuntimeError("Sign in to confirm you’re not a bot")

    monkeypatch.setattr("ramscout.ingest._ytdlp_extract", boom)
    monkeypatch.setattr("ramscout.ingest._discover_proxies", lambda: [])
    info = fetch_video_info("https://youtu.be/m9uLAGKtenM")
    assert info["id"] == "m9uLAGKtenM"
    assert "Qualification 65" in info["title"]
    assert "Florida" in info["title"]
    assert info["source"] == "oembed"
    assert "blocked" in (info.get("warning") or "").lower() or "cookies" in (info.get("warning") or "").lower()


def test_local_file_download_copies(tmp_path):
    src = tmp_path / "match.mp4"
    src.write_bytes(b"fake-mp4")
    dest = tmp_path / "job"
    path = download_video(str(src), dest)
    assert path.exists()
    assert path.read_bytes() == b"fake-mp4"


def test_store_uploaded_video(tmp_path):
    src = tmp_path / "upload.webm"
    src.write_bytes(b"abc")
    out = store_uploaded_video(src, tmp_path / "job", preferred_name="m9uLAGKtenM")
    assert out.name == "m9uLAGKtenM.webm"
    assert out.read_bytes() == b"abc"


def test_download_tries_proxy_after_direct_block(monkeypatch, tmp_path):
    calls = []

    def fake_extract(url, *, download, strategy, outtmpl=None, progress_hooks=None, compact=False):
        calls.append(strategy.label)
        if strategy.proxy:
            if download:
                path = Path(outtmpl.replace("%(id)s", "vid").replace("%(ext)s", "mp4"))
                path.write_bytes(b"ok")
                return {"id": "vid", "ext": "mp4", "requested_downloads": [{"filepath": str(path)}]}
            return {"id": "vid", "title": "Qualification 1", "is_live": False}
        raise RuntimeError("Sign in to confirm you’re not a bot")

    monkeypatch.setattr("ramscout.ingest._ytdlp_extract", fake_extract)
    monkeypatch.setattr("ramscout.ingest._configured_proxy", lambda: None)
    monkeypatch.setattr("ramscout.ingest._auto_proxy_enabled", lambda: True)
    monkeypatch.setattr("ramscout.ingest._discover_proxies", lambda: ["1.2.3.4:8080"])
    path = download_video("https://youtu.be/m9uLAGKtenM", tmp_path / "out")
    assert path.exists()
    assert any(label.startswith("auto-proxy") for label in calls)
