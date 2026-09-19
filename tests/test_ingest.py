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
        calls.append(("extract", strategy.label))
        raise RuntimeError("Sign in to confirm you’re not a bot")

    def fake_auto(url, dest_dir, strategy, on_progress):
        calls.append(("auto", strategy.label))
        path = dest_dir / "vid.mp4"
        path.write_bytes(b"ok" * 60_000)
        return path

    def fake_cli(*_a, **_k):
        calls.append(("cli", "skip"))
        raise RuntimeError("cli blocked")

    monkeypatch.setattr("ramscout.ingest._ytdlp_extract", fake_extract)
    monkeypatch.setattr("ramscout.ingest._download_via_auto_proxy", fake_auto)
    monkeypatch.setattr("ramscout.ingest._download_via_cli", fake_cli)
    monkeypatch.setattr("ramscout.ingest._configured_proxy", lambda: None)
    monkeypatch.setattr("ramscout.ingest._auto_proxy_enabled", lambda: True)
    monkeypatch.setattr("ramscout.ingest._discover_proxies", lambda: ["1.2.3.4:8080"])
    path = download_video("https://youtu.be/m9uLAGKtenM", tmp_path / "out")
    assert path.exists()
    assert any(kind == "auto" for kind, _ in calls)


def test_resolve_cookies_path_env_and_appdata(tmp_path, monkeypatch):
    from ramscout.ingest import resolve_cookies_path

    cookie = tmp_path / "cookies.txt"
    cookie.write_text("# Netscape\n.youtube.com\tTRUE\t/\tFALSE\t0\tA\tB\n", encoding="utf-8")
    monkeypatch.setenv("YTDLP_COOKIES", str(cookie))
    assert resolve_cookies_path() == cookie.resolve()

    monkeypatch.delenv("YTDLP_COOKIES", raising=False)
    appdata = tmp_path / "AppData"
    dest = appdata / "RamScoutAI" / "cookies.txt"
    dest.parent.mkdir(parents=True)
    dest.write_text(cookie.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("APPDATA", str(appdata))
    monkeypatch.setattr("ramscout.ingest.app_dir", lambda: tmp_path / "missing-app")
    monkeypatch.setattr("ramscout.ingest.data_dir", lambda: tmp_path / "missing-data" / "data")
    found = resolve_cookies_path()
    assert found == dest.resolve()


def test_is_youtube_url_classification():
    from ramscout.ingest import is_youtube_url

    assert is_youtube_url("https://youtu.be/m9uLAGKtenM")
    assert is_youtube_url("https://www.youtube.com/watch?v=m9uLAGKtenM")
    assert not is_youtube_url("/tmp/match.mp4")
    assert not is_youtube_url("file:///tmp/match.mp4")
    assert not is_youtube_url("demo://sample")


def test_local_upload_info_skips_youtube_bot_hint(tmp_path):
    from ramscout.ingest import fetch_video_info

    video = tmp_path / "match.mp4"
    video.write_bytes(b"x" * 2048)
    info = fetch_video_info(str(video))
    assert info["source"] == "local"
    assert "warning" not in info or "bot" not in (info.get("warning") or "").lower()
