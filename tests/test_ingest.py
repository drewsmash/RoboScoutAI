from ramscout.ingest import fetch_video_info


def test_oembed_fallback_for_blocked_youtube(monkeypatch):
    def boom(url, download=False, outtmpl=None, progress_hooks=None):
        raise RuntimeError("Sign in to confirm you’re not a bot")

    monkeypatch.setattr("ramscout.ingest._ytdlp_info", boom)
    info = fetch_video_info("https://youtu.be/m9uLAGKtenM")
    assert info["id"] == "m9uLAGKtenM"
    assert "Qualification 65" in info["title"]
    assert "Florida" in info["title"]
    assert info["source"] == "oembed"
    assert "blocked" in (info.get("warning") or "").lower() or "cookies" in (info.get("warning") or "").lower()
