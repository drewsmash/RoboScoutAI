"""Extra YouTube download ladder coverage."""

from __future__ import annotations

from pathlib import Path

from ramscout.ingest import _Strategy, _format_ladder, _iter_download_strategies, download_video


def test_format_ladder_respects_override(monkeypatch):
    monkeypatch.setenv("YTDLP_FORMAT", "best[height<=360]")
    ladder = _format_ladder()
    assert ladder[0] == "best[height<=360]"


def test_download_strategies_include_cli_and_android(monkeypatch):
    monkeypatch.setattr("ramscout.ingest._configured_proxy", lambda: None)
    monkeypatch.setattr("ramscout.ingest._auto_proxy_enabled", lambda: False)
    labels = [s.label for s in _iter_download_strategies(download=True)]
    assert any("android" in lab for lab in labels)
    assert any(lab.startswith("cli/") for lab in labels)
    assert any("/prog" in lab for lab in labels)


def test_download_falls_back_across_formats(monkeypatch, tmp_path):
    calls: list[str] = []

    def fake_extract(url, *, download, strategy, outtmpl=None, progress_hooks=None, compact=False):
        calls.append(strategy.label)
        if "prog" in strategy.label and not strategy.via_cli:
            raise RuntimeError("Requested format is not available")
        if download:
            path = Path(outtmpl.replace("%(id)s", "vid").replace("%(ext)s", "mp4"))
            path.write_bytes(b"ok" * 80_000)
            return {"id": "vid", "ext": "mp4", "is_live": False}
        return {"id": "vid", "is_live": False, "title": "t"}

    monkeypatch.setattr("ramscout.ingest._ytdlp_extract", fake_extract)
    monkeypatch.setattr("ramscout.ingest._configured_proxy", lambda: None)
    monkeypatch.setattr("ramscout.ingest._auto_proxy_enabled", lambda: False)
    monkeypatch.setattr(
        "ramscout.ingest._download_via_cli",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("cli skip")),
    )
    path = download_video("https://youtu.be/m9uLAGKtenM", tmp_path / "out")
    assert path.exists()
    assert any("/mux" in c for c in calls)


def test_strategy_carries_format():
    s = _Strategy("direct/android/prog", ("android",), fmt="18/b")
    assert s.fmt == "18/b"
    assert s.clients == ("android",)
