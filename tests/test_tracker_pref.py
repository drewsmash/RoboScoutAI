"""The last winning local tracker is reused instead of a full tryout."""

from __future__ import annotations

from ramscout.tracker_pref import load_last_tracker, save_last_tracker


def test_last_local_tracker_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("ramscout.tracker_pref.data_dir", lambda: tmp_path)
    assert load_last_tracker() is None
    save_last_tracker("hybrid", scores={"hybrid": {"total": 0.4}})
    assert load_last_tracker() == "hybrid"
    save_last_tracker("openai")
    assert load_last_tracker() == "hybrid"
    save_last_tracker("color")
    assert load_last_tracker() == "color"
