"""Job progress / thinking_stages regression tests."""

from __future__ import annotations

from ramscout.pipeline import Job, JobStore, STORE


def test_job_has_thinking_stages_and_view_fields():
    job = Job(id="abc", url="https://example.com", created_at="now")
    assert job.thinking_stages == []
    assert job.views == {}
    assert job.bev == {}
    assert job.side_cues == []
    public = job.public()
    assert "thinking_stages" in public
    assert "views" in public
    assert "bev" in public
    assert "side_cues" in public


def test_set_progress_appends_thinking_stages(tmp_path, monkeypatch):
    monkeypatch.setattr("ramscout.pipeline.DATA", tmp_path)
    store = JobStore()
    job = store.create(url="https://example.com/watch?v=1")
    store.set_progress(job, "resolving", "Reading metadata…", 8)
    store.set_progress(job, "downloading", "Fetching video…", 30)
    assert len(job.thinking_stages) == 2
    assert job.thinking_stages[0]["status"] == "resolving"
    assert job.thinking_stages[1]["status"] == "downloading"
    assert job.status == "downloading"
    assert job.progress == 30
    assert job.public()["thinking_stages"][1]["message"] == "Fetching video…"


def test_set_progress_never_rewinds(tmp_path, monkeypatch):
    monkeypatch.setattr("ramscout.pipeline.DATA", tmp_path)
    store = JobStore()
    job = store.create(url="https://example.com/watch?v=1")
    store.set_progress(job, "tryout", "Auto mode: benchmarking 'hybrid' (1/4)…", 50)
    store.set_progress(job, "tracking", "Tracking robots…", 50)
    assert job.progress == 50
    store.set_progress(job, "tracking", "Tracking robots…", 48)
    assert job.progress == 50
    assert job.thinking_stages[-1]["progress"] == 50.0


def test_track_progress_ranges_do_not_overlap_backwards():
    from ramscout.pipeline import map_track_progress

    tryout_end = map_track_progress("Auto mode: benchmarking 'motion' (4/4)…", 100)[1]
    track_start = map_track_progress("Tracking robots…", 0)[1]
    assert tryout_end <= track_start
    assert map_track_progress("Tracking robots…", 100)[1] > track_start


def test_pipeline_does_not_call_laya_during_layout():
    from pathlib import Path

    text = Path("ramscout/pipeline.py").read_text(encoding="utf-8")
    assert "classify_camera_layout" not in text


def test_cancel_raises_before_the_bar_moves(tmp_path, monkeypatch):
    from ramscout.pipeline import JobCancelled

    monkeypatch.setattr("ramscout.pipeline.DATA", tmp_path)
    store = JobStore()
    job = store.create(url="https://example.com/watch?v=1")
    store.set_progress(job, "tracking", "Tracking robots…", 60)
    store.update(job, cancel_requested=True)
    try:
        store.set_progress(job, "tracking", "Tracking robots…", 70)
    except JobCancelled:
        pass
    else:
        raise AssertionError("cancel should stop the job")
    assert job.progress == 60


def test_set_progress_dedupes_identical_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr("ramscout.pipeline.DATA", tmp_path)
    store = JobStore()
    job = store.create(url="https://example.com/watch?v=1")
    store.set_progress(job, "tracking", "Working…", 55)
    store.set_progress(job, "tracking", "Working…", 60)
    assert len(job.thinking_stages) == 1
    assert job.thinking_stages[0]["progress"] == 60.0
