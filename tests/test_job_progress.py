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


def test_set_progress_dedupes_identical_ticks(tmp_path, monkeypatch):
    monkeypatch.setattr("ramscout.pipeline.DATA", tmp_path)
    store = JobStore()
    job = store.create(url="https://example.com/watch?v=1")
    store.set_progress(job, "tracking", "Working…", 55)
    store.set_progress(job, "tracking", "Working…", 60)
    assert len(job.thinking_stages) == 1
    assert job.thinking_stages[0]["progress"] == 60.0
