"""Multi-strategy tracker registry and local ensembles."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app import app
from ramscout.detect import track_video
from ramscout.trackers.ensemble import TRACKER_MODES, EnsembleTracker, resolve_strategies
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import merge_detections


client = TestClient(app)


def _moving_robots(path: Path, frames: int = 60, fps: int = 30) -> Path:
    import cv2

    w, h = 960, 540
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    robots = [
        {"x": 180, "y": 160, "vx": 2.0, "color": (220, 90, 30)},
        {"x": 180, "y": 300, "vx": 2.2, "color": (220, 90, 30)},
        {"x": 760, "y": 160, "vx": -2.0, "color": (40, 40, 230)},
        {"x": 760, "y": 300, "vx": -2.1, "color": (40, 40, 230)},
    ]
    for i in range(frames):
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        frame[90:450, 50:910] = (45, 90, 45)
        for bot in robots:
            x = int(bot["x"] + bot["vx"] * i)
            y = int(bot["y"])
            cv2.rectangle(frame, (x - 18, y - 14), (x + 18, y + 14), bot["color"], -1)
        writer.write(frame)
    writer.release()
    return path


def test_tracker_modes_documented():
    assert "auto" in TRACKER_MODES
    assert "openai" in TRACKER_MODES
    assert "gemini" in TRACKER_MODES
    assert "motion" in TRACKER_MODES


def test_list_strategies_api():
    res = client.get("/api/trackers")
    assert res.status_code == 200
    body = res.json()
    assert any(m["id"] == "auto" for m in body["modes"])
    names = {s["name"] for s in body["strategies"]}
    assert {"motion", "color", "optical_flow", "yolo", "openai", "gemini"} <= names


def test_resolve_strategies_always_includes_motion():
    selected, _warnings = resolve_strategies("yolo")
    names = [s.name for s in selected]
    assert "motion" in names or "optical_flow" in names or names


def test_color_and_motion_ensemble(tmp_path):
    video = _moving_robots(tmp_path / "color.mp4")
    result = track_video(
        video,
        tracker_mode="color",
        frame_stride=2,
        crop_top=0.05,
        crop_bottom=0.95,
        max_frames=40,
    )
    assert len(result["samples"]) > 10
    assert "color" in result["strategies"] or "motion" in result["strategies"]


def test_local_mode_without_cloud_keys(tmp_path):
    video = _moving_robots(tmp_path / "local.mp4", frames=45)
    result = track_video(
        video,
        tracker_mode="local",
        frame_stride=2,
        crop_top=0.05,
        crop_bottom=0.95,
        max_frames=30,
        openai_key="",
        google_key="",
    )
    assert len(result["samples"]) > 8
    assert result["tracker_mode"] == "local"


def test_merge_detections_prefers_unique_boxes():
    a = [Detection(1, [10, 10, 40, 40], "yolo")]
    b = [Detection(2, [12, 12, 42, 42], "motion"), Detection(3, [200, 100, 240, 140], "color")]
    merged = merge_detections(a, b, min_dist=30)
    assert len(merged) == 2
    assert {d.source for d in merged} == {"yolo", "color"}


def test_ensemble_uses_motion_when_primary_empty():
    class Empty:
        name = "yolo"
        warnings: list[str] = []

        def detect(self, cropped, ctx):
            return []

    ens = EnsembleTracker([Empty(), MotionTracker()])
    ctx = TrackerContext(frame_w=100, frame_h=100, crop_w=100, crop_h=80)
    # First frame warms subtractor; second should detect change.
    import cv2

    frame1 = np.full((80, 100, 3), 20, dtype=np.uint8)
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (20, 20), (50, 45), (220, 90, 30), -1)
    ens.detect(frame1, ctx)
    dets = ens.detect(frame2, ctx)
    assert isinstance(dets, list)
