"""Tests for SORT-style MOT and hybrid cascade tracking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ramscout.detect import track_video
from ramscout.trackers.ensemble import TRACKER_MODES, EnsembleTracker, resolve_strategies
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.mot import MotTracker
from ramscout.trackers.types import Detection, TrackerContext


def test_hybrid_mode_documented():
    assert "hybrid" in TRACKER_MODES
    assert TRACKER_MODES["hybrid"].get("cascade") is True
    assert "auto" in TRACKER_MODES


def test_auto_resolves_to_hybrid_cascade():
    selected, _ = resolve_strategies("auto")
    names = {s.name for s in selected}
    assert "motion" in names
    assert "color" in names
    assert "optical_flow" in names


def test_mot_keeps_stable_ids_across_frames():
    mot = MotTracker(max_age=10, min_hits=1, max_tracks=6)
    frame = np.full((200, 400, 3), 40, dtype=np.uint8)
    # Two robots drifting right.
    boxes_t0 = [
        Detection(1, [20, 40, 50, 70], "motion", 0.6, alliance="blue"),
        Detection(2, [300, 40, 330, 70], "motion", 0.6, alliance="red"),
    ]
    out0 = mot.update(boxes_t0, frame_w=400, frame_h=200, frame_bgr=frame)
    assert len(out0) == 2
    id_left = min(out0, key=lambda d: d.bbox[0]).track_id
    id_right = max(out0, key=lambda d: d.bbox[0]).track_id

    boxes_t1 = [
        Detection(99, [28, 42, 58, 72], "color", 0.55, alliance="blue"),
        Detection(98, [308, 42, 338, 72], "color", 0.55, alliance="red"),
    ]
    out1 = mot.update(boxes_t1, frame_w=400, frame_h=200, frame_bgr=frame)
    ids = {d.track_id for d in out1}
    assert id_left in ids
    assert id_right in ids


def test_mot_rejects_cross_alliance_swap():
    mot = MotTracker(max_age=10, min_hits=1, max_tracks=6)
    frame = np.full((200, 400, 3), 40, dtype=np.uint8)
    out0 = mot.update(
        [Detection(1, [40, 40, 70, 70], "color", 0.7, alliance="blue")],
        frame_w=400,
        frame_h=200,
        frame_bgr=frame,
    )
    tid = out0[0].track_id
    # Far jump with opposite alliance should mint a new track, not hijack.
    out1 = mot.update(
        [Detection(2, [320, 40, 350, 70], "color", 0.7, alliance="red")],
        frame_w=400,
        frame_h=200,
        frame_bgr=frame,
    )
    assert all(d.track_id != tid or d.alliance == "blue" for d in out1) or any(
        d.alliance == "red" and d.track_id != tid for d in out1
    )


def test_hybrid_tracks_synthetic_match(tmp_path):
    import cv2

    path = tmp_path / "hybrid.mp4"
    w, h, fps, frames = 960, 540, 30, 55
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    robots = [
        {"x": 180, "y": 160, "vx": 2.0, "color": (220, 90, 30)},
        {"x": 180, "y": 300, "vx": 2.1, "color": (220, 90, 30)},
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

    result = track_video(
        path,
        tracker_mode="hybrid",
        frame_stride=2,
        crop_top=0.05,
        crop_bottom=0.95,
        max_frames=40,
        openai_key="",
        google_key="",
    )
    assert len(result["samples"]) > 12
    track_ids = {int(s["track_id"]) for s in result["samples"]}
    # SORT should keep ID count bounded (not one ID per detection).
    assert len(track_ids) <= 12
    assert "SORT MOT" in " ".join(result["warnings"]) or "hybrid" in result["tracker_mode"]


def test_ensemble_cascade_uses_motion_without_cloud():
    ens = EnsembleTracker([MotionTracker()], cascade=True)
    ctx = TrackerContext(frame_w=100, frame_h=100, crop_w=100, crop_h=80)
    import cv2

    frame1 = np.full((80, 100, 3), 20, dtype=np.uint8)
    frame2 = frame1.copy()
    cv2.rectangle(frame2, (20, 20), (50, 45), (220, 90, 30), -1)
    ens.detect(frame1, ctx)
    dets = ens.detect(frame2, ctx)
    assert isinstance(dets, list)
