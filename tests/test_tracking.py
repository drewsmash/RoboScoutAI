"""Synthetic-video tests for tracking + scouting."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ramscout.detect import track_video
from ramscout.events import Pose, build_cards, detect_events
from ramscout.identity import assign_by_start, stitch_occlusions


def _write_moving_robots(path: Path, frames: int = 90, fps: int = 30) -> Path:
    import cv2

    w, h = 960, 540
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert writer.isOpened(), "could not open video writer"
    # Six robots: 3 blue left, 3 red right, drifting toward midfield hubs.
    robots = [
        {"x": 180, "y": 140, "vx": 2.2, "vy": 0.4, "color": (220, 80, 40)},   # blue-ish BGR? use blue
        {"x": 180, "y": 270, "vx": 2.0, "vy": 0.0, "color": (220, 80, 40)},
        {"x": 180, "y": 400, "vx": 2.4, "vy": -0.3, "color": (220, 80, 40)},
        {"x": 760, "y": 140, "vx": -2.1, "vy": 0.3, "color": (40, 40, 220)},
        {"x": 760, "y": 270, "vx": -2.3, "vy": 0.0, "color": (40, 40, 220)},
        {"x": 760, "y": 400, "vx": -2.0, "vy": -0.2, "color": (40, 40, 220)},
    ]
    # Fix colors: OpenCV BGR — blue bumpers=(220,80,40)? Actually blue is (255,0,0) in BGR = high B
    robots[0]["color"] = robots[1]["color"] = robots[2]["color"] = (220, 90, 30)  # blue-ish
    robots[3]["color"] = robots[4]["color"] = robots[5]["color"] = (40, 40, 230)  # red

    for i in range(frames):
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        # field carpet band
        frame[80:460, 40:920] = (45, 90, 45)
        for bot in robots:
            x = int(bot["x"] + bot["vx"] * i)
            y = int(bot["y"] + bot["vy"] * i)
            cv2.rectangle(frame, (x - 18, y - 14), (x + 18, y + 14), bot["color"], -1)
            cv2.rectangle(frame, (x - 18, y - 14), (x + 18, y + 14), (240, 240, 240), 2)
        writer.write(frame)
    writer.release()
    return path


def test_motion_tracking_produces_paths_and_scout_cards(tmp_path):
    video = _write_moving_robots(tmp_path / "bots.mp4")
    result = track_video(
        video,
        model_path=None,  # force motion-only path through ensure; still may load weights
        frame_stride=2,
        crop_top=0.05,
        crop_bottom=0.95,
        max_frames=50,
    )
    # Even if YOLO loads, motion/YOLO should see moving colored blocks.
    samples = stitch_occlusions(result["samples"])
    assert len(samples) > 20, f"expected tracks, got {len(samples)} warnings={result['warnings']}"

    blue = ["59", "2383", "11138"]
    red = ["179", "5472", "9201"]
    assignments = assign_by_start(samples, blue, red)
    assert assignments, "expected team assignments from start poses"

    poses = []
    for sample in samples:
        tid = int(sample["track_id"])
        team = assignments.get(tid) or sample.get("team") or str(tid)
        alliance = "blue" if team in blue else "red" if team in red else sample.get("alliance") or "blue"
        poses.append(
            Pose(
                t=float(sample["t"]),
                x=float(sample["x"]),
                y=float(sample["y"]),
                team=str(team),
                alliance=alliance,
                track_id=tid,
            )
        )
    events = detect_events(poses)
    cards = build_cards(poses, events)
    assert len(cards) >= 3
    assert sum(c.path_length_in for c in cards) > 0
