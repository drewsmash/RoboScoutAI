"""End-to-end regression for the failure modes users reported:

* the field perimeter / LED wall being tracked as robots,
* alliance-colored static field elements and driver-station glass,
* red/blue confusion under a broadcast white-balance cast and with a
  wrong-color mechanism on top of the robot.

A synthetic broadcast with ground truth lets us assert on-robot rate, 3/3
alliances, and per-sample alliance accuracy without any model or API key.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from ramscout.detect import track_video
from ramscout.identity import balance_alliances, keep_top_tracks, stitch_occlusions

W, H, FPS = 960, 540, 30
CROP_TOP, CROP_BOT = 0.10, 0.65


def _wb(color, gains):
    return tuple(int(min(255, c * g)) for c, g in zip(color, gains))


def _broadcast(path: Path, *, frames: int = 150, gains=(1.0, 1.0, 1.0), seed: int = 0):
    import cv2

    rng = np.random.default_rng(seed)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    assert writer.isOpened()
    y0, y1 = int(H * CROP_TOP), int(H * CROP_BOT)
    ch = y1 - y0
    top_y, bot_y = y0 + int(0.08 * ch), y0 + int(0.92 * ch)
    tl, tr, bl, br = int(0.18 * W), int(0.82 * W), int(0.03 * W), int(0.97 * W)
    poly = np.array([[tl, top_y], [tr, top_y], [br, bot_y], [bl, bot_y]], np.int32)
    blue_c, red_c = _wb((200, 70, 20), gains), _wb((30, 30, 210), gains)
    carpet, wall = _wb((60, 80, 60), gains), _wb((170, 170, 175), gains)

    def edge_x(y, side):
        f = (y - top_y) / max(bot_y - top_y, 1)
        return (tl + (bl - tl) * f) if side == "l" else (tr + (br - tr) * f)

    robots = []
    for i in range(3):
        robots.append({"x": 0.22, "y": 0.2 + 0.3 * i, "vx": 0.0028, "vy": rng.uniform(-0.0008, 0.0008), "al": "blue"})
        robots.append({"x": 0.78, "y": 0.2 + 0.3 * i, "vx": -0.0028, "vy": rng.uniform(-0.0008, 0.0008), "al": "red"})
    truth: list[tuple[int, int, str, float, float]] = []
    mid_y = (top_y + bot_y) // 2
    for f in range(frames):
        frame = np.full((H, W, 3), 25, np.uint8)
        cv2.fillPoly(frame, [poly], carpet)
        # Flickering LED wall along the whole perimeter.
        jitter = np.clip(np.array(wall) + rng.integers(-30, 30, 3), 0, 255)
        cv2.polylines(frame, [poly], True, tuple(int(v) for v in jitter), 8)
        # Alliance-lit driver-station glass outside the field.
        cv2.rectangle(frame, (tl - 40, top_y - 30), (tl + 150, top_y - 6), _wb((150, 60, 40), gains), -1)
        cv2.rectangle(frame, (tr - 150, top_y - 30), (tr + 40, top_y - 6), _wb((40, 40, 150), gains), -1)
        # Static alliance-colored field elements inside the field.
        hx_l = int(edge_x(mid_y, "l") + 0.28 * W)
        hx_r = int(edge_x(mid_y, "r") - 0.28 * W)
        cv2.rectangle(frame, (hx_l - 16, mid_y - 16), (hx_l + 16, mid_y + 16), blue_c, -1)
        cv2.rectangle(frame, (hx_r - 16, mid_y - 16), (hx_r + 16, mid_y + 16), red_c, -1)
        # Scorebug with alliance colors and a ticking clock.
        cv2.rectangle(frame, (int(0.25 * W), 0), (int(0.75 * W), int(0.13 * H)), (20, 20, 20), -1)
        cv2.rectangle(frame, (int(0.27 * W), int(0.02 * H)), (int(0.49 * W), int(0.11 * H)), red_c, -1)
        cv2.rectangle(frame, (int(0.51 * W), int(0.02 * H)), (int(0.73 * W), int(0.11 * H)), blue_c, -1)
        cv2.putText(frame, f"{f // 30:02d}", (int(0.47 * W), int(0.09 * H)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        for i, bot in enumerate(robots):
            bot["x"] += bot["vx"]
            bot["y"] += bot["vy"]
            if not 0.1 < bot["y"] < 0.9:
                bot["vy"] *= -1
            if not 0.12 < bot["x"] < 0.88:
                bot["vx"] *= -1
            fy = top_y + bot["y"] * (bot_y - top_y)
            fx = edge_x(fy, "l") + bot["x"] * (edge_x(fy, "r") - edge_x(fy, "l"))
            scale = 0.7 + 0.6 * bot["y"]
            bw, bh = int(34 * scale), int(40 * scale)
            x1, ytop = int(fx - bw / 2), int(fy - bh)
            cv2.rectangle(frame, (x1, ytop), (x1 + bw, int(fy)), _wb((90, 90, 90), gains), -1)
            # Wrong-color mechanism on top, correct bumper at the bottom.
            decoy = red_c if bot["al"] == "blue" else blue_c
            cv2.rectangle(frame, (x1 + 4, ytop + 3), (x1 + bw - 4, ytop + int(bh * 0.25)), decoy, -1)
            bump = blue_c if bot["al"] == "blue" else red_c
            cv2.rectangle(frame, (x1, int(fy - bh * 0.28)), (x1 + bw, int(fy)), bump, -1)
            truth.append((f, i, bot["al"], fx, fy))
        frame = cv2.add(frame, rng.integers(0, 5, (H, W, 3), dtype=np.uint8))
        writer.write(frame)
    writer.release()
    return truth


def _score(samples, truth, tol_px: float = 55.0):
    by_frame = defaultdict(list)
    for f, i, al, fx, fy in truth:
        by_frame[f].append((i, al, fx, fy))
    on = off = correct = wrong = 0
    for s in samples:
        gts = by_frame.get(int(s["frame"]), [])
        if not gts:
            continue
        best = min(gts, key=lambda g: (g[2] - s["px"]) ** 2 + (g[3] - s["py"]) ** 2)
        if np.hypot(best[2] - s["px"], best[3] - s["py"]) > tol_px:
            off += 1
            continue
        on += 1
        if s["alliance"] == best[1]:
            correct += 1
        else:
            wrong += 1
    return on, off, correct, wrong


@pytest.mark.parametrize("gains", [(1.0, 1.0, 1.0), (0.75, 0.95, 1.25)], ids=["neutral", "warm-cast"])
def test_wall_and_alliance_regression(tmp_path, gains):
    truth = _broadcast(tmp_path / "wall.mp4", gains=gains)
    result = track_video(
        tmp_path / "wall.mp4",
        tracker_mode="potato",
        frame_stride=3,
        max_frames=50,
        crop_top=CROP_TOP,
        crop_bottom=CROP_BOT,
        use_bev=False,
        openai_key="",
        google_key="",
    )
    samples = keep_top_tracks(stitch_occlusions(result["samples"]), 6)
    samples = balance_alliances(samples)
    assert len(samples) > 60, result["warnings"]

    on, off, correct, wrong = _score(samples, truth)
    # Wall / glass / static field elements must not be reported as robots.
    assert off <= 0.05 * (on + off), f"off-robot samples {off} of {on + off}"
    # Exactly 3 red + 3 blue across the kept tracks.
    per_track = defaultdict(Counter)
    for s in samples:
        per_track[s["track_id"]][s["alliance"]] += 1
    alliances = Counter(c.most_common(1)[0][0] for c in per_track.values())
    assert alliances == {"red": 3, "blue": 3}, dict(alliances)
    # Bumper color, not the mechanism on top, decides the alliance.
    assert correct >= 0.85 * (correct + wrong), f"alliance accuracy {correct}/{correct + wrong}"
    assert result["field_gate"] is not None
    assert any("Field gate" in w for w in result["warnings"])
