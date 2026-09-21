"""Tests for the 0.6.1 tracking / 3D / decomposition overhaul.

Covers: auto-benchmark selection, ByteTrack two-stage association and the
parked-robot rule, field-line homography refinement, RTS smoothing, bumper
OCR voting, temporal layout decomposition and the layout-driven segments in
track_video.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.trackers.mot import MotTracker, _proposal_score
from ramscout.trackers.types import Detection


# ----------------------------------------------------------------- selection
def _fake_result(samples, *, fps=30.0, window=(3.0, 15.0)):
    return {
        "samples": samples,
        "fps": fps,
        "window": list(window),
        "homography": np.eye(3).tolist(),
        "src_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
        "source_hits": {"motion": len(samples)},
        "strategies": ["motion"],
        "warnings": [],
    }


def _synthetic_samples(n_tracks: int, *, dt=0.1, t0=3.0, t1=15.0, speed_in_s=60.0, jitter=0.0, seed=0):
    rng = np.random.RandomState(seed)
    out = []
    for k in range(n_tracks):
        alliance = "red" if k % 2 == 0 else "blue"
        x0 = 60.0 + 80.0 * (k // 2)
        y0 = 40.0 + 60.0 * (k % 2)
        t = t0
        while t < t1:
            frac = (t - t0) / (t1 - t0)
            x = x0 + speed_in_s * (t - t0) * 0.3 + rng.randn() * jitter
            y = y0 + 40.0 * np.sin(frac * 3.0) + rng.randn() * jitter
            out.append(
                {
                    "t": round(t, 3),
                    "frame": int(t / dt),
                    "track_id": 1000 + k,
                    "x": float(np.clip(x, 5, FIELD_LENGTH - 5)),
                    "y": float(np.clip(y, 5, FIELD_WIDTH - 5)),
                    "alliance": alliance,
                    "speed_in_s": speed_in_s,
                }
            )
            t += dt
    return out


def test_score_samples_prefers_six_balanced_persistent_robots():
    from ramscout.trackers.selection import score_samples

    good = score_samples(_synthetic_samples(6), window_s=12.0, dt_s=0.1)
    few = score_samples(_synthetic_samples(2), window_s=12.0, dt_s=0.1)
    # Fragmented: same robots but IDs change every 2 s.
    frag = []
    for s in _synthetic_samples(6):
        s = dict(s)
        s["track_id"] = s["track_id"] * 10 + int(s["t"] // 2)
        frag.append(s)
    fragmented = score_samples(frag, window_s=12.0, dt_s=0.1)
    empty = score_samples([], window_s=12.0, dt_s=0.1)
    assert good["total"] > few["total"] > 0
    assert good["total"] > fragmented["total"]
    assert good["balance"] == 1.0
    assert empty["total"] == 0.0


def test_select_tracker_mode_picks_best_candidate_and_records_scores(tmp_path):
    from ramscout.trackers.selection import select_tracker_mode

    calls: list[str] = []

    def runner(video_path, **kwargs):
        mode = kwargs["tracker_mode"]
        calls.append(mode)
        assert kwargs["smooth"] is False and kwargs["bumper_ocr"] is False
        n = {"motion": 3, "color": 6, "potato": 4, "hybrid": 5}[mode]
        return _fake_result(_synthetic_samples(n))

    sel = select_tracker_mode(
        Path("dummy.mp4"),
        candidates=["motion", "color", "potato", "hybrid"],
        runner=runner,
        frame_stride=3,
        benchmark_s=12.0,
    )
    assert sel["chosen"] == "color"
    assert set(sel["scores"]) == {"motion", "color", "potato", "hybrid"}
    assert sel["ranking"][0] == "color"
    assert calls == ["motion", "color", "potato", "hybrid"]
    assert sel["homography"] is not None  # shared with later candidates


def test_select_tracker_mode_survives_candidate_errors():
    from ramscout.trackers.selection import select_tracker_mode

    def runner(video_path, **kwargs):
        if kwargs["tracker_mode"] == "color":
            raise RuntimeError("boom")
        return _fake_result(_synthetic_samples(6))

    sel = select_tracker_mode(Path("x.mp4"), candidates=["color", "hybrid"], runner=runner)
    assert sel["chosen"] == "hybrid"
    assert "error" in sel["scores"]["color"]


def test_candidate_modes_only_includes_keyed_cloud_and_installed_yolo(monkeypatch):
    from ramscout.trackers import selection

    monkeypatch.setattr(
        selection,
        "list_strategies",
        lambda model_path=None: [
            {"name": n, "available": n in {"motion", "color", "optical_flow"}} for n in ("motion", "color", "optical_flow", "yolo", "openai", "gemini")
        ],
    )
    modes = selection.candidate_modes()
    assert "yolo" not in modes and "gemini" not in modes and "openai" not in modes
    assert modes[0] == "motion" and "hybrid" in modes
    keyed = selection.candidate_modes(google_key="k")
    assert "gemini" in keyed


def test_auto_mode_is_default_and_benchmarks():
    from ramscout.trackers.ensemble import TRACKER_MODES

    assert TRACKER_MODES["auto"].get("benchmark") is True
    assert "benchmark" in TRACKER_MODES["auto"]["label"].lower()


# ----------------------------------------------------------------- ByteTrack
def _frame():
    return np.full((200, 400, 3), 40, dtype=np.uint8)


def test_bytetrack_low_confidence_never_spawns_but_extends():
    mot = MotTracker(max_age=10, min_hits=1, max_tracks=6)
    fr = _frame()
    # Low-score proposal alone: no track.
    out = mot.update([Detection(1, [20, 40, 50, 70], "motion", 0.2)], frame_w=400, frame_h=200, frame_bgr=fr)
    assert out == [] and not mot.tracks
    # High-score proposal spawns; then a low-score box extends it.
    out = mot.update([Detection(1, [20, 40, 50, 70], "motion", 0.6)], frame_w=400, frame_h=200, frame_bgr=fr)
    assert len(out) == 1
    tid = out[0].track_id
    out = mot.update([Detection(9, [26, 41, 56, 71], "motion", 0.2)], frame_w=400, frame_h=200, frame_bgr=fr)
    assert [d.track_id for d in out] == [tid]
    assert mot.stage_hits["low"] == 1 and mot.stage_hits["high"] == 0


def test_proposal_score_uses_raw_confidence_before_gate_penalty():
    d = Detection(1, [0, 0, 10, 10], "color", 0.2, meta={"raw_confidence": 0.55})
    assert _proposal_score(d) == pytest.approx(0.55)
    assert _proposal_score(Detection(1, [0, 0, 10, 10], "yolo", 0.1)) == 1.0


def test_ocsort_reupdate_trusts_observed_gap_velocity():
    mot = MotTracker(max_age=20, min_hits=1, max_tracks=6, bytetrack=False)
    fr = _frame()
    mot.update([Detection(1, [20, 40, 50, 70], "motion", 0.7)], frame_w=400, frame_h=200, frame_bgr=fr)
    mot.update([Detection(1, [30, 40, 60, 70], "motion", 0.7)], frame_w=400, frame_h=200, frame_bgr=fr)
    tid = next(iter(mot.tracks))
    for _ in range(4):  # occlusion
        mot.update([], frame_w=400, frame_h=200, frame_bgr=fr)
    out = mot.update([Detection(1, [90, 40, 120, 70], "motion", 0.7)], frame_w=400, frame_h=200, frame_bgr=fr)
    assert [d.track_id for d in out] == [tid]
    assert mot.stage_hits["recovered"] == 1
    vx = mot.tracks[tid].kalman.mean[4]
    # Observed displacement over the gap: (105-45)/5 = 12 px/step; 0.8 weight → > 8.
    assert vx > 8.0


def test_parked_robot_confirms_without_motion_but_wall_strip_does_not():
    mot = MotTracker(max_age=10, min_hits=3, max_tracks=6)
    mot.require_motion_to_confirm = True
    fr = _frame()
    robot_meta = {"footprint_in": 32.0, "perimeter": False}
    wall_meta = {"footprint_in": 90.0, "perimeter": True}
    for _ in range(mot.parked_confirm_hits + 1):
        out = mot.update(
            [
                Detection(1, [20, 40, 50, 70], "color", 0.6, alliance="red", meta=dict(robot_meta)),
                Detection(2, [300, 40, 380, 60], "color", 0.6, alliance="blue", meta=dict(wall_meta)),
            ],
            frame_w=400,
            frame_h=200,
            frame_bgr=fr,
        )
    boxes = [d.bbox[0] for d in out]
    assert 20.0 in [round(b) for b in boxes]
    assert 300.0 not in [round(b) for b in boxes]


def test_in_place_wiggler_is_pruned():
    """A scale plate tilting in place accumulates path but no excursion."""
    mot = MotTracker(max_age=10, min_hits=1, max_tracks=6, static_after_hits=6)
    mot.field_projector = lambda bbox: ((bbox[0] + bbox[2]) * 0.5, bbox[3])
    fr = _frame()
    alive = 0
    for i in range(40):
        dx = 6.0 if i % 2 else -6.0
        out = mot.update([Detection(1, [100 + dx, 50, 130 + dx, 80], "motion", 0.7)], frame_w=400, frame_h=200, frame_bgr=fr)
        alive = len(out)
    assert alive == 0


# ---------------------------------------------------------------- fieldlines
def _synthetic_field(w=640, h=360):
    """Grey carpet trapezoid on a busy 'crowd' background with red/blue tape."""
    import cv2

    rng = np.random.RandomState(3)
    img = rng.randint(0, 255, size=(h, w, 3), dtype=np.uint8)  # busy crowd
    quad = np.array([[150, 120], [490, 120], [600, 320], [40, 320]], dtype=np.int32)
    cv2.fillConvexPoly(img, quad, (95, 96, 98))
    # Add a little carpet noise so the model is not degenerate.
    noise = rng.randint(-6, 7, size=(h, w, 1)).astype(np.int16)
    carpet = cv2.fillConvexPoly(np.zeros((h, w), np.uint8), quad, 255) > 0
    img = img.astype(np.int16)
    img[carpet] = np.clip(img[carpet] + noise[carpet], 0, 255)
    img = img.astype(np.uint8)
    # Alliance tape: red thin line on the left third, blue on the right third.
    cv2.line(img, (150, 200), (200, 300), (40, 40, 230), 3)
    cv2.line(img, (480, 200), (440, 300), (230, 60, 40), 3)
    # Driver-station panels outside the carpet.
    cv2.rectangle(img, (100, 60), (140, 118), (40, 40, 230), -1)
    cv2.rectangle(img, (500, 60), (540, 118), (230, 60, 40), -1)
    return img, quad


def test_detect_field_quad_recovers_carpet_corners():
    from ramscout.fieldlines import detect_field_quad

    img, quad = _synthetic_field()
    fq = detect_field_quad([img, img])
    assert fq is not None
    assert fq.confidence >= 0.5
    got = np.array(fq.corners)
    exp = quad.astype(np.float64)
    assert np.all(np.abs(got - exp) < 14), (got, exp)


def test_alliance_orientation_reads_tape_and_wall_panels():
    from ramscout.fieldlines import alliance_orientation, detect_field_quad, orient_corners

    img, _quad = _synthetic_field()
    fq = detect_field_quad([img])
    cue = alliance_orientation([img], fq)
    assert cue.blue_left is False and cue.confidence > 0.2
    mirrored = orient_corners([[0, 0], [10, 0], [10, 5], [0, 5]], cue.blue_left)
    assert mirrored[0] == [10.0, 0.0] and mirrored[1] == [0.0, 0.0]
    same = orient_corners([[0, 0], [10, 0], [10, 5], [0, 5]], True)
    assert same[0] == [0.0, 0.0]


def test_calibrate_bev_uses_field_quad_and_mirrors(monkeypatch):
    from ramscout import bev as bev_mod
    from ramscout.depth import DepthResult

    img, quad = _synthetic_field()
    depth = np.tile(np.linspace(1, 0, img.shape[0], dtype=np.float32)[:, None], (1, img.shape[1]))
    monkeypatch.setattr(
        bev_mod,
        "estimate_depth",
        lambda crop, prefer_neural=True: DepthResult(depth=depth[: crop.shape[0], : crop.shape[1]], pitch_deg=40.0, tilt_strength=0.6, source="classical", detail="mock"),
    )
    cal, _ = bev_mod.calibrate_bev(img, crop_top=0.0, crop_bottom=1.0, field_lines=True)
    assert cal.method.startswith("field_quad")
    assert cal.orientation and cal.orientation["blue_left"] is False
    # Mirrored: first corner (blue wall, far) is the top-RIGHT carpet corner.
    assert abs(cal.src_points[0][0] - quad[1][0]) < 20
    off, _ = bev_mod.calibrate_bev(img, crop_top=0.0, crop_bottom=1.0, field_lines=False)
    assert off.method == "depth_trapezoid" and off.field_quad is None


# ----------------------------------------------------------------- smoothing
def test_rts_smoother_reduces_jitter_and_keeps_raw():
    from ramscout.smoothing import smooth_samples

    noisy = _synthetic_samples(1, jitter=6.0, seed=7)
    clean = _synthetic_samples(1, jitter=0.0, seed=7)
    out = smooth_samples(noisy, dt_hint=0.1)
    err_raw = np.mean([abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) for a, b in zip(noisy, clean)])
    err_smooth = np.mean([abs(a["x"] - b["x"]) + abs(a["y"] - b["y"]) for a, b in zip(out, clean)])
    assert err_smooth < err_raw * 0.7
    assert all("x_raw" in s and s.get("smoothed") for s in out)
    assert all(s["speed_in_s"] >= 0 for s in out)


def test_smoother_splits_runs_at_long_gaps():
    from ramscout.smoothing import smooth_samples

    a = _synthetic_samples(1, t0=0.0, t1=2.0)
    b = _synthetic_samples(1, t0=10.0, t1=12.0)
    for s in b:
        s["x"] += 300.0
    out = smooth_samples(a + b, dt_hint=0.1, max_gap_s=1.0)
    # The second run is not dragged toward the first.
    assert out[len(a)]["x"] > out[len(a) - 1]["x"] + 200


# ----------------------------------------------------------------------- OCR
def test_bumper_ocr_votes_and_assigns_each_team_once():
    from ramscout.ocr import BumperOCR, match_team

    reads = iter(["4183", "41B3", "4183", "2046", "2046", "2046", "4183", "4183"])
    ocr = BumperOCR(["4183", "2046", "3250"], reader=lambda img: next(reads, ""))
    assert ocr.available and ocr.backend == "custom"
    crop = np.full((40, 80, 3), 120, dtype=np.uint8)
    for _ in range(3):
        ocr.observe(1, crop)
    for _ in range(3):
        ocr.observe(2, crop)
    for _ in range(2):
        ocr.observe(3, crop)  # a second track claiming 4183 with fewer votes
    assigned = ocr.assignments()
    assert assigned[1] == "4183" and assigned[2] == "2046"
    assert 3 not in assigned  # 4183 already taken by the stronger tally
    team, score = match_team("S25O", ["3250", "1234"])
    assert team == "3250" and score > 0


# -------------------------------------------------------------------- layout
def _broadcast_frames(n=8, w=640, h=360, seed=0):
    """Stacked-top broadcast: field camera on top, close-up below, scorebug."""
    import cv2

    rng = np.random.RandomState(seed)
    frames = []
    split = int(h * 0.55)
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        # Top: grey carpet with a crowd band above it, robots moving.
        img[:split] = (90, 92, 95)
        img[: int(split * 0.3)] = rng.randint(0, 255, size=(int(split * 0.3), w, 3))
        for k in range(4):
            x = 60 + 120 * k + 6 * i
            cv2.rectangle(img, (x, int(split * 0.6)), (x + 30, int(split * 0.6) + 24), (0, 0, 220) if k % 2 else (220, 80, 0), -1)
        # Bottom: close-up pane (saturated, big robot, different colour cast).
        img[split:] = (60, 40, 30)
        cv2.rectangle(img, (200 + 8 * i, split + 40), (420 + 8 * i, h - 40), (30, 30, 200), -1)
        # Scorebug: static high-contrast bar at the very bottom.
        cv2.rectangle(img, (int(w * 0.3), h - 40), (int(w * 0.7), h - 8), (255, 255, 255), -1)
        cv2.putText(img, "86  1:56  55", (int(w * 0.32), h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Seam line.
        img[split - 1 : split + 1] = 0
        frames.append(img)
    return frames, split / h


def test_analyze_frames_finds_stacked_layout_and_overview_on_top():
    from ramscout.layout import analyze_frames

    frames, split = _broadcast_frames()
    layout = analyze_frames(frames)
    assert layout.mode in {"stacked_top", "stacked_sides", "grid"} or len(layout.panes) >= 2
    ov = next((p for p in layout.panes if p.role == "overview"), None)
    assert ov is not None, layout.detail
    # The crowd band above the carpet may be split off as its own pane; the
    # overview must still end at the composition seam.
    assert ov.box.y0 < 0.25 and abs(ov.box.y1 - split) < 0.08
    assert layout.row_cuts and any(abs(c - split) < 0.05 for c in layout.row_cuts)
    assert layout.scorebug is not None


def test_layout_from_decomposition_keeps_confidence_and_timeline():
    from ramscout.layout import LayoutSegment, LayoutTimeline, analyze_frames
    from ramscout.multicam import layout_from_decomposition

    frames, _ = _broadcast_frames()
    fl = analyze_frames(frames)
    tl = LayoutTimeline(segments=[LayoutSegment(t0=0.0, t1=60.0, layout=fl)], duration=60.0, fps=30.0, frame_size=(640, 360))
    cam = layout_from_decomposition(fl, timeline=tl)
    assert cam.method == "decomposition"
    assert cam.overview_pane() is not None
    assert cam.timeline and cam.timeline[0]["t0"] == 0.0
    assert all("confidence" in p.as_dict() for p in cam.panes)


def test_track_video_segments_follow_layout_timeline_with_gaps():
    from ramscout.detect import _segments_from_layout

    class L:
        timeline = [
            {"t0": 0.0, "t1": 30.0, "overview": [0.0, 0.0, 1.0, 0.55]},
            {"t0": 30.0, "t1": 40.0, "overview": None},
            {"t0": 40.0, "t1": 90.0, "overview": [0.0, 0.0, 1.0, 0.56]},
        ]

    segs = _segments_from_layout(L(), 90.0, (0.0, 0.1, 1.0, 0.65))
    assert len(segs) == 3
    assert segs[1].box is None
    # Same geometry within 3 % → same pane index (tracks resume, homography reused).
    assert segs[0].index == segs[2].index
    plain = _segments_from_layout(None, 120.0, (0.0, 0.1, 1.0, 0.65))
    assert len(plain) == 1 and plain[0].box == (0.0, 0.1, 1.0, 0.65)


def test_field_mask_for_crop_covers_field_not_crowd():
    from ramscout.detect import field_mask_for_crop
    from ramscout.geometry import homography_from_corners

    # Field occupies the lower 60 % of a 1280x400 pane, trapezoid.
    pts = [[200, 160], [1080, 160], [1240, 380], [40, 380]]
    H = homography_from_corners(pts)
    mask = field_mask_for_crop(H, 0, 0, 1280, 400)
    assert mask is not None
    assert mask[300, 640] == 255  # centre of carpet
    assert mask[20, 640] == 0  # crowd band
    assert mask[390, 5] == 0  # off the near-left corner


# -------------------------------------------------------------- verify-only
def test_coco_yolo_only_verifies_local_proposals():
    from ramscout.trackers.ensemble import verify_proposals

    local = [Detection(1, [10, 10, 50, 50], "motion", 0.5), Detection(2, [200, 10, 240, 50], "motion", 0.5)]
    yolo = [Detection(7, [12, 12, 52, 52], "yolo", 0.9), Detection(8, [400, 300, 440, 340], "yolo", 0.9)]
    out = verify_proposals(local, yolo)
    assert len(out) == 2  # the unmatched YOLO box did not become a proposal
    assert out[0].confidence > 0.5 and out[0].meta.get("verified_by") == "yolo"
    assert out[1].confidence == 0.5


def test_trackers_endpoint_reports_mode_availability_and_depth():
    from fastapi.testclient import TestClient

    from app import app

    client = TestClient(app)
    data = client.get("/api/trackers").json()
    modes = {m["id"]: m for m in data["modes"]}
    assert modes["auto"]["default"] is True and modes["auto"]["benchmark"] is True
    assert "status" in modes["gemini"] and "missing" in modes["gemini"]
    assert "backends" in data["depth"]
