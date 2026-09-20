"""Field-aware gating, BEV speed gating, and motion-tracker robustness."""

from __future__ import annotations

import numpy as np

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.geometry import default_source_points, homography_from_corners
from ramscout.identity import keep_top_tracks, track_quality
from ramscout.trackers.color import lowest_band_rule
from ramscout.trackers.ensemble import EnsembleTracker, fuse_local_proposals
from ramscout.trackers.field_gate import FieldGate, distance_to_perimeter, point_in_field
from ramscout.trackers.motion import MotionTracker
from ramscout.trackers.mot import MotTracker
from ramscout.trackers.types import Detection, TrackerContext

FW, FH = 1280, 720
CROP_TOP, CROP_BOT = 0.10, 0.65
Y0 = int(FH * CROP_TOP)
CROP_H = int(FH * CROP_BOT) - Y0


def _gate(**kw) -> FieldGate:
    H = homography_from_corners(default_source_points(FW, FH, CROP_TOP, CROP_BOT))
    return FieldGate(
        H,
        crop_y0=Y0,
        crop_w=FW,
        crop_h=CROP_H,
        frame_h=FH,
        scorebug_bands=[(0.0, 0.14), (0.82, 1.0)],
        dt_s=0.1,
        **kw,
    )


def test_point_in_field_and_perimeter_distance():
    assert point_in_field(10, 10)
    assert not point_in_field(-30, 10)
    assert point_in_field(-10, 10, margin_in=18)
    assert distance_to_perimeter(5, 100) == 5
    assert distance_to_perimeter(FIELD_LENGTH / 2, FIELD_WIDTH / 2) > 100


def test_gate_accepts_mid_field_robot_box():
    gate = _gate()
    det = Detection(1, [600, 180, 650, 240], "motion", 0.6)
    verdict = gate.evaluate(det)
    assert verdict.keep, verdict.reason
    x, y = verdict.info["field_xy"]
    assert 0 <= x <= FIELD_LENGTH and 0 <= y <= FIELD_WIDTH
    w_in, _h_in = verdict.info["footprint_in"]
    assert 10 <= w_in <= 96


def test_gate_rejects_perimeter_wall_segment():
    gate = _gate()
    # Wide, low blob whose foot sits on the near field boundary (0.92 of crop).
    foot_y = 0.92 * CROP_H
    det = Detection(2, [300, foot_y - 18, 620, foot_y], "motion", 0.6)
    verdict = gate.evaluate(det)
    assert not verdict.keep
    assert verdict.reason in {"perimeter_wall", "footprint_huge"}


def test_gate_rejects_box_outside_field_polygon():
    gate = _gate()
    # Foot well below the near boundary → projects outside the field + margin.
    det = Detection(3, [600, CROP_H - 40, 640, CROP_H - 1], "color", 0.6)
    verdict = gate.evaluate(det)
    assert not verdict.keep
    assert verdict.reason == "outside_field"


def test_gate_rejects_implausible_footprints():
    gate = _gate()
    huge = gate.evaluate(Detection(4, [200, 150, 1000, 260], "motion", 0.6))
    assert not huge.keep and huge.reason == "footprint_huge"
    tiny = gate.evaluate(Detection(5, [640, 200, 646, 206], "motion", 0.6))
    assert not tiny.keep and tiny.reason in {"footprint_tiny", "projection"}


def test_gate_rejects_scorebug_overlay_box():
    gate = _gate()
    # Entirely inside the top scorebug band (frame y 72..100 → 0.10..0.139).
    det = Detection(6, [500, 4, 620, 24], "color", 0.7)
    verdict = gate.evaluate(det)
    assert not verdict.keep
    assert verdict.reason == "scorebug"


def test_gate_flags_static_blob_but_not_moving_robot():
    import cv2

    gate = _gate()
    rng = np.random.default_rng(1)
    wall = [40, 100, 400, 130]
    for i in range(12):
        frame = np.full((CROP_H, FW, 3), 50, np.uint8)
        frame = cv2.add(frame, rng.integers(0, 4, frame.shape, dtype=np.uint8))
        cv2.rectangle(frame, (wall[0], wall[1]), (wall[2], wall[3]), (180, 180, 190), -1)
        x = 500 + 12 * i
        cv2.rectangle(frame, (x, 180), (x + 48, 236), (30, 30, 220), -1)
        gate.observe(frame)
    assert gate.frame_active
    robot = Detection(7, [500 + 12 * 11, 180, 548 + 12 * 11, 236], "motion", 0.6)
    v_robot = gate.evaluate(robot)
    assert v_robot.keep and v_robot.info.get("static") is False
    # A static field element away from the robot's path (geometry passes).
    v_wall = gate.evaluate(Detection(8, [200, 150, 260, 200], "motion", 0.6))
    assert v_wall.keep
    assert v_wall.info.get("static") is True
    assert v_wall.penalty > v_robot.penalty


def test_static_flagged_proposal_cannot_spawn_track():
    mot = MotTracker(min_hits=1, require_motion_to_confirm=True)
    frame = np.full((200, 400, 3), 40, np.uint8)
    det = Detection(1, [40, 40, 80, 80], "motion", 0.6, meta={"static": True})
    out = mot.update([det], frame_w=400, frame_h=200, frame_bgr=frame)
    assert out == []
    assert not mot.tracks
    # A moving proposal spawns, then confirms once it has actually moved.
    moving = Detection(2, [40, 40, 80, 80], "motion", 0.6, meta={"static": False})
    assert mot.update([moving], frame_w=400, frame_h=200, frame_bgr=frame) == []
    moved = Detection(2, [52, 40, 92, 80], "motion", 0.6, meta={"static": False})
    out = mot.update([moved], frame_w=400, frame_h=200, frame_bgr=frame)
    assert len(out) == 1


def test_bev_speed_gate_rejects_teleporting_association():
    # 10 inches per pixel: a 30 px hop is 300 in in 0.1 s ⇒ 3000 in/s, impossible.
    proj = lambda b: ((b[0] + b[2]) * 5.0, (b[1] + b[3]) * 5.0)  # noqa: E731
    frame = np.full((500, 1000, 3), 40, np.uint8)

    gated = MotTracker(min_hits=1, dt_s=0.1, field_projector=proj)
    out0 = gated.update([Detection(1, [100, 100, 130, 130], "yolo", 0.8)], frame_w=1000, frame_h=500, frame_bgr=frame)
    tid0 = out0[0].track_id
    out1 = gated.update([Detection(2, [130, 100, 160, 130], "yolo", 0.8)], frame_w=1000, frame_h=500, frame_bgr=frame)
    assert all(d.track_id != tid0 for d in out1), "teleport must not be associated"

    plain = MotTracker(min_hits=1, dt_s=0.1)
    out0 = plain.update([Detection(1, [100, 100, 130, 130], "yolo", 0.8)], frame_w=1000, frame_h=500, frame_bgr=frame)
    tid0 = out0[0].track_id
    out1 = plain.update([Detection(2, [130, 100, 160, 130], "yolo", 0.8)], frame_w=1000, frame_h=500, frame_bgr=frame)
    assert any(d.track_id == tid0 for d in out1), "pixel-only tracker associates the small hop"


def test_bev_tracks_report_field_position_and_speed():
    proj = lambda b: ((b[0] + b[2]) * 0.25, (b[1] + b[3]) * 0.25)  # noqa: E731  0.5 in/px
    frame = np.full((500, 1000, 3), 40, np.uint8)
    mot = MotTracker(min_hits=1, dt_s=0.1, field_projector=proj)
    for i in range(4):
        x = 100 + 10 * i
        out = mot.update([Detection(1, [x, 100, x + 30, 130], "yolo", 0.8)], frame_w=1000, frame_h=500, frame_bgr=frame)
    assert len(out) == 1
    meta = out[0].meta
    assert "field_xy" in meta and "speed_in_s" in meta
    # 10 px/step × 0.5 in/px = 5 in / 0.1 s ≈ 50 in/s (filter-smoothed).
    assert 15.0 <= meta["speed_in_s"] <= 60.0


def test_output_nms_suppresses_bumper_box_inside_robot_box():
    mot = MotTracker(min_hits=1)
    frame = np.full((300, 400, 3), 40, np.uint8)
    dets = [
        Detection(1, [100, 100, 160, 170], "motion", 0.7),
        Detection(2, [105, 150, 155, 168], "color", 0.6, alliance="red"),
    ]
    out = mot.update(dets, frame_w=400, frame_h=300, frame_bgr=frame)
    assert len(out) == 1
    assert out[0].bbox[3] - out[0].bbox[1] > 40


def test_motion_tracker_finds_robots_inside_perimeter_ring():
    import cv2

    tracker = MotionTracker()
    ctx = TrackerContext(frame_w=640, frame_h=360, crop_w=640, crop_h=300)
    base = np.full((300, 640, 3), 40, np.uint8)
    tracker.detect(base, ctx)
    tracker.detect(base, ctx)
    frame = base.copy()
    # Closed bright ring (LED wall) plus a robot-sized blob inside it.
    cv2.rectangle(frame, (10, 10), (630, 290), (200, 200, 200), 8)
    cv2.rectangle(frame, (300, 140), (340, 180), (30, 30, 220), -1)
    dets = tracker.detect(frame, ctx)
    centers = [((d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2) for d in dets]
    assert any(abs(cx - 320) < 15 and abs(cy - 160) < 15 for cx, cy in centers), centers
    # And the ring itself is not reported as a robot.
    assert all((d.bbox[2] - d.bbox[0]) < 200 for d in dets)


def test_color_tracker_finds_bumper_inside_led_wall_ring():
    import cv2

    from ramscout.trackers.color import ColorTracker

    ctx = TrackerContext(frame_w=640, frame_h=360, crop_w=640, crop_h=300)
    frame = np.full((300, 640, 3), (45, 90, 45), np.uint8)
    # Blue LED strip along the whole perimeter + one blue bumper inside.
    cv2.rectangle(frame, (6, 6), (634, 294), (230, 80, 20), 6)
    cv2.rectangle(frame, (300, 160), (348, 178), (230, 80, 20), -1)
    dets = ColorTracker().detect(frame, ctx)
    blue = [d for d in dets if d.alliance == "blue"]
    assert any(abs((d.bbox[0] + d.bbox[2]) / 2 - 324) < 12 for d in blue), [d.bbox for d in dets]
    assert all((d.bbox[2] - d.bbox[0]) < 200 for d in dets)


def test_lowest_band_rule_drops_decoy_above_bumper():
    bumper = ("blue", 400.0, [100.0, 160.0, 150.0, 175.0])
    decoy = ("red", 300.0, [105.0, 110.0, 145.0, 122.0])
    other = ("red", 380.0, [400.0, 160.0, 450.0, 175.0])
    kept = lowest_band_rule([bumper, decoy, other])
    assert bumper in kept and other in kept
    assert decoy not in kept


def test_fuse_local_proposals_takes_alliance_only_from_lower_blob():
    motion = [Detection(1, [100, 100, 160, 180], "motion", 0.5)]
    top_stripe = [Detection(2, [110, 104, 150, 118], "color", 0.6, alliance="red")]
    fused = fuse_local_proposals(motion, top_stripe)
    assert len(fused) == 1 and fused[0].alliance == "unknown"
    bumper = [Detection(3, [105, 160, 155, 178], "color", 0.6, alliance="blue")]
    fused = fuse_local_proposals(motion, bumper)
    assert len(fused) == 1 and fused[0].alliance == "blue"
    assert fused[0].confidence > 0.6


def test_keep_top_tracks_prefers_moving_in_field_tracks():
    samples = []
    for tid in range(6):
        for i in range(20):
            samples.append({"track_id": tid, "t": i * 0.1, "x": 100 + tid * 60 + i * 4.0, "y": 100.0 + tid * 10})
    # Static wall blob with more samples, sitting on the perimeter.
    for i in range(30):
        samples.append({"track_id": 99, "t": i * 0.1, "x": 3.0, "y": 150.0})
    kept = keep_top_tracks(samples, max_tracks=6)
    assert 99 not in {s["track_id"] for s in kept}
    assert track_quality([s for s in samples if s["track_id"] == 0]) > track_quality(
        [s for s in samples if s["track_id"] == 99]
    )


def test_ensemble_configure_geometry_sets_bev_mot():
    ens = EnsembleTracker([MotionTracker()], cascade=True)
    gate = _gate()
    ens.configure_geometry(gate, dt_s=0.1)
    assert ens.mot.field_projector is not None
    assert ens.mot.require_motion_to_confirm is True
    ens.configure_geometry(None, dt_s=0.1)
    assert ens.mot.field_projector is None
    assert ens.mot.require_motion_to_confirm is False
