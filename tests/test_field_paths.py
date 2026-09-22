"""Regression: invalid projections and path gaps must not create corner spiders."""

from __future__ import annotations

import math

from ramscout.crop import MIN_OVERVIEW_HEIGHT, OverviewCrop, propose_top_overview
from ramscout.events import Pose, build_cards, detect_events
from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.geometry import (
    annotate_field_sample,
    default_source_points,
    field_point_status,
    is_field_sample_drawable,
    reproject_samples,
    sanitize_samples_field_coords,
)


def test_reproject_does_not_clamp_outside_to_field_edge():
    src = default_source_points(1920, 1080)
    # Pixel far outside the calibrated quad → wild field inches.
    samples = [{"px": 5.0, "py": 5.0, "x": 0.0, "y": 0.0}]
    out = reproject_samples(samples, src.tolist())
    assert out[0]["field_valid"] is False
    assert out[0]["x"] is None and out[0]["y"] is None
    # Must not look like a corner contact.
    assert out[0].get("x_raw") is not None
    assert abs(float(out[0]["x_raw"])) > 50 or abs(float(out[0]["y_raw"])) > 50


def test_reproject_keeps_true_corner_near_origin_without_hard_clip_flag():
    src = default_source_points(1920, 1080)
    samples = [{"px": float(src[0][0]), "py": float(src[0][1])}]
    out = reproject_samples(samples, src.tolist())
    assert out[0]["field_valid"] is True
    assert abs(out[0]["x"]) < 5
    assert abs(out[0]["y"]) < 5


def test_annotate_rejects_hard_outside_and_keeps_soft_interior():
    bad = annotate_field_sample({}, -80.0, 40.0)
    assert bad["field_valid"] is False and bad["x"] is None
    ok = annotate_field_sample({}, 100.0, 50.0)
    assert ok["field_valid"] is True and ok["x"] == 100.0
    soft = annotate_field_sample({}, -5.0, 10.0)
    assert soft["field_valid"] is True
    assert soft["x"] == -5.0  # not clamped to 0


def test_sanitize_marks_clamped_edge_junk_when_flagged():
    samples = [
        {"x": 0.0, "y": 0.0, "field_valid": True},
        {"x": None, "y": None},
        {"x": FIELD_LENGTH / 2, "y": FIELD_WIDTH / 2},
    ]
    sanitize_samples_field_coords(samples)
    assert samples[1]["field_valid"] is False
    assert samples[2]["field_valid"] is True
    assert is_field_sample_drawable(samples[2])
    assert not is_field_sample_drawable(samples[1])


def test_path_length_skips_teleport_gaps():
    poses = [
        Pose(t=0.0, x=100.0, y=100.0, team="59", alliance="blue", track_id=1),
        Pose(t=0.2, x=110.0, y=100.0, team="59", alliance="blue", track_id=1),
        # Camera cut + invalid corner dump then resume elsewhere.
        Pose(t=5.0, x=500.0, y=200.0, team="59", alliance="blue", track_id=1),
        Pose(t=5.2, x=510.0, y=200.0, team="59", alliance="blue", track_id=1),
    ]
    cards = build_cards(poses, detect_events(poses))
    assert len(cards) == 1
    # Only the two short segments (~10 in each) count — not the ~400 in teleport.
    assert cards[0].path_length_in < 40
    assert cards[0].path_length_in > 15


def test_path_length_skips_short_but_huge_jumps():
    poses = [
        Pose(t=0.0, x=100.0, y=100.0, team="59", alliance="blue", track_id=1),
        Pose(t=0.2, x=400.0, y=250.0, team="59", alliance="blue", track_id=1),  # 300in in 0.2s
        Pose(t=0.4, x=410.0, y=250.0, team="59", alliance="blue", track_id=1),
    ]
    cards = build_cards(poses, detect_events(poses))
    assert cards[0].path_length_in < 20


def test_crop_covers_overview_height_and_rejects_amputation():
    panes = [
        {
            "role": "overview",
            "crop_left": 0.0,
            "crop_top": 0.0,
            "crop_right": 1.0,
            "crop_bottom": 0.56,
            "confidence": 0.9,
        }
    ]
    crop = propose_top_overview(panes)
    assert crop.y1 >= 0.56 - 1e-6
    assert crop.y1 - crop.y0 >= MIN_OVERVIEW_HEIGHT - 1e-6

    short = OverviewCrop(0.0, 0.0, 1.0, 0.15).clamp()
    assert short.y1 - short.y0 >= 0.28 - 1e-6

    fallback = propose_top_overview(None, fallback_bottom=0.58)
    assert fallback.y1 >= 0.58 - 1e-6


def test_field_point_status_degenerate():
    assert field_point_status(float("nan"), 10.0)[0] is False
    assert field_point_status(1e9, 1e9)[0] is False
    assert field_point_status(FIELD_LENGTH / 2, FIELD_WIDTH / 2)[0] is True


def test_drawable_breaks_on_gap_flag():
    assert not is_field_sample_drawable({"x": 10, "y": 10, "gap": True})
    assert not is_field_sample_drawable({"x": 10, "y": 10, "camera_cut": True})
    assert is_field_sample_drawable({"x": 10, "y": 10, "field_valid": True})


def test_confident_layout_is_not_cut_by_form_crop():
    """Form crop numbers must not shrink a detected widescreen overview."""
    from ramscout.multicam import CameraLayout, CameraPane, apply_layout

    layout = CameraLayout(
        mode="stacked_top",
        crop_top=0.0,
        crop_bottom=0.685,
        confidence=0.98,
        detail="overview to 0.685",
        split_y=0.685,
        panes=[
            CameraPane(role="overview", crop_top=0.0, crop_bottom=0.685),
            CameraPane(role="overview_alt", crop_top=0.685, crop_bottom=1.0),
        ],
    )
    top, bottom, chosen = apply_layout(
        layout, user_crop_top=0.02, user_crop_bottom=0.52, auto=True, locked=False
    )
    assert chosen.mode == "stacked_top"
    assert bottom >= 0.68
    assert top <= 0.01

    top_l, bottom_l, locked = apply_layout(
        layout, user_crop_top=0.02, user_crop_bottom=0.52, auto=True, locked=True
    )
    assert locked.mode == "manual"
    assert abs(bottom_l - 0.52) < 1e-6
    assert abs(top_l - 0.02) < 1e-6


def test_perimeter_track_loses_to_shorter_interior_path():
    from ramscout.identity import keep_top_tracks

    samples = []
    # Long jitter on the left wall.
    for i in range(40):
        samples.append({"track_id": 1, "t": i * 0.2, "x": 4.0 + (i % 3), "y": 140.0, "field_valid": True})
    # Shorter path through the carpet.
    for i in range(12):
        samples.append({"track_id": 2, "t": i * 0.2, "x": 180.0 + i * 12.0, "y": 160.0, "field_valid": True})
    kept = {s["track_id"] for s in keep_top_tracks(samples, max_tracks=1)}
    assert kept == {2}


def test_skewed_carpet_quad_is_rejected():
    from ramscout.fieldlines import quad_geometry_ok

    # Far edge tilted and a corner extrapolated past the pane (1920-wide).
    bad = [[336.0, 332.0], [1615.0, 503.0], [2304.0, 662.0], [108.0, 639.0]]
    assert quad_geometry_ok(bad, 1920, 740) is False
    good = [[289.0, 387.0], [1626.0, 403.0], [1861.0, 687.0], [82.0, 666.0]]
    assert quad_geometry_ok(good, 1920, 740) is True


def test_flapping_layout_keeps_one_overview():
    """A confident stacked overview wins over sideline/grid misreads."""
    from ramscout.detect import _segments_from_layout

    class Layout:
        mode = "stacked_top"
        confidence = 0.98
        crop_top = 0.0
        crop_bottom = 0.685
        crop_left = 0.0
        crop_right = 1.0
        timeline = [
            {"t0": 0.0, "t1": 8.0, "overview": None},
            {"t0": 8.0, "t1": 26.0, "overview": [0.0, 0.0, 1.0, 0.685]},
            {"t0": 26.0, "t1": 48.0, "overview": [0.0, 0.189, 1.0, 0.685]},
            {"t0": 48.0, "t1": 90.0, "overview": [0.0, 0.0, 1.0, 1.0]},
            {"t0": 90.0, "t1": 110.0, "overview": [0.5, 0.189, 1.0, 0.685]},
            {"t0": 110.0, "t1": 120.0, "overview": None},
        ]

    segs = _segments_from_layout(Layout(), 120.0, (0.0, 0.1, 1.0, 0.65))
    assert segs[0].box is None
    # The match itself is one pane, not a new tracker every time the
    # decomposition flickers.
    match = [s for s in segs if s.box is not None and s.t0 < 110]
    assert len(match) == 1
    assert match[0].box[1] <= 0.02
    assert abs(match[0].box[3] - 0.685) < 0.02
    assert match[0].t0 == 8.0 and match[0].t1 == 110.0
    assert segs[-1].box is None
