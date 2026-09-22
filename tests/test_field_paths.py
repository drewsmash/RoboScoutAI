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
