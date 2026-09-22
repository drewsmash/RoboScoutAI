"""Regression fixtures for tracking redesign invariants.

These encode the confirmed bugs from docs/TRACKING_REDESIGN.md so they
cannot silently return.
"""

from __future__ import annotations

import numpy as np

from ramscout.crop import (
    OverviewCrop,
    bbox_crop_to_video,
    crop_to_video,
    propose_top_overview,
    slice_overview,
    video_to_crop,
)
from ramscout.identity import assign_by_start
from ramscout.identity_book import IdentityBook, alliance_css
from ramscout.onnx_detector import detector_status, load_meta


def test_propose_top_overview_from_layout_panes():
    panes = [
        {"role": "overview", "crop_left": 0.0, "crop_top": 0.02, "crop_right": 1.0, "crop_bottom": 0.48, "confidence": 0.9},
        {"role": "blue_side", "crop_left": 0.0, "crop_top": 0.52, "crop_right": 0.5, "crop_bottom": 1.0},
        {"role": "red_side", "crop_left": 0.5, "crop_top": 0.52, "crop_right": 1.0, "crop_bottom": 1.0},
    ]
    crop = propose_top_overview(panes)
    # Must cover the full detected overview pane (never shorten it).
    assert abs(crop.y1 - 0.48) < 1e-6
    assert crop.y0 <= 0.02 + 1e-6
    assert crop.source == "auto"
    assert crop.y0 < crop.y1
    assert crop.y1 - crop.y0 >= 0.28


def test_coordinate_roundtrip_video_crop():
    crop = OverviewCrop(0.1, 0.05, 0.9, 0.5)
    fw, fh = 1280, 720
    vx, vy = 640.0, 200.0
    cx, cy = video_to_crop(vx, vy, crop, fw, fh)
    back = crop_to_video(cx, cy, crop, fw, fh)
    assert abs(back[0] - vx) < 1.0 and abs(back[1] - vy) < 1.0
    box = bbox_crop_to_video([10, 20, 40, 60], crop, fw, fh)
    assert box[0] > 10 and box[1] > 20


def test_slice_overview_shape():
    frame = np.zeros((400, 800, 3), dtype=np.uint8)
    crop = OverviewCrop(0.0, 0.0, 1.0, 0.5)
    sliced, coords = slice_overview(frame, crop)
    assert sliced.shape[0] == 200
    assert sliced.shape[1] == 800
    assert coords.video_w == 800 and coords.crop_h == 200


def test_unknown_alliance_not_inferred_from_field_half():
    tracks = [
        {"track_id": 1, "t": 0.1, "x": 100.0, "y": 50.0, "alliance": "unknown"},
        {"track_id": 2, "t": 0.1, "x": 500.0, "y": 50.0, "alliance": "unknown"},
        {"track_id": 3, "t": 0.1, "x": 120.0, "y": 100.0, "alliance": "blue"},
    ]
    mapping = assign_by_start(tracks, blue_teams=["195"], red_teams=["230"])
    assert tracks[0]["alliance"] == "unknown"
    assert tracks[1]["alliance"] == "unknown"
    # Unknowns must not be soft-filled into roster slots from field half.
    assert 1 not in mapping and 2 not in mapping
    assert mapping.get(3) == "195"
    assert alliance_css("unknown") == "unknown"
    assert alliance_css("purple") == "unknown"


def test_identity_book_does_not_recolor_confirmed():
    book = IdentityBook()
    ident = book.bind(10, t=1.0)
    for _ in range(12):
        book.observe_alliance(ident.identity_id, "blue", 0.8)
        book.bind(10, t=1.0)
    ident = book.identity_for_tracklet(10)
    assert ident is not None
    assert ident.alliance == "blue"
    book.observe_alliance(ident.identity_id, "red", 0.9)
    ident = book.identity_for_tracklet(10)
    assert ident.alliance == "blue"
    assert "alliance_conflict" in ident.flags


def test_alliance_css_never_coerces_unknown_to_blue():
    assert alliance_css("") == "unknown"
    assert alliance_css(None) == "unknown"  # type: ignore[arg-type]
    assert alliance_css("blue") == "blue"
    assert alliance_css("red") == "red"


def test_onnx_detector_does_not_claim_frc_ready_without_weights():
    status = detector_status()
    assert status["frc_ready"] is False
    meta = load_meta(None)
    assert meta.frc_ready is False
    assert "verified" in meta.notes.lower() or "no verified" in meta.notes.lower()


def test_crop_lock_interval_override():
    from ramscout.crop import CropLockState

    lock = CropLockState(
        active=OverviewCrop(0.0, 0.0, 1.0, 0.5, locked=True),
        intervals=[OverviewCrop(0.0, 0.1, 1.0, 0.4, t0=30.0, t1=45.0, source="interval")],
    )
    assert lock.crop_at(10.0).y0 == 0.0
    assert abs(lock.crop_at(35.0).y0 - 0.1) < 1e-6
    assert lock.mode == "top_overview_only"
