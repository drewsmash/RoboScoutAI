"""Tests for multi-view correlation and per-pane detection helpers."""

from __future__ import annotations

import numpy as np

from ramscout.correlate import attach_teams_to_cues, correlate_views
from ramscout.multicam import CameraLayout, CameraPane
from ramscout.multiview import _thin_pane_detections, detect_all_panes, merge_side_cues_into_events


def test_correlate_views_links_side_to_overview_by_alliance_and_time():
    overview = [
        {"t": 10.0, "track_id": 1, "alliance": "blue", "team": "195", "x_norm": 0.3, "y_norm": 0.5, "confidence": 0.8},
        {"t": 10.1, "track_id": 2, "alliance": "red", "team": "230", "x_norm": 0.7, "y_norm": 0.5, "confidence": 0.8},
    ]
    panes = [
        {"t": 10.05, "role": "blue_side", "alliance": "blue", "x_norm": 0.4, "y_norm": 0.5, "confidence": 0.6},
        {"t": 10.2, "role": "red_side", "alliance": "red", "x_norm": 0.6, "y_norm": 0.55, "confidence": 0.7},
    ]
    result = correlate_views(overview, panes)
    assert result["stats"]["matched"] == 2
    by_role = {d["role"]: d for d in result["pane_detections"]}
    assert by_role["blue_side"]["track_id"] == 1
    assert by_role["blue_side"]["team"] == "195"
    assert by_role["red_side"]["team"] == "230"
    assert any(c.get("team") == "195" for c in result["side_cues"])


def test_attach_teams_to_cues_fills_from_assignments():
    cues = [{"t": 1.0, "track_id": 7, "alliance": "blue", "kind": "hub_activity", "confidence": 0.5}]
    out = attach_teams_to_cues(cues, {"7": "195"}, [])
    assert out[0]["team"] == "195"


def test_merge_side_cues_prefers_correlated_team():
    events = []
    cues = [
        {
            "t": 12.0,
            "alliance": "blue",
            "kind": "hub_activity",
            "confidence": 0.7,
            "team": "195",
            "track_id": 3,
            "detail": "correlated",
            "source": "correlated_pane",
        }
    ]
    out = merge_side_cues_into_events(events, cues, cards=[{"team": "999", "alliance": "blue"}])
    assert len(out) == 1
    assert out[0]["team"] == "195"
    assert out[0]["track_id"] == 3


def test_thin_pane_detections_keeps_strongest_per_bin():
    dets = [
        {"t": 1.0, "role": "blue_side", "alliance": "blue", "confidence": 0.4},
        {"t": 1.2, "role": "blue_side", "alliance": "blue", "confidence": 0.9},
        {"t": 5.0, "role": "blue_side", "alliance": "blue", "confidence": 0.5},
    ]
    thin = _thin_pane_detections(dets, bin_s=1.0)
    assert len(thin) == 2
    assert thin[0]["confidence"] == 0.9


def test_detect_all_panes_on_synthetic_stacked_frame(tmp_path):
    import cv2

    # Synthetic stacked broadcast: top overview, bottom blue|red.
    h, w = 240, 320
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:120, :] = (40, 40, 40)
    frame[120:, :160] = (80, 40, 20)  # blue-ish
    frame[120:, 160:] = (20, 20, 90)  # red-ish
    # Bumper-colored blobs in each side pane.
    cv2.rectangle(frame, (40, 160), (70, 190), (40, 40, 220), -1)  # red bumper (BGR)
    cv2.rectangle(frame, (200, 160), (230, 190), (220, 80, 40), -1)  # blue-ish

    path = tmp_path / "stacked.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h))
    for _ in range(12):
        writer.write(frame)
    writer.release()

    layout = CameraLayout(
        mode="stacked_sides",
        crop_top=0.0,
        crop_bottom=0.5,
        confidence=0.9,
        detail="test",
        panes=[
            CameraPane("overview", 0.0, 0.5, 0.0, 1.0, purpose="movement"),
            CameraPane("blue_side", 0.5, 1.0, 0.0, 0.5, purpose="blue", alliance_bias="blue"),
            CameraPane("red_side", 0.5, 1.0, 0.5, 1.0, purpose="red", alliance_bias="red"),
        ],
    )
    result = detect_all_panes(path, layout, frame_stride=2, max_frames=20)
    assert "detections" in result
    assert any("pane" in w.lower() or "detection" in w.lower() for w in result["warnings"])
