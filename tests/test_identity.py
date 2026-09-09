from ramscout.identity import constrain_to_teams, linear_assignment, stitch_occlusions
import numpy as np


def test_constrain_prefers_exact_team_number():
    assert constrain_to_teams("59", ["59", "159", "195"]) == "59"
    assert constrain_to_teams("159", ["59", "159"]) == "159"
    assert constrain_to_teams("5", ["59", "195"]) is None
    assert constrain_to_teams("195x", ["59", "195"]) == "195"


def test_stitch_occlusions_merges_nearby_track_ids():
    samples = [
        {"track_id": 1, "t": 0.0, "x": 10.0, "y": 40.0, "alliance": "blue"},
        {"track_id": 1, "t": 1.0, "x": 12.0, "y": 41.0, "alliance": "blue"},
        {"track_id": 9, "t": 1.4, "x": 14.0, "y": 42.0, "alliance": "blue"},
        {"track_id": 9, "t": 2.2, "x": 20.0, "y": 44.0, "alliance": "blue"},
    ]
    stitched = stitch_occlusions(samples)
    assert {row["track_id"] for row in stitched} == {1}


def test_stitch_does_not_merge_opposite_alliances():
    samples = [
        {"track_id": 1, "t": 0.0, "x": 10.0, "y": 40.0, "alliance": "blue"},
        {"track_id": 1, "t": 1.0, "x": 12.0, "y": 41.0, "alliance": "blue"},
        {"track_id": 2, "t": 1.3, "x": 13.0, "y": 42.0, "alliance": "red"},
        {"track_id": 2, "t": 2.0, "x": 14.0, "y": 42.0, "alliance": "red"},
    ]
    stitched = stitch_occlusions(samples)
    assert {row["track_id"] for row in stitched} == {1, 2}


def test_linear_assignment_picks_min_cost():
    cost = np.array([[9.0, 1.0], [2.0, 8.0]])
    pairs = dict(linear_assignment(cost))
    assert pairs[0] == 1
    assert pairs[1] == 0
