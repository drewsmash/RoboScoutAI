"""Robust alliance color: bumper band, adaptive calibration, voting, 3/3 balance."""

from __future__ import annotations

import numpy as np

from ramscout.identity import balance_alliances, bumper_alliance
from ramscout.trackers.alliance import (
    AllianceCalibrator,
    AllianceVoter,
    assign_alliance_slots,
    bumper_band,
    chroma_features,
    hsv_alliance,
    side_prior,
)

RED_BGR = (40, 40, 220)
BLUE_BGR = (220, 90, 30)


def _patch(color, shift=(0, 0, 0), gains=(1.0, 1.0, 1.0), size=(18, 40), seed=0):
    rng = np.random.default_rng(seed)
    base = np.array(color, dtype=np.float32)
    img = np.tile(base, (size[0], size[1], 1))
    img += rng.normal(0, 6, img.shape).astype(np.float32)
    img = img * np.array(gains, dtype=np.float32) + np.array(shift, dtype=np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_bumper_band_is_lower_part_of_box():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[70:100, 20:60] = RED_BGR  # bumper strip at the bottom
    frame[20:40, 20:60] = BLUE_BGR  # decoy up top
    band = bumper_band(frame, [20, 20, 60, 100])
    assert band is not None
    label, _conf = hsv_alliance(chroma_features(band))
    assert label == "red"
    assert bumper_alliance(frame[20:100, 20:60]) == "red"


def test_calibration_classifies_under_white_balance_shift():
    # Strong magenta cast: blue bumpers drift out of the global HSV blue window
    # (they read as hue ≈150, i.e. "red" to fixed thresholds).
    shift = (0, 0, 190)
    reds = [_patch(RED_BGR, shift=shift, seed=i) for i in range(30)]
    blues = [_patch(BLUE_BGR, shift=shift, seed=100 + i) for i in range(30)]
    hsv_blue_ok = sum(hsv_alliance(chroma_features(p))[0] == "blue" for p in blues)
    assert hsv_blue_ok < len(blues) // 2, "global thresholds should struggle here"

    cal = AllianceCalibrator(min_samples=20)
    for p in reds + blues:
        assert cal.add(chroma_features(p))
    assert cal.fit()
    assert cal.calibrated
    assert all(cal.classify(p)[0] == "red" for p in reds)
    assert all(cal.classify(p)[0] == "blue" for p in blues)
    assert all(cal.classify(p)[2] == "calibrated" for p in reds + blues)


def test_calibration_refuses_single_cluster():
    cal = AllianceCalibrator(min_samples=10)
    for i in range(20):
        cal.add(chroma_features(_patch(RED_BGR, seed=i)))
    assert not cal.fit()
    assert not cal.calibrated
    # Falls back to HSV thresholds.
    label, conf, method = cal.classify(_patch(BLUE_BGR))
    assert label == "blue" and method == "hsv" and conf >= 0.5


def test_voter_hysteresis_resists_flicker_then_flips():
    voter = AllianceVoter(window=10, flip_ratio=0.65, min_votes=3)
    for _ in range(5):
        voter.vote(7, "red", 0.9)
    assert voter.label(7)[0] == "red"
    for _ in range(3):
        voter.vote(7, "blue", 0.9)
    assert voter.label(7)[0] == "red", "3 of 8 opposing votes must not flip"
    for _ in range(6):
        voter.vote(7, "blue", 0.9)
    assert voter.label(7)[0] == "blue"


def test_side_prior_only_early_and_in_alliance_zone():
    label, w = side_prior(20.0, 5.0, field_length=651.2, alliance_depth=158.6)
    assert label == "blue" and w > 0.5
    label, w = side_prior(630.0, 5.0, field_length=651.2, alliance_depth=158.6)
    assert label == "red" and w > 0.5
    label, w = side_prior(630.0, 90.0, field_length=651.2, alliance_depth=158.6)
    assert label == "unknown" and w == 0.0
    _label, w = side_prior(325.0, 5.0, field_length=651.2, alliance_depth=158.6)
    assert w < 0.1


def test_assign_alliance_slots_forces_three_and_three():
    labels = assign_alliance_slots([0.95, 0.9, 0.8, 0.6, 0.2, 0.1])
    assert labels.count("red") == 3 and labels.count("blue") == 3
    assert labels[3] == "blue", "weakest red-leaning track is pushed to blue"
    # Fewer tracks than slots: no forcing beyond the evidence.
    assert assign_alliance_slots([0.9, 0.8, 0.7, 0.2]) == ["red", "red", "red", "blue"]
    # Prior can break a coin flip.
    labels = assign_alliance_slots([0.5, 0.5], prior_red=[0.9, 0.1], prior_weight=0.5)
    assert labels == ["red", "blue"]


def test_balance_alliances_on_samples():
    samples = []
    # Four tracks lean red, two blue — consensus keeps evidence; does NOT force 3v3.
    spec = {1: ("red", 0.9, 600), 2: ("red", 0.9, 580), 3: ("red", 0.85, 560), 4: ("red", 0.9, 540), 5: ("blue", 0.9, 30), 6: ("blue", 0.9, 60)}
    for tid, (al, conf, x) in spec.items():
        for i in range(10):
            samples.append({"track_id": tid, "t": i * 0.5, "x": float(x), "y": 100.0, "alliance": al, "alliance_conf": conf})
    out = balance_alliances(samples)
    per = {tid: {s["alliance"] for s in out if s["track_id"] == tid} for tid in spec}
    assert all(len(v) == 1 for v in per.values())
    final = {tid: next(iter(v)) for tid, v in per.items()}
    # No forced flip of a strong red into blue to invent a 3v3 roster.
    assert final[4] == "red"
    assert final[1] == "red" and final[5] == "blue"
    assert sum(1 for v in final.values() if v == "red") == 4
    assert all("alliance_conf" in s for s in out)


def test_balance_alliances_keeps_unknown():
    samples = [
        {"track_id": 1, "t": 0.0, "x": 50.0, "y": 100.0, "alliance": "unknown", "alliance_conf": 0.1},
        {"track_id": 1, "t": 1.0, "x": 55.0, "y": 100.0, "alliance": "unknown", "alliance_conf": 0.1},
        {"track_id": 2, "t": 0.0, "x": 500.0, "y": 100.0, "alliance": "red", "alliance_conf": 0.9},
        {"track_id": 2, "t": 1.0, "x": 510.0, "y": 100.0, "alliance": "red", "alliance_conf": 0.9},
    ]
    out = balance_alliances(samples)
    assert all(s["alliance"] == "unknown" for s in out if s["track_id"] == 1)
    assert all(s["alliance"] == "red" for s in out if s["track_id"] == 2)
