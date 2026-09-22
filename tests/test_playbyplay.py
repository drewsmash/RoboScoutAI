"""Play-by-play rows are built from tracks, without a language model."""

from __future__ import annotations

from ramscout.field import FIELD_LENGTH
from ramscout.playbyplay import annotate_cards, build_play_by_play, carpet_share_by_team, why_line


def _sample(**kwargs):
    base = {
        "track_id": 47,
        "team": "59",
        "alliance": "blue",
        "field_valid": True,
        "speed_in_s": 30,
    }
    base.update(kwargs)
    return base


def test_play_by_play_rows_use_robot_time_position_and_action():
    samples = [
        _sample(t=5, x=40, y=160, speed_in_s=20),
        _sample(t=42.1, x=210, y=140, speed_in_s=40),
        _sample(t=42.4, x=212, y=142, speed_in_s=40),
        _sample(t=90, x=300, y=150, speed_in_s=0),
    ]
    rows = build_play_by_play(samples)
    assert rows[0]["robot"] == "T47"
    assert rows[0]["team"] == "59"
    assert rows[0]["doing"] == "in alliance zone"
    assert rows[0]["period"] == "auto"
    hub = next(row for row in rows if row["doing"] == "near hub")
    assert hub["t"] == 42.1
    assert hub["x"] == 210.0
    assert hub["period"] == "teleop"
    assert any(row["doing"] == "stopped" for row in rows)
    # Consecutive identical actions collapse.
    assert sum(1 for row in rows if row["doing"] == "near hub") == 1


def test_crossing_midfield_and_invalid_points_are_skipped():
    mid = FIELD_LENGTH / 2
    samples = [
        _sample(t=30, x=mid - 40, y=150, speed_in_s=80),
        _sample(t=32, x=mid + 40, y=150, speed_in_s=80),
        _sample(t=33, x=None, y=None, field_valid=False),
    ]
    rows = build_play_by_play(samples)
    assert any(row["doing"] == "crossing midfield" for row in rows)
    assert all(row["x"] is not None for row in rows)


def test_trust_is_carpet_share_and_why_prefers_the_model():
    samples = [
        _sample(t=1, x=200, y=150),
        _sample(t=2, x=220, y=150),
        _sample(t=3, x=8, y=8),
        _sample(t=4, x=12, y=10),
    ]
    shares = carpet_share_by_team(samples)
    assert 0 < shares["59"] < 1
    rows = build_play_by_play(samples)
    assert why_line("59", rows) == "near hub (1 stretch)"
    modeled = why_line("59", rows, {"role": {"choice": "scorer"}, "climb": {"choice": "climb"}})
    assert modeled == "Model: scorer, climb climb"
    cards = [{"team": "59"}]
    annotate_cards(cards, samples, rows, {}, model_skipped=True)
    assert cards[0]["trust_carpet"] == round(shares["59"], 3)
    assert "skipped" in cards[0]["scout_note"]
    assert cards[0]["why"]
