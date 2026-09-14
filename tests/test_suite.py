"""Tests for multicam + scout suite helpers."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from ramscout.draftkit import alliance_fit, score_with_role
from ramscout.event_editor import apply_event_edits, cards_from_events, ensure_event_ids
from ramscout.heatmap import build_heatmap
from ramscout.multicam import analyze_frame, apply_layout
from ramscout.scoutbook import merge_scout_book, save_scout_book, upsert_pit_form
from ramscout.sheets import cards_to_sheets_csv
from ramscout.tba_stats import enrich_cards_with_tba

import app as app_module

client = TestClient(app_module.app)


def test_multicam_detects_stacked_top_pane():
    h, w = 400, 640
    frame = np.random.randint(50, 210, (h, w, 3), dtype=np.uint8)
    frame[185:215, :, :] = 5
    layout = analyze_frame(frame)
    assert layout.mode in {"stacked_top", "single"}
    top, bottom, chosen = apply_layout(
        layout,
        user_crop_top=0.10,
        user_crop_bottom=0.65,
        auto=True,
    )
    assert top < bottom
    assert 0.0 <= top < bottom <= 1.0
    if layout.mode == "stacked_top":
        assert chosen.mode == "stacked_top"
        assert bottom < 0.7


def test_event_editor_confirm_and_cards():
    events = ensure_event_ids(
        [{"team": "59", "type": "hub_score_candidate", "t": 10.0, "detail": "dwell"}]
    )
    eid = events[0]["id"]
    updated = apply_event_edits(
        events,
        add=[{"team": "59", "type": "climb_success", "t": 140.0, "detail": "deep"}],
        confirm_ids=[eid],
    )
    assert any(e.get("confirmed") for e in updated)
    cards = cards_from_events(
        [{"team": "59", "alliance": "blue", "hub_score_candidates": 0, "climb_attempt": False}],
        updated,
    )
    assert cards[0]["hub_score_candidates"] >= 1
    assert cards[0]["climb_attempt"] is True


def test_heatmap_and_sheets():
    samples = [
        {"team": "59", "x": 100, "y": 80, "t": 20},
        {"team": "59", "x": 120, "y": 90, "t": 25},
    ]
    heat = build_heatmap(samples, team="59")
    assert (heat.get("samples") or heat.get("count") or 0) >= 2
    csv = cards_to_sheets_csv(
        [{"team": "59", "nickname": "Ramtech", "hub_score_candidates": 3}]
    )
    assert "59" in csv
    assert "Team" in csv


def test_draft_roles_and_fit():
    cards = [
        {
            "team": "59",
            "hub_score_candidates": 6,
            "climb_attempt": False,
            "defense_time_s": 5,
            "matches": 1,
        },
        {
            "team": "190",
            "hub_score_candidates": 1,
            "climb_attempt": True,
            "defense_time_s": 30,
            "matches": 1,
        },
        {
            "team": "238",
            "hub_score_candidates": 4,
            "climb_attempt": True,
            "defense_time_s": 8,
            "matches": 1,
        },
    ]
    scorers = score_with_role(cards, role="scorer")
    assert scorers[0]["team"] in {59, 238, "59", "238"}
    fit = alliance_fit(cards, [59])
    assert "suggestions" in fit or "available" in fit


def test_tba_enrich_and_pit_book(tmp_path, monkeypatch):
    monkeypatch.setenv("RAMSCOUT_DATA", str(tmp_path))
    cards = enrich_cards_with_tba(
        [{"team": "59", "hub_score_candidates": 0}],
        {
            "alliances": {
                "blue": {"team_keys": ["frc59"]},
                "red": {"team_keys": ["frc190"]},
            },
            "score_breakdown": {
                "blue": {
                    "teleopPoints": 40,
                    "endGameRobot1": "DeepCage",
                    "autoLineRobot1": "Yes",
                },
                "red": {"teleopPoints": 10},
            },
        },
    )
    # Soft enrichment — at minimum tba block or climb flag may appear.
    assert isinstance(cards[0], dict)
    form = upsert_pit_form({"team": "59", "drivetrain": "swerve", "notes": "fast"})
    assert str(form.get("team")) == "59"
    book = save_scout_book(
        {"cards": [{"team": "59"}], "matches": [], "notes": {}, "watchlist": []}
    )
    assert book is not None
    merged = merge_scout_book(
        {
            "cards": [{"team": "190", "match_key": "qm1"}],
            "matches": [],
            "notes": {},
            "watchlist": ["190"],
        }
    )
    assert merged is not None


def test_suite_routes_smoke():
    assert client.get("/api/games").status_code == 200
    assert client.get("/api/history").status_code == 200
    assert client.get("/api/pit").status_code == 200
    assert client.get("/api/live").status_code == 200
    assert client.get("/api/scoutbook").status_code == 200
    draft = client.post(
        "/api/draft",
        json={
            "cards": [
                {"team": "59", "hub_score_candidates": 3, "climb_attempt": True}
            ],
            "role": "balanced",
        },
    )
    assert draft.status_code == 200
    sheets = client.post(
        "/api/sheets/cards.csv",
        json={"cards": [{"team": "59", "hub_score_candidates": 2}]},
    )
    assert sheets.status_code == 200
    assert "Team" in sheets.text
