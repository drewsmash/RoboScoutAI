"""Tests for draft / pick-list ranking helpers and API."""

from fastapi.testclient import TestClient

from app import app
from ramscout.picklist import (
    aggregate_cards,
    alliance_summary,
    compare_teams,
    draft_scores,
    suggest_picks,
)


client = TestClient(app)


SAMPLE_CARDS = [
    {
        "team": "59",
        "nickname": "Ramtech",
        "alliance": "blue",
        "hub_score_candidates": 5,
        "climb_attempt": True,
        "defense_time_s": 8,
        "path_length_in": 1200,
        "max_speed_in_s": 80,
        "collection_time_s": 12,
    },
    {
        "team": "254",
        "nickname": "Cheesy Poofs",
        "alliance": "red",
        "hub_score_candidates": 2,
        "climb_attempt": False,
        "defense_time_s": 30,
        "path_length_in": 900,
        "max_speed_in_s": 60,
        "collection_time_s": 5,
    },
    {
        "team": "1678",
        "alliance": "blue",
        "hub_score_candidates": 4,
        "climb_attempt": True,
        "defense_time_s": 4,
        "path_length_in": 1100,
        "max_speed_in_s": 70,
        "collection_time_s": 10,
    },
]


def test_draft_scores_rank_climbers_and_hubs_first():
    ranked = draft_scores(SAMPLE_CARDS)
    assert [r["team"] for r in ranked][0] in {59, 1678}
    assert ranked[0]["score"] >= ranked[-1]["score"]
    assert ranked[0]["reasons"]


def test_suggest_picks_excludes_taken_teams():
    picks = suggest_picks(SAMPLE_CARDS, already_picked=[59], limit=10)
    assert 59 not in [r["team"] for r in picks["ranked"]]
    assert picks["excluded"] == [59]
    assert len(picks["first_round"]) <= 3


def test_aggregate_cards_merges_matches():
    doubled = SAMPLE_CARDS + [
        {**SAMPLE_CARDS[0], "hub_score_candidates": 3, "climb_attempt": False},
    ]
    merged = aggregate_cards(doubled)
    ram = next(c for c in merged if c["team"] == "59")
    assert ram["matches"] == 2
    assert ram["hub_score_candidates"] == 8
    assert 0 < ram["climb_rate"] < 1


def test_alliance_summary_and_compare():
    summary = alliance_summary(SAMPLE_CARDS, [59, 254, 1678])
    assert summary["complete"] is True
    assert summary["combined_hubs_per_match"] > 0
    cmp = compare_teams(SAMPLE_CARDS, [59, 9999])
    assert cmp["teams"][0]["found"] is True
    assert cmp["teams"][1]["found"] is False


def test_picklist_api():
    res = client.post("/api/picklist", json={"cards": SAMPLE_CARDS, "limit": 5})
    assert res.status_code == 200
    body = res.json()
    assert body["ranked"]
    assert "first_round" in body


def test_compare_api():
    res = client.post("/api/compare", json={"cards": SAMPLE_CARDS, "teams": [59, 254]})
    assert res.status_code == 200
    body = res.json()
    assert body["compare"]["teams"][0]["team"] == 59
    assert body["alliance"]["combined_hubs_per_match"] > 0


def test_index_has_icons_and_picklist():
    res = client.get("/")
    assert res.status_code == 200
    assert "manifest.webmanifest" in res.text
    assert "apple-touch-icon" in res.text
    assert "Pick list" in res.text
    icons = client.get("/static/icons/icon-192.png")
    assert icons.status_code == 200
    assert icons.content[:8] == b"\x89PNG\r\n\x1a\n"
    fav = client.get("/static/favicon.png")
    assert fav.status_code == 200
