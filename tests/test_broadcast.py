from ramscout.broadcast import match_from_overlay, merge_tba, parse_broadcast_text
from ramscout.simulate import DEMO_VIDEO


def test_parses_alliances_and_scores_from_broadcast_text():
    reading = parse_broadcast_text(DEMO_VIDEO["title"], DEMO_VIDEO["description"])
    assert reading.year == 2026
    assert reading.match_number == 12
    assert reading.blue_teams == ["195", "230", "177"]
    assert reading.red_teams == ["59", "319", "238"]
    assert reading.blue_score == 148
    assert reading.red_score == 131
    assert "youtube_title" in reading.sources


def test_match_from_overlay_uses_video_source():
    reading = parse_broadcast_text(DEMO_VIDEO["title"], DEMO_VIDEO["description"])
    match = match_from_overlay(reading)
    assert match["alliances"]["blue"]["score"] == 148
    assert match["alliances"]["red"]["team_keys"] == ["frc59", "frc319", "frc238"]
    assert match["source"]["scores"] == "video"
    assert match["winning_alliance"] == "blue"


def test_vs_line_fills_alliances_when_labels_missing():
    reading = parse_broadcast_text(
        "2026 Event - Qualification Match 4",
        "195 230 177 vs 59 319 238\nFinal score: Blue 100, Red 88",
    )
    assert reading.blue_teams == ["195", "230", "177"]
    assert reading.red_teams == ["59", "319", "238"]
    assert reading.blue_score == 100
    assert reading.red_score == 88


def test_video_scores_win_over_tba():
    reading = parse_broadcast_text(DEMO_VIDEO["title"], DEMO_VIDEO["description"])
    video = match_from_overlay(reading)
    tba = {
        "key": "2026nhdur_qm12",
        "alliances": {
            "blue": {"score": 1, "team_keys": ["frc1", "frc2", "frc3"]},
            "red": {"score": 2, "team_keys": ["frc4", "frc5", "frc6"]},
        },
        "teams": {
            "frc195": {"key": "frc195", "team_number": 195, "nickname": "CyberKnights", "alliance": "blue"},
        },
    }
    merged = merge_tba(video, tba)
    assert merged["alliances"]["blue"]["score"] == 148
    assert merged["alliances"]["red"]["team_keys"] == ["frc59", "frc319", "frc238"]
    assert merged["source"]["scores"] == "video"
    assert merged["teams"]["frc195"]["nickname"] == "CyberKnights"
