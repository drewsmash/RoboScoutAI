"""TBA HTML scrape fallback (no API key)."""

from __future__ import annotations

from ramscout.tba_scrape import scrape_match, _split_match_key


def test_split_match_key():
    assert _split_match_key("2024nhdur_qm12") == ("qm", 1, 12)
    assert _split_match_key("2024nhdur_sf1m2") == ("sf", 1, 2)
    assert _split_match_key("2024nhdur_f1m1") == ("f", 1, 1)


def test_scrape_match_public_page():
    match = scrape_match("2024nhdur_qm1")
    assert match is not None
    assert match["key"] == "2024nhdur_qm1"
    assert match["source"] == "tba_html"
    assert match["alliances"]["red"]["team_keys"] == ["frc1350", "frc8046", "frc6691"]
    assert match["alliances"]["blue"]["team_keys"] == ["frc663", "frc4546", "frc7674"]
    assert match["alliances"]["red"]["score"] == 47
    assert match["alliances"]["blue"]["score"] == 54
