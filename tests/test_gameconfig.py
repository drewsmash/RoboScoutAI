from ramscout.gameconfig import available_years, load_game, public_game


def test_loads_2026_rebuilt_layout():
    game = load_game(2026)
    assert game["name"] == "REBUILT"
    assert "2026.png" in game["field_image"]
    assert game["robot_icons"]["blue"].endswith("blue.png")


def test_unknown_year_falls_back_to_shipped_config():
    game = load_game(2099)
    assert game["year_fallback"] is True
    assert game["config_year"] in available_years()


def test_public_game_exposes_field_assets():
    pub = public_game(2026)
    assert pub["field_image"].startswith("/static/")
    assert pub["match_end_s"] == 160
    assert pub["landmarks"]["blue_hub"]["label"] == "Hub"
