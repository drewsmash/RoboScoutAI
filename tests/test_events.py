from ramscout.events import build_cards, detect_events
from ramscout.simulate import demo_poses


def test_demo_paths_produce_hub_and_climb_events():
    poses = demo_poses()
    events = detect_events(poses)
    types = {e.type for e in events}
    assert "hub_score_candidate" in types
    assert "climb_attempt" in types
    teams = {e.team for e in events if e.type == "hub_score_candidate"}
    assert "195" in teams
    assert "59" in teams


def test_demo_cards_cover_six_robots():
    poses = demo_poses()
    events = detect_events(poses)
    cards = build_cards(poses, events, nicknames={"59": "Ramtech"})
    assert len(cards) == 6
    ramtech = next(c for c in cards if c.team == "59")
    assert ramtech.nickname == "Ramtech"
    assert ramtech.alliance == "red"
    assert ramtech.hub_score_candidates >= 1
    assert ramtech.climb_attempt
    assert ramtech.path_length_in > 0
