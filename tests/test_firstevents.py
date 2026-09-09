from ramscout.firstevents import resolve_firstevents
from ramscout.titles import parse_match_title


def test_resolve_south_florida_quals_65():
    hints = parse_match_title("Qualification 65 - South Florida Regional")
    # Title omits year; pipeline normally fills DEFAULT_YEAR.
    from ramscout.titles import TitleHints

    hints = TitleHints(
        year=2026,
        event_name=hints.event_name,
        comp_level=hints.comp_level,
        set_number=hints.set_number,
        match_number=hints.match_number,
        video_id=hints.video_id,
        title=hints.title,
    )
    match = resolve_firstevents(hints, event_key="2026flwp")
    assert match is not None
    assert match["key"] == "2026flwp_qm65"
    assert match["alliances"]["blue"]["score"] == 507
    assert match["alliances"]["red"]["score"] == 183
    assert match["alliances"]["blue"]["team_keys"] == ["frc2383", "frc11138", "frc59"]
    assert match["alliances"]["red"]["team_keys"] == ["frc179", "frc5472", "frc9201"]
    assert match["winning_alliance"] == "blue"
    assert match["source"]["scores"] == "firstevents"
