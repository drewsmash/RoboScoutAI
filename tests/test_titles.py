from ramscout.titles import extract_youtube_id, normalize_event_name, parse_match_title


def test_extract_youtube_id_from_watch_url():
    assert extract_youtube_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_extract_youtube_id_from_short_url():
    assert extract_youtube_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"


def test_parse_qualification_title():
    hints = parse_match_title(
        "2026 New Hampshire District Event - Qualification Match 12",
        "https://youtu.be/dQw4w9WgXcQ",
    )
    assert hints.year == 2026
    assert hints.comp_level == "qm"
    assert hints.match_number == 12
    assert hints.event_name and "Hampshire" in hints.event_name
    assert hints.video_id == "dQw4w9WgXcQ"


def test_parse_quals_shorthand():
    hints = parse_match_title("Quals 17 - 2020 PNW District Glacier Peak Event")
    assert hints.year == 2020
    assert hints.comp_level == "qm"
    assert hints.match_number == 17


def test_parse_playoff_title():
    hints = parse_match_title("2026 New England Championship - Playoff Match 6")
    assert hints.comp_level == "sf"
    assert hints.match_number == 6


def test_parse_short_qualification_title():
    hints = parse_match_title(
        "Qualification 65 - South Florida Regional",
        "https://youtu.be/m9uLAGKtenM",
    )
    assert hints.comp_level == "qm"
    assert hints.match_number == 65
    assert hints.event_name and "Florida" in hints.event_name
    assert hints.video_id == "m9uLAGKtenM"
