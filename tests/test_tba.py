from ramscout.tba import resolve_match
from ramscout.titles import parse_match_title


class FakeTBA:
    def __init__(self):
        self.matches = [
            {
                "key": "2026nhdur_qm12",
                "event_key": "2026nhdur",
                "comp_level": "qm",
                "match_number": 12,
                "set_number": 1,
                "videos": [{"type": "youtube", "key": "dQw4w9WgXcQ"}],
                "alliances": {
                    "blue": {"team_keys": ["frc195", "frc230", "frc177"], "score": 148},
                    "red": {"team_keys": ["frc59", "frc319", "frc238"], "score": 131},
                },
            }
        ]

    def match(self, match_key: str):
        return next((m for m in self.matches if m["key"] == match_key), None)

    def event_matches(self, event_key: str):
        return [m for m in self.matches if m["event_key"] == event_key]

    def events_for_year(self, year: int):
        return [{"key": "2026nhdur", "name": "New Hampshire District Event", "short_name": "NH"}]


def test_resolve_match_by_youtube_video_id():
    hints = parse_match_title(
        "2026 Some Other Event Name - Qualification Match 99",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    )
    found = resolve_match(FakeTBA(), hints, event_key="2026nhdur")
    assert found is not None
    assert found["key"] == "2026nhdur_qm12"


def test_resolve_match_by_explicit_key():
    hints = parse_match_title("unrelated title")
    found = resolve_match(FakeTBA(), hints, match_key="2026nhdur_qm12")
    assert found["match_number"] == 12


def test_resolve_match_by_title_number():
    hints = parse_match_title("2026 New Hampshire District Event - Qualification Match 12")
    found = resolve_match(FakeTBA(), hints, event_key="2026nhdur")
    assert found["key"] == "2026nhdur_qm12"
