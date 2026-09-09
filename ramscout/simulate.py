"""Synthetic REBUILT trajectories for the sample-match demo and tests."""

from __future__ import annotations

import math
from typing import Any

from ramscout.events import Pose
from ramscout.field import (
    ALLIANCE_DEPTH,
    BLUE_HUB,
    BLUE_TOWER,
    FIELD_LENGTH,
    FIELD_WIDTH,
    MATCH_END_S,
    RED_HUB,
    RED_TOWER,
)


DEMO_MATCH = {
    "key": "2026nhdur_qm12",
    "event_key": "2026nhdur",
    "comp_level": "qm",
    "set_number": 1,
    "match_number": 12,
    "time": None,
    "winning_alliance": "blue",
    "alliances": {
        "blue": {"score": 148, "team_keys": ["frc195", "frc230", "frc177"]},
        "red": {"score": 131, "team_keys": ["frc59", "frc319", "frc238"]},
    },
    "score_breakdown": {
        "blue": {"totalPoints": 148, "autoPoints": 32, "teleopPoints": 86, "endgamePoints": 30},
        "red": {"totalPoints": 131, "autoPoints": 18, "teleopPoints": 83, "endgamePoints": 30},
    },
    "videos": [{"type": "youtube", "key": "dQw4w9wgGcQ"}],
    "teams": {
        "frc195": {"key": "frc195", "team_number": 195, "nickname": "CyberKnights", "alliance": "blue"},
        "frc230": {"key": "frc230", "team_number": 230, "nickname": "Gaelhawks", "alliance": "blue"},
        "frc177": {"key": "frc177", "team_number": 177, "nickname": "Bobcat Robotics", "alliance": "blue"},
        "frc59": {"key": "frc59", "team_number": 59, "nickname": "Ramtech", "alliance": "red"},
        "frc319": {"key": "frc319", "team_number": 319, "nickname": "Big Bad Bob", "alliance": "red"},
        "frc238": {"key": "frc238", "team_number": 238, "nickname": "Crusaders", "alliance": "red"},
    },
    "zebra": None,
    "source": {"scores": "video", "teams": "video", "channels": ["youtube_title", "youtube_description"]},
}

DEMO_VIDEO = {
    "id": "demo",
    "title": "2026 New Hampshire District Event - Qualification Match 12",
    "description": (
        "Qualification Match 12 at the 2026 New Hampshire District Event.\n"
        "Blue Alliance: 195, 230, 177\n"
        "Red Alliance: 59, 319, 238\n"
        "Final score: Blue 148, Red 131\n"
        "Read from the match broadcast overlay."
    ),
    "duration": 165,
    "uploader": "FIRST Robotics Competition",
    "webpage_url": "",
    "thumbnail": "",
    "is_live": False,
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _goto(t0: float, t1: float, start: tuple[float, float], end: tuple[float, float], team: str, alliance: str, track_id: int) -> list[Pose]:
    poses: list[Pose] = []
    steps = max(int((t1 - t0) * 4), 2)
    for i in range(steps):
        u = i / (steps - 1)
        poses.append(
            Pose(
                t=round(t0 + (t1 - t0) * u, 2),
                x=_lerp(start[0], end[0], u),
                y=_lerp(start[1], end[1], u),
                team=team,
                alliance=alliance,
                track_id=track_id,
            )
        )
    return poses


def _hold(t0: float, t1: float, at: tuple[float, float], team: str, alliance: str, track_id: int) -> list[Pose]:
    return _goto(t0, t1, at, at, team, alliance, track_id)


def demo_poses() -> list[Pose]:
    """Six stylized robots: hub cycler, collector, defender, climber."""
    poses: list[Pose] = []

    # Blue 195 — hub cycler
    poses += _goto(0, 3, (40, 80), (ALLIANCE_DEPTH - 10, BLUE_HUB[1]), "195", "blue", 1)
    poses += _goto(3, 5, (ALLIANCE_DEPTH - 10, BLUE_HUB[1]), BLUE_HUB, "195", "blue", 1)
    poses += _hold(5, 8.5, BLUE_HUB, "195", "blue", 1)
    poses += _goto(8.5, 20, BLUE_HUB, (ALLIANCE_DEPTH + 40, FIELD_WIDTH * 0.45), "195", "blue", 1)
    for cycle in range(4):
        t0 = 22 + cycle * 18
        poses += _goto(t0, t0 + 5, (ALLIANCE_DEPTH + 50, 160), BLUE_HUB, "195", "blue", 1)
        poses += _hold(t0 + 5, t0 + 7.2, BLUE_HUB, "195", "blue", 1)
        poses += _goto(t0 + 7.2, t0 + 12, BLUE_HUB, (ALLIANCE_DEPTH + 50, 160), "195", "blue", 1)
    poses += _goto(130, 150, (ALLIANCE_DEPTH + 20, 140), BLUE_TOWER, "195", "blue", 1)
    poses += _goto(150, MATCH_END_S, BLUE_TOWER, BLUE_TOWER, "195", "blue", 1)

    # Blue 230 — collector in the neutral zone
    poses += _goto(0, 6, (50, 240), (220, 200), "230", "blue", 2)
    poses += _goto(6, 20, (220, 200), (300, 180), "230", "blue", 2)
    x = 300
    for i in range(0, 100):
        t = 22 + i * 1.0
        x = 260 + 80 * math.sin(i / 6)
        y = 150 + 40 * math.cos(i / 5)
        poses.append(Pose(t=t, x=x, y=y, team="230", alliance="blue", track_id=2))
    poses += _goto(130, MATCH_END_S, (280, 160), (90, 200), "230", "blue", 2)

    # Blue 177 — defense on red
    poses += _goto(0, 20, (45, 160), (200, 160), "177", "blue", 3)
    poses += _goto(20, 40, (200, 160), (FIELD_LENGTH - 120, 150), "177", "blue", 3)
    for i in range(0, 80):
        t = 40 + i * 1.1
        poses.append(
            Pose(
                t=t,
                x=FIELD_LENGTH - 110 + 12 * math.sin(i / 4),
                y=140 + 18 * math.cos(i / 3),
                team="177",
                alliance="blue",
                track_id=3,
            )
        )
    poses += _goto(130, MATCH_END_S, (FIELD_LENGTH - 110, 140), BLUE_TOWER, "177", "blue", 3)

    # Red 59 Ramtech — hub cycler + climb
    poses += _goto(0, 4, (FIELD_LENGTH - 40, 90), RED_HUB, "59", "red", 4)
    poses += _hold(4, 8.2, RED_HUB, "59", "red", 4)
    poses += _goto(8.2, 20, RED_HUB, (FIELD_LENGTH - ALLIANCE_DEPTH - 30, 150), "59", "red", 4)
    for cycle in range(3):
        t0 = 24 + cycle * 20
        poses += _goto(t0, t0 + 6, (FIELD_LENGTH - 220, 170), RED_HUB, "59", "red", 4)
        poses += _hold(t0 + 6, t0 + 8.2, RED_HUB, "59", "red", 4)
        poses += _goto(t0 + 8.2, t0 + 14, RED_HUB, (FIELD_LENGTH - 220, 170), "59", "red", 4)
    poses += _goto(128, 145, (FIELD_LENGTH - 180, 160), RED_TOWER, "59", "red", 4)
    poses += _hold(145, MATCH_END_S, RED_TOWER, "59", "red", 4)

    # Red 319 — midfield collector
    poses += _goto(0, 20, (FIELD_LENGTH - 50, 230), (380, 190), "319", "red", 5)
    for i in range(0, 100):
        t = 22 + i * 1.0
        poses.append(
            Pose(
                t=t,
                x=360 + 70 * math.cos(i / 7),
                y=170 + 35 * math.sin(i / 4),
                team="319",
                alliance="red",
                track_id=5,
            )
        )
    poses += _goto(130, MATCH_END_S, (360, 170), (FIELD_LENGTH - 80, 220), "319", "red", 5)

    # Red 238 — defense on blue hub
    poses += _goto(0, 20, (FIELD_LENGTH - 48, 140), (420, 140), "238", "red", 6)
    poses += _goto(20, 35, (420, 140), (130, 150), "238", "red", 6)
    for i in range(0, 80):
        t = 36 + i * 1.1
        poses.append(
            Pose(
                t=t,
                x=120 + 14 * math.sin(i / 5),
                y=145 + 16 * math.cos(i / 4),
                team="238",
                alliance="red",
                track_id=6,
            )
        )
    poses += _goto(128, MATCH_END_S, (120, 145), RED_TOWER, "238", "red", 6)
    return poses


def demo_tracks() -> list[dict[str, Any]]:
    return [
        {
            "t": p.t,
            "frame": int(p.t * 10),
            "track_id": p.track_id,
            "x": p.x,
            "y": p.y,
            "alliance": p.alliance,
            "team": p.team,
            "bbox": [0, 0, 0, 0],
        }
        for p in demo_poses()
    ]
