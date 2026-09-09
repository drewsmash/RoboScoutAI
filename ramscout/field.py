"""2026 REBUILT field geometry in inches.

Coordinate frame:
  x = 0 at the blue alliance wall, x = FIELD_LENGTH at the red alliance wall
  y = 0 at the scoring-table side, y = FIELD_WIDTH at the opposite side
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

FIELD_LENGTH = 651.2
FIELD_WIDTH = 317.7
ALLIANCE_DEPTH = 158.6
NEUTRAL_DEPTH = 283.0
HUB_SIZE = 47.0
AUTO_END_S = 20.0
TELEOP_END_S = 160.0
ENDGAME_START_S = 130.0
MATCH_END_S = 160.0

# Hub centers sit on the alliance / neutral boundary, mid-width.
BLUE_HUB = (ALLIANCE_DEPTH, FIELD_WIDTH / 2.0)
RED_HUB = (FIELD_LENGTH - ALLIANCE_DEPTH, FIELD_WIDTH / 2.0)

# Towers sit against the alliance wall, offset toward the scoring table.
BLUE_TOWER = (28.0, FIELD_WIDTH * 0.38)
RED_TOWER = (FIELD_LENGTH - 28.0, FIELD_WIDTH * 0.38)

# Depots / outposts are approximate footprints used for occupancy stats.
BLUE_DEPOT = (40.0, FIELD_WIDTH * 0.82)
RED_DEPOT = (FIELD_LENGTH - 40.0, FIELD_WIDTH * 0.82)
BLUE_OUTPOST = (22.0, FIELD_WIDTH * 0.18)
RED_OUTPOST = (FIELD_LENGTH - 22.0, FIELD_WIDTH * 0.18)

HUB_SCORE_RADIUS = 62.0
TOWER_RADIUS = 48.0
DEFENSE_PROXIMITY = 42.0


@dataclass(frozen=True)
class FieldPoint:
    x: float
    y: float

    def clip(self) -> "FieldPoint":
        return FieldPoint(
            x=float(np.clip(self.x, 0, FIELD_LENGTH)),
            y=float(np.clip(self.y, 0, FIELD_WIDTH)),
        )


def zone_name(x: float, y: float) -> str:
    """Named occupancy zone for a field point."""
    if x < ALLIANCE_DEPTH:
        if _near(x, y, BLUE_TOWER, TOWER_RADIUS):
            return "blue_tower"
        if _near(x, y, BLUE_HUB, HUB_SCORE_RADIUS):
            return "blue_hub"
        if _near(x, y, BLUE_DEPOT, 40):
            return "blue_depot"
        if _near(x, y, BLUE_OUTPOST, 36):
            return "blue_outpost"
        return "blue_alliance"
    if x > FIELD_LENGTH - ALLIANCE_DEPTH:
        if _near(x, y, RED_TOWER, TOWER_RADIUS):
            return "red_tower"
        if _near(x, y, RED_HUB, HUB_SCORE_RADIUS):
            return "red_hub"
        if _near(x, y, RED_DEPOT, 40):
            return "red_depot"
        if _near(x, y, RED_OUTPOST, 36):
            return "red_outpost"
        return "red_alliance"
    return "neutral"


def own_hub(alliance: str) -> tuple[float, float]:
    return BLUE_HUB if alliance == "blue" else RED_HUB


def opponent_hub(alliance: str) -> tuple[float, float]:
    return RED_HUB if alliance == "blue" else BLUE_HUB


def own_tower(alliance: str) -> tuple[float, float]:
    return BLUE_TOWER if alliance == "blue" else RED_TOWER


def on_own_half(x: float, alliance: str) -> bool:
    mid = FIELD_LENGTH / 2.0
    return x < mid if alliance == "blue" else x >= mid


def on_opponent_half(x: float, alliance: str) -> bool:
    return not on_own_half(x, alliance)


def period_name(t: float) -> str:
    if t < AUTO_END_S:
        return "auto"
    if t >= ENDGAME_START_S:
        return "endgame"
    return "teleop"


def _near(x: float, y: float, target: Sequence[float], radius: float) -> bool:
    dx = x - target[0]
    dy = y - target[1]
    return dx * dx + dy * dy <= radius * radius


def distance(a: Iterable[float], b: Iterable[float]) -> float:
    ax, ay = a
    bx, by = b
    return float(np.hypot(bx - ax, by - ay))
