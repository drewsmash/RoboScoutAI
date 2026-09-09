"""Year-specific FRC game layouts, field art, and overlay crop bands."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

GAMES_DIR = Path(__file__).resolve().parent / "games"
DEFAULT_YEAR = 2026


@lru_cache(maxsize=8)
def load_game(year: int | None = None) -> dict[str, Any]:
    """Load a year config. Falls back to the newest shipped year."""
    wanted = int(year or DEFAULT_YEAR)
    path = GAMES_DIR / f"{wanted}.json"
    if not path.exists():
        years = available_years()
        if not years:
            raise FileNotFoundError(f"No game configs in {GAMES_DIR}")
        path = GAMES_DIR / f"{years[-1]}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["requested_year"] = wanted
    data["config_year"] = int(data.get("year") or wanted)
    data["year_fallback"] = data["config_year"] != wanted
    return data


def available_years() -> list[int]:
    years = []
    for path in GAMES_DIR.glob("*.json"):
        try:
            years.append(int(path.stem))
        except ValueError:
            continue
    return sorted(years)


def public_game(year: int | None = None) -> dict[str, Any]:
    game = load_game(year)
    timing = game.get("timing") or {}
    field = game.get("field") or {}
    return {
        "year": game.get("config_year"),
        "requested_year": game.get("requested_year"),
        "name": game.get("name"),
        "field_image": game.get("field_image"),
        "robot_icons": game.get("robot_icons") or {},
        "length_in": field.get("length_in"),
        "width_in": field.get("width_in"),
        "alliance_depth_in": field.get("alliance_depth_in"),
        "auto_end_s": timing.get("auto_end_s"),
        "endgame_start_s": timing.get("endgame_start_s"),
        "match_end_s": timing.get("match_end_s"),
        "landmarks": game.get("landmarks") or {},
        "overlay": game.get("overlay") or {},
        "year_fallback": game.get("year_fallback", False),
        "available_years": available_years(),
        "upgrade": game.get("upgrade") or {},
    }
