"""Best-effort match lookup from frc-events.firstinspires.org HTML pages.

Used when TBA is unavailable and the VOD cannot be downloaded. Not an official API.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx

from ramscout.titles import TitleHints, normalize_event_name

log = logging.getLogger(__name__)

# Common regional / championship short codes seen in FIRST Event Web URLs.
_EVENT_ALIASES = {
    "south florida": "flwp",
    "south florida regional": "flwp",
    "new hampshire": "nhdur",
    "new hampshire district": "nhdur",
}


def resolve_firstevents(hints: TitleHints, event_key: str | None = None) -> dict[str, Any] | None:
    year = hints.year
    if not year or not hints.match_number:
        return None
    keys: list[str] = []
    if event_key:
        keys.append(event_key.lower())
    guessed = _guess_event_key(year, hints.event_name)
    if guessed and guessed not in keys:
        keys.append(guessed)
    level = hints.comp_level or "qm"
    path_level = "qualifications" if level == "qm" else "playoffs"
    for key in keys:
        code = key[4:] if key.startswith(str(year)) else key
        url = f"https://frc-events.firstinspires.org/{year}/{code.upper()}/{path_level}/{hints.match_number}"
        try:
            html = httpx.get(url, timeout=20.0, follow_redirects=True, headers={"User-Agent": "RamScoutAI/0.2"}).text
        except Exception as exc:  # noqa: BLE001
            log.info("FIRST Events fetch failed for %s: %s", url, exc)
            continue
        parsed = _parse_match_html(html)
        if not parsed:
            continue
        event_full = key if key.startswith(str(year)) else f"{year}{code.lower()}"
        return _as_match(parsed, event_full, level, int(hints.match_number), url)
    return None


def _guess_event_key(year: int, event_name: str | None) -> str | None:
    if not event_name:
        return None
    norm = normalize_event_name(event_name)
    for alias, code in _EVENT_ALIASES.items():
        if normalize_event_name(alias) in norm or norm in normalize_event_name(alias):
            return f"{year}{code}"
    # Fallback: take significant tokens.
    tokens = [t for t in norm.split() if len(t) > 2][:2]
    if not tokens:
        return None
    slug = "".join(t[:3] for t in tokens)
    return f"{year}{slug}"


def _parse_match_html(html: str) -> dict[str, Any] | None:
    blue_block = re.search(
        r'<td class="info[^"]*"[^>]*>\s*<div class="row">(.*?)</div>\s*</td>',
        html,
        re.I | re.S,
    )
    red_block = re.search(
        r'<td class="danger[^"]*"[^>]*>\s*<div class="row">(.*?)</div>\s*</td>',
        html,
        re.I | re.S,
    )
    if not blue_block or not red_block:
        return None
    blue_teams = re.findall(r">(\d{1,5})</div>", blue_block.group(1))
    red_teams = re.findall(r">(\d{1,5})</div>", red_block.group(1))
    if len(blue_teams) < 3 or len(red_teams) < 3:
        return None
    blue_teams, red_teams = blue_teams[:3], red_teams[:3]
    scores = re.search(
        r"Final Score</.*?<strong>(\d+)</strong>.*?<strong>(\d+)</strong>",
        html,
        re.I | re.S,
    )
    if not scores:
        return None
    # FIRST uses info=blue, danger=red on the Final Score row.
    blue_score, red_score = int(scores.group(1)), int(scores.group(2))
    return {
        "blue_teams": blue_teams,
        "red_teams": red_teams,
        "blue_score": blue_score,
        "red_score": red_score,
    }


def _as_match(parsed: dict[str, Any], event_key: str, level: str, number: int, url: str) -> dict[str, Any]:
    teams: dict[str, Any] = {}
    for color, nums in (("blue", parsed["blue_teams"]), ("red", parsed["red_teams"])):
        for num in nums:
            teams[f"frc{num}"] = {
                "key": f"frc{num}",
                "team_number": int(num),
                "nickname": "",
                "alliance": color,
            }
    blue_score = parsed["blue_score"]
    red_score = parsed["red_score"]
    winner = "blue" if blue_score > red_score else "red" if red_score > blue_score else ""
    return {
        "key": f"{event_key}_{level}{number}",
        "event_key": event_key,
        "comp_level": level,
        "set_number": 1,
        "match_number": number,
        "time": None,
        "winning_alliance": winner,
        "alliances": {
            "blue": {"score": blue_score, "team_keys": [f"frc{t}" for t in parsed["blue_teams"]]},
            "red": {"score": red_score, "team_keys": [f"frc{t}" for t in parsed["red_teams"]]},
        },
        "score_breakdown": None,
        "videos": [],
        "teams": teams,
        "zebra": None,
        "source": {
            "scores": "firstevents",
            "teams": "firstevents",
            "channels": ["firstevents", url],
        },
    }
