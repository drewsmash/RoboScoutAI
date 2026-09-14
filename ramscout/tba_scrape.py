"""TBA HTML scraper fallback when the official API key is missing or fails.

Scrapes public The Blue Alliance pages (match / event / team). This is a
best-effort fallback inspired by community TBA archives (frc-db, TBA data dumps)
and is not a substitute for the official API when you have a key.
"""

from __future__ import annotations

import logging
import re
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

log = logging.getLogger(__name__)

TBA_SITE = "https://www.thebluealliance.com"
USER_AGENT = "RamScoutAI/0.4 (+https://github.com/drewsmash/RamScoutAI; TBA HTML fallback)"


def scrape_match(match_key: str) -> dict[str, Any] | None:
    """Fetch a public TBA match page and parse teams + scores."""
    key = (match_key or "").strip()
    if not key:
        return None
    html = _get(f"{TBA_SITE}/match/{quote(key)}")
    if not html:
        return None
    parsed = _parse_match_html(html, key)
    if parsed:
        parsed["source"] = "tba_html"
    return parsed


def scrape_event_matches(event_key: str) -> list[dict[str, Any]]:
    """Fetch public event match table rows (keys + alliances when present)."""
    key = (event_key or "").strip()
    if not key:
        return []
    html = _get(f"{TBA_SITE}/event/{quote(key)}")
    if not html:
        return []
    return _parse_event_matches_html(html, key)


def scrape_team(team_number: int | str) -> dict[str, Any] | None:
    """Fetch public team page nickname/location."""
    num = str(team_number).strip().removeprefix("frc")
    if not num.isdigit():
        return None
    html = _get(f"{TBA_SITE}/team/{num}")
    if not html:
        return None
    nickname = ""
    m = re.search(r'<h2[^>]*>\s*Team\s+\d+\s*[-–—]\s*([^<]+)', html, re.I)
    if m:
        nickname = m.group(1).strip()
    if not nickname:
        m = re.search(r'<title>\s*(\d+)\s*[-–—]\s*([^|<]+)', html, re.I)
        if m:
            nickname = m.group(2).strip()
    return {
        "key": f"frc{num}",
        "team_number": int(num),
        "nickname": nickname,
        "source": "tba_html",
    }


def resolve_match_html(
    *,
    match_key: str | None = None,
    event_key: str | None = None,
    comp_level: str | None = None,
    match_number: int | None = None,
    set_number: int | None = None,
) -> dict[str, Any] | None:
    """Resolve a match via HTML only (no TBA API key)."""
    if match_key:
        found = scrape_match(match_key)
        if found:
            return found
    if not event_key or not comp_level or match_number is None:
        return None
    # Build canonical TBA match key when possible.
    level = comp_level.lower()
    if level == "qm":
        guess = f"{event_key}_qm{int(match_number)}"
    elif level in {"qf", "sf", "f"}:
        sn = int(set_number or 1)
        guess = f"{event_key}_{level}{sn}m{int(match_number)}"
    else:
        guess = f"{event_key}_{level}{int(match_number)}"
    found = scrape_match(guess)
    if found:
        return found
    # Scan event page for a matching row as a second chance.
    for row in scrape_event_matches(event_key):
        if row.get("comp_level") != level:
            continue
        if int(row.get("match_number") or 0) != int(match_number):
            continue
        if set_number and int(row.get("set_number") or 1) != int(set_number):
            continue
        detailed = scrape_match(str(row.get("key") or ""))
        return detailed or row
    return None


def enrich_from_html(match: dict[str, Any]) -> dict[str, Any]:
    """Attach nicknames scraped from team pages when missing."""
    teams = dict(match.get("teams") or {})
    for color in ("blue", "red"):
        for key in match.get("alliances", {}).get(color, {}).get("team_keys", []):
            if key in teams and teams[key].get("nickname"):
                continue
            info = scrape_team(key)
            if not info:
                continue
            teams[key] = {
                "key": key,
                "team_number": info.get("team_number"),
                "nickname": info.get("nickname") or "",
                "alliance": color,
            }
    out = dict(match)
    out["teams"] = teams
    out.setdefault("source", "tba_html")
    return out


def _get(url: str) -> str | None:
    try:
        req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"})
        with urlopen(req, timeout=25) as resp:
            return resp.read().decode("utf-8", "replace")
    except Exception as exc:  # noqa: BLE001
        log.info("TBA HTML fetch failed for %s (%s)", url, exc)
        return None


def _parse_match_html(html: str, match_key: str) -> dict[str, Any] | None:
    # Primary alliance table: <td class="red"> / <td class="blue"> with /team/N links.
    red_teams = _teams_from_class(html, "red")
    blue_teams = _teams_from_class(html, "blue")
    # Scores from the Total Score row (redScore / blueScore).
    red_score = blue_score = None
    m = re.search(
        r'class="redScore"[^>]*>\s*(?:<b>)?\s*(\d+)\s*(?:</b>)?\s*</td>\s*'
        r'<th>\s*Total Score\s*</th>\s*'
        r'<td[^>]*class="blueScore"[^>]*>\s*(?:<b>)?\s*(\d+)',
        html,
        re.I | re.S,
    )
    if m:
        red_score, blue_score = int(m.group(1)), int(m.group(2))
    else:
        m = re.search(
            r'redScore[^>]*>\s*<b>\s*(\d+)\s*</b>.*?Total Score.*?blueScore[^>]*>\s*<b>\s*(\d+)\s*</b>',
            html,
            re.I | re.S,
        )
        if m:
            red_score, blue_score = int(m.group(1)), int(m.group(2))

    if not red_teams and not blue_teams:
        return None

    event_key = match_key.split("_")[0] if "_" in match_key else ""
    comp_level, set_number, match_number = _split_match_key(match_key)
    winner = ""
    if red_score is not None and blue_score is not None:
        if red_score > blue_score:
            winner = "red"
        elif blue_score > red_score:
            winner = "blue"

    teams: dict[str, dict[str, Any]] = {}
    for color, nums in (("red", red_teams), ("blue", blue_teams)):
        for num in nums:
            key = f"frc{num}"
            teams[key] = {
                "key": key,
                "team_number": num,
                "nickname": "",
                "alliance": color,
            }

    return {
        "key": match_key,
        "event_key": event_key,
        "comp_level": comp_level,
        "set_number": set_number,
        "match_number": match_number,
        "winning_alliance": winner,
        "alliances": {
            "red": {"team_keys": [f"frc{n}" for n in red_teams], "score": red_score},
            "blue": {"team_keys": [f"frc{n}" for n in blue_teams], "score": blue_score},
        },
        "teams": teams,
        "videos": [],
        "score_breakdown": None,
        "zebra": None,
        "source": "tba_html",
    }


def _teams_from_class(html: str, color: str) -> list[int]:
    # Match cells like <td ... class="red"> ... <a href="/team/1350/2024">1350</a>
    teams: list[int] = []
    for m in re.finditer(
        rf'<td[^>]*class="[^"]*\b{color}\b[^"]*"[^>]*>.*?href="/team/(\d+)',
        html,
        re.I | re.S,
    ):
        num = int(m.group(1))
        if num not in teams:
            teams.append(num)
        if len(teams) >= 3:
            break
    # Fallback: data-team="frcNNNN" inside colored cells.
    if len(teams) < 3:
        for m in re.finditer(
            rf'<td[^>]*class="[^"]*\b{color}\b[^"]*"[^>]*>.*?data-team="frc(\d+)',
            html,
            re.I | re.S,
        ):
            num = int(m.group(1))
            if num not in teams:
                teams.append(num)
            if len(teams) >= 3:
                break
    return teams[:3]


def _parse_event_matches_html(html: str, event_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in re.finditer(rf'href="/match/({re.escape(event_key)}_[^"]+)"', html, re.I):
        key = m.group(1)
        if any(r.get("key") == key for r in rows):
            continue
        comp_level, set_number, match_number = _split_match_key(key)
        rows.append(
            {
                "key": key,
                "event_key": event_key,
                "comp_level": comp_level,
                "set_number": set_number,
                "match_number": match_number,
                "source": "tba_html",
            }
        )
    return rows


def _split_match_key(match_key: str) -> tuple[str, int, int]:
    # 2024nhdur_qm12 / 2024nhdur_sf1m2 / 2024nhdur_f1m1
    m = re.search(r"_(qm|qf|sf|f)(\d+)(?:m(\d+))?$", match_key, re.I)
    if not m:
        return "qm", 1, 0
    level = m.group(1).lower()
    if level == "qm":
        return "qm", 1, int(m.group(2))
    return level, int(m.group(2)), int(m.group(3) or 1)
