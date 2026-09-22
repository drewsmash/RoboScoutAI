"""Play-by-play rows from tracked field samples.

Each row is something a robot did (time, field position, action). No language
model is required to build or render the list.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from ramscout.field import (
    ALLIANCE_DEPTH,
    ENDGAME_START_S,
    FIELD_LENGTH,
    FIELD_WIDTH,
    period_name,
    zone_name,
)
from ramscout.geometry import FIELD_SOFT_MARGIN_IN, is_field_sample_drawable

HUB_ZONES = {"blue_hub", "red_hub"}
TOWER_ZONES = {"blue_tower", "red_tower"}
DEPOT_ZONES = {"blue_depot", "red_depot", "blue_outpost", "red_outpost"}
ALLIANCE_ZONE_NAMES = {"blue_alliance", "red_alliance"}


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def on_carpet(x: float, y: float, margin: float = FIELD_SOFT_MARGIN_IN) -> bool:
    """True when a point sits on the carpet, not the wall band."""
    return margin <= x <= FIELD_LENGTH - margin and margin <= y <= FIELD_WIDTH - margin


def _speed(sample: dict[str, Any], prev: dict[str, Any] | None) -> float | None:
    given = _num(sample.get("speed_in_s"))
    if given is not None:
        return given
    if prev is None or not is_field_sample_drawable(prev):
        return None
    dt = max(float(sample["t"]) - float(prev["t"]), 1e-3)
    dx = float(sample["x"]) - float(prev["x"])
    dy = float(sample["y"]) - float(prev["y"])
    return (dx * dx + dy * dy) ** 0.5 / dt


def action_for(sample: dict[str, Any], prev: dict[str, Any] | None) -> str:
    """Short action label for one field sample."""
    x = float(sample["x"])
    y = float(sample["y"])
    t = float(sample.get("t") or 0.0)
    zone = zone_name(x, y)
    if zone in TOWER_ZONES and t >= ENDGAME_START_S:
        return "climb"
    if zone in HUB_ZONES:
        return "near hub"
    if prev is not None and is_field_sample_drawable(prev):
        mid = FIELD_LENGTH / 2.0
        if (float(prev["x"]) - mid) * (x - mid) < 0 and abs(x - float(prev["x"])) > 12:
            return "crossing midfield"
    if zone in ALLIANCE_ZONE_NAMES or x < ALLIANCE_DEPTH or x > FIELD_LENGTH - ALLIANCE_DEPTH:
        return "in alliance zone"
    if zone in DEPOT_ZONES:
        return "at depot"
    speed = _speed(sample, prev)
    if speed is not None and speed < 8.0:
        return "stopped"
    return "moving"


def _row(sample: dict[str, Any], doing: str) -> dict[str, Any]:
    tid = sample.get("track_id")
    team = str(sample.get("team") or "")
    if team.startswith("T") and team[1:].isdigit():
        team = ""
    alliance = sample.get("alliance") if sample.get("alliance") in {"red", "blue"} else "unknown"
    return {
        "robot": f"T{tid}",
        "team": team,
        "t": round(float(sample["t"]), 1),
        "x": round(float(sample["x"]), 1),
        "y": round(float(sample["y"]), 1),
        "period": period_name(float(sample["t"])),
        "doing": doing,
        "alliance": alliance,
    }


def build_play_by_play(
    samples: list[dict[str, Any]],
    events: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Collapse each track into action changes. The model is not involved."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples or []:
        try:
            tid = int(sample.get("track_id"))
        except (TypeError, ValueError):
            continue
        grouped.setdefault(tid, []).append(sample)

    rows: list[dict[str, Any]] = []
    climbed: set[str] = set()
    for tid, group in grouped.items():
        ordered = sorted(group, key=lambda sample: float(sample.get("t") or 0.0))
        prev: dict[str, Any] | None = None
        last_doing: str | None = None
        for sample in ordered:
            if not is_field_sample_drawable(sample):
                prev = None
                continue
            doing = action_for(sample, prev)
            prev = sample
            if doing == last_doing:
                continue
            last_doing = doing
            row = _row(sample, doing)
            rows.append(row)
            if doing == "climb":
                climbed.add(str(row.get("team") or row["robot"]))

    for event in events or []:
        kind = str(event.get("type") or "")
        if "climb" not in kind:
            continue
        team = str(event.get("team") or "")
        if team in climbed:
            continue
        t = _num(event.get("t")) or 0.0
        rows.append(
            {
                "robot": team or "T?",
                "team": "" if team.startswith("T") else team,
                "t": round(t, 1),
                "x": None,
                "y": None,
                "period": period_name(t),
                "doing": "climb",
                "alliance": event.get("alliance") if event.get("alliance") in {"red", "blue"} else "unknown",
            }
        )
        climbed.add(team)

    rows.sort(key=lambda row: (float(row["t"]), str(row["robot"])))
    return rows


def carpet_share_by_team(samples: list[dict[str, Any]]) -> dict[str, float]:
    """Fraction of each robot's path length on the carpet versus the wall band."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for sample in samples or []:
        team = str(sample.get("team") or "")
        if not team:
            team = f"T{sample.get('track_id')}"
        grouped.setdefault(team, []).append(sample)
        track_key = f"T{sample.get('track_id')}"
        if track_key != team:
            grouped.setdefault(track_key, []).append(sample)

    shares: dict[str, float] = {}
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda sample: float(sample.get("t") or 0.0))
        carpet = 0.0
        wall = 0.0
        prev: dict[str, Any] | None = None
        for sample in ordered:
            if not is_field_sample_drawable(sample):
                prev = None
                continue
            if prev is None:
                prev = sample
                continue
            px, py = float(prev["x"]), float(prev["y"])
            dx = float(sample["x"]) - px
            dy = float(sample["y"]) - py
            dist = (dx * dx + dy * dy) ** 0.5
            prev = sample
            if dist <= 0 or dist > 200:
                continue
            if on_carpet(float(sample["x"]), float(sample["y"])) and on_carpet(px, py):
                carpet += dist
            else:
                wall += dist
        total = carpet + wall
        if total >= 1.0:
            shares[key] = carpet / total
    return shares


def why_line(team: str, rows: list[dict[str, Any]], answers: dict[str, Any] | None = None) -> str:
    """One sentence for a pick-list rank, from the model answer or the play-by-play."""
    answers = answers or {}
    role = answers.get("role") if isinstance(answers.get("role"), dict) else None
    climb = answers.get("climb") if isinstance(answers.get("climb"), dict) else None
    if role and role.get("choice"):
        text = f"Model: {role.get('choice')}"
        if climb and climb.get("choice"):
            text = f"{text}, climb {climb.get('choice')}"
        return text
    mine = [
        row
        for row in rows
        if str(row.get("team") or "") == str(team) or str(row.get("robot") or "") == str(team)
    ]
    if not mine:
        return "Play-by-play ready"
    doing, count = Counter(str(row.get("doing") or "moving") for row in mine).most_common(1)[0]
    noun = "stretch" if count == 1 else "stretches"
    return f"{doing} ({count} {noun})"


def annotate_cards(
    cards: list[dict[str, Any]],
    samples: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    team_scouts: dict[str, Any] | None = None,
    *,
    model_skipped: bool = False,
) -> None:
    """Write trust, why, and the skipped-model note onto scout cards."""
    shares = carpet_share_by_team(samples)
    scouts = team_scouts or {}
    for card in cards:
        team = str(card.get("team") or "")
        share = shares.get(team)
        if share is None and card.get("track_id") is not None:
            share = shares.get(f"T{card.get('track_id')}")
        if share is not None:
            card["trust_carpet"] = round(float(share), 3)
        scout = scouts.get(team) or {}
        available = bool(scout.get("available"))
        answers = scout.get("answers") if available and isinstance(scout.get("answers"), dict) else None
        card["why"] = why_line(team, rows, answers)
        if available:
            card.pop("scout_note", None)
            role = (answers or {}).get("role") if isinstance((answers or {}).get("role"), dict) else None
            climb = (answers or {}).get("climb") if isinstance((answers or {}).get("climb"), dict) else None
            if role and role.get("choice"):
                card["scout_role"] = role.get("choice")
            if climb and climb.get("choice"):
                card["scout_climb"] = climb.get("choice")
        elif model_skipped:
            card["scout_note"] = "Play-by-play is ready. The local scout model was skipped."
