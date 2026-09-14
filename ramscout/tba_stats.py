"""Merge TBA score_breakdown into video scout cards."""

from __future__ import annotations

from typing import Any


def _team_num(value: Any) -> str:
    return str(value).replace("frc", "").strip()


def breakdown_team_stats(match: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Extract per-team official-ish stats from a TBA match payload."""
    if not match:
        return {}
    breakdown = match.get("score_breakdown") or {}
    if not isinstance(breakdown, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    alliances = match.get("alliances") or {}
    for color in ("blue", "red"):
        side = breakdown.get(color) or {}
        if not isinstance(side, dict):
            continue
        keys = (alliances.get(color) or {}).get("team_keys") or []
        # Alliance-level totals applied equally as context; robot-level keys when present.
        alliance_totals = {
            "tba_auto_points": _num(side.get("autoPoints") or side.get("auto_points")),
            "tba_teleop_points": _num(side.get("teleopPoints") or side.get("teleop_points")),
            "tba_endgame_points": _num(
                side.get("endGameBargePoints")
                or side.get("endgamePoints")
                or side.get("endGameParkPoints")
                or side.get("totalPoints")
            ),
            "tba_foul_points": _num(side.get("foulPoints") or side.get("foul_points")),
            "tba_total_points": _num(side.get("totalPoints") or side.get("total_points")),
            "tba_rp": _num(side.get("rp") or side.get("RP")),
        }
        for idx, key in enumerate(keys):
            team = _team_num(key)
            robot = {
                **alliance_totals,
                "tba_alliance": color,
                "tba_station": idx + 1,
            }
            # Common per-robot endgame keys across recent games.
            for prefix in (
                f"endGameRobot{idx + 1}",
                f"endgameRobot{idx + 1}",
                f"endGameRobot{idx + 1}Status",
            ):
                if prefix in side:
                    robot["tba_endgame"] = side.get(prefix)
            for prefix in (f"autoLineRobot{idx + 1}", f"mobilityRobot{idx + 1}"):
                if prefix in side:
                    robot["tba_auto_leave"] = side.get(prefix)
            out[team] = robot
    return out


def enrich_cards_with_tba(
    cards: list[dict[str, Any]],
    match: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    stats = breakdown_team_stats(match)
    enriched = []
    for card in cards or []:
        row = dict(card)
        team = _team_num(row.get("team"))
        tba = stats.get(team) or {}
        if tba:
            row["tba"] = tba
            # Soft boost: if TBA says they climbed/parked, reflect on the card.
            endgame = str(tba.get("tba_endgame") or "").lower()
            if any(token in endgame for token in ("deep", "shallow", "climb", "cage", "park")):
                if "park" in endgame and "climb" not in endgame and "deep" not in endgame:
                    row.setdefault("tba_park", True)
                else:
                    row["climb_attempt"] = True
                    row["tba_climb"] = True
            leave = tba.get("tba_auto_leave")
            if leave in (True, "Yes", "yes", "YesMounted"):
                row["tba_auto_leave"] = True
            # Surface collection/hub context from alliance points when video is thin.
            if float(row.get("hub_score_candidates") or 0) == 0 and float(tba.get("tba_teleop_points") or 0) >= 20:
                row["tba_scoring_context"] = "Alliance teleop points suggest active scoring this match."
        enriched.append(row)
    return enriched


def _num(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
