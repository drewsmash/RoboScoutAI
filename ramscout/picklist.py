"""Draft / pick-list helpers for alliance selection from RoboScoutAI cards."""

from __future__ import annotations

from typing import Any


def _team_int(value: Any) -> int | None:
    try:
        return int(str(value).replace("frc", ""))
    except (TypeError, ValueError):
        return None


def _normalize_card(card: dict[str, Any]) -> dict[str, Any] | None:
    """Map a TeamCard (or aggregated scout-book entry) into draft metrics."""
    if not isinstance(card, dict):
        return None
    team = _team_int(card.get("team"))
    if team is None:
        return None

    matches = max(1, int(card.get("matches") or 1))
    hubs = float(card.get("hub_score_candidates") or 0)
    climb_bool = bool(card.get("climb_attempt"))
    climb_rate = float(card.get("climb_rate") if card.get("climb_rate") is not None else (1.0 if climb_bool else 0.0))
    defense = float(card.get("defense_time_s") or 0)
    path = float(card.get("path_length_in") or 0)
    max_speed = float(card.get("max_speed_in_s") or 0)
    collection = float(card.get("collection_time_s") or 0)

    # Legacy / TBA-style nested shapes if present
    end = card.get("endgame") or {}
    if "climb_rate" in end:
        climb_rate = float(end.get("climb_rate") or climb_rate)
    tele = card.get("teleop") or {}
    if tele.get("coral") or tele.get("algae"):
        coral = float((tele.get("coral") or {}).get("total") or 0)
        algae = float((tele.get("algae") or {}).get("total") or 0)
        hubs = max(hubs, coral + algae)

    hubs_pm = hubs / matches
    defense_pm = defense / matches
    path_pm = path / matches

    score = (
        hubs_pm * 22.0
        + climb_rate * 24.0
        + min(defense_pm, 45.0) * 0.35
        + min(path_pm / 100.0, 8.0) * 2.5
        + min(max_speed / 50.0, 3.0) * 3.0
        + min(collection / matches, 20.0) * 0.4
    )

    reasons: list[str] = []
    why = str(card.get("why") or "").strip()
    if why:
        reasons.append(why)
    if hubs_pm >= 3:
        reasons.append(f"{hubs_pm:.1f} hub dwells/match")
    if climb_rate >= 0.5:
        reasons.append(f"{climb_rate:.0%} climb rate" if matches > 1 or climb_rate < 1 else "climb attempt")
    if defense_pm >= 12:
        reasons.append(f"{defense_pm:.0f}s defense/match")
    if path_pm >= 800:
        reasons.append("high field coverage")
    if not reasons:
        reasons.append("solid all-around profile")

    return {
        "team": team,
        "nickname": card.get("nickname") or "",
        "alliance": card.get("alliance") or "",
        "score": round(score, 2),
        "matches": matches,
        "hubs_per_match": round(hubs_pm, 2),
        "climb_rate": round(climb_rate, 3),
        "defense_s_per_match": round(defense_pm, 1),
        "path_in_per_match": round(path_pm, 0),
        "reasons": reasons,
        "why": why or (reasons[0] if reasons else ""),
        "card": card,
    }


def draft_scores(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank scouting cards for alliance draft order."""
    ranked: list[dict[str, Any]] = []
    for card in cards:
        row = _normalize_card(card)
        if row:
            ranked.append(row)
    ranked.sort(key=lambda row: (-row["score"], row["team"]))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def suggest_picks(
    cards: list[dict[str, Any]],
    *,
    already_picked: list[int] | None = None,
    limit: int = 12,
) -> dict[str, Any]:
    """Suggest first / second / third round style picks from ranked cards."""
    taken = {int(t) for t in (already_picked or [])}
    ranked = [r for r in draft_scores(cards) if r["team"] not in taken][: max(1, limit)]
    top = ranked[:3]
    return {
        "ranked": ranked,
        "first_round": top,
        "second_round": ranked[3:6],
        "third_round": ranked[6:9],
        "excluded": sorted(taken),
        "suggested_alliance": [
            {"team": row["team"], "why": row.get("why") or (row.get("reasons") or [""])[0]}
            for row in top
        ],
    }


def alliance_summary(cards: list[dict[str, Any]], teams: list[int]) -> dict[str, Any]:
    """Summarize an alliance of up to three teams from scouting cards."""
    by_team: dict[int, dict[str, Any]] = {}
    for row in draft_scores(cards):
        by_team[row["team"]] = row

    selected = []
    missing = []
    total_hubs = 0.0
    total_climb = 0.0
    found = 0
    for t in [int(x) for x in teams[:3]]:
        row = by_team.get(t)
        if not row:
            missing.append(t)
            selected.append({"team": t, "found": False})
            continue
        total_hubs += float(row["hubs_per_match"])
        total_climb += float(row["climb_rate"])
        found += 1
        selected.append(
            {
                "team": t,
                "found": True,
                "nickname": row.get("nickname") or "",
                "hubs_per_match": row["hubs_per_match"],
                "climb_rate": row["climb_rate"],
                "defense_s_per_match": row["defense_s_per_match"],
                "score": row["score"],
            }
        )

    return {
        "teams": selected,
        "missing": missing,
        "combined_hubs_per_match": round(total_hubs, 2),
        "avg_climb_rate": round(total_climb / found, 3) if found else 0.0,
        "complete": len(missing) == 0 and len(teams) >= 1,
    }


def compare_teams(cards: list[dict[str, Any]], teams: list[int]) -> dict[str, Any]:
    """Side-by-side comparison rows for selected teams."""
    by_team: dict[int, dict[str, Any]] = {}
    for row in draft_scores(cards):
        by_team[row["team"]] = row

    rows = []
    for t in [int(x) for x in teams]:
        row = by_team.get(t)
        if not row:
            rows.append({"team": t, "found": False})
            continue
        rows.append(
            {
                "team": t,
                "found": True,
                "nickname": row.get("nickname") or "",
                "matches": row["matches"],
                "hubs_per_match": row["hubs_per_match"],
                "climb_rate": row["climb_rate"],
                "defense_s_per_match": row["defense_s_per_match"],
                "path_in_per_match": row["path_in_per_match"],
                "score": row["score"],
                "reasons": row["reasons"],
            }
        )
    return {"teams": rows}


def aggregate_cards(cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge multiple match cards for the same team into one scout-book row."""
    buckets: dict[int, list[dict[str, Any]]] = {}
    for card in cards:
        team = _team_int(card.get("team") if isinstance(card, dict) else None)
        if team is None:
            continue
        buckets.setdefault(team, []).append(card)

    merged: list[dict[str, Any]] = []
    for team, group in buckets.items():
        n = len(group)
        merged.append(
            {
                "team": str(team),
                "nickname": next((c.get("nickname") for c in group if c.get("nickname")), ""),
                "alliance": group[-1].get("alliance") or "",
                "matches": n,
                "hub_score_candidates": sum(float(c.get("hub_score_candidates") or 0) for c in group),
                "climb_attempt": any(bool(c.get("climb_attempt")) for c in group),
                "climb_rate": sum(1.0 if c.get("climb_attempt") else 0.0 for c in group) / n,
                "defense_time_s": sum(float(c.get("defense_time_s") or 0) for c in group),
                "path_length_in": sum(float(c.get("path_length_in") or 0) for c in group),
                "max_speed_in_s": max((float(c.get("max_speed_in_s") or 0) for c in group), default=0),
                "collection_time_s": sum(float(c.get("collection_time_s") or 0) for c in group),
                "why": next((str(c.get("why")) for c in reversed(group) if c.get("why")), ""),
                "scout_role": next((c.get("scout_role") for c in reversed(group) if c.get("scout_role")), ""),
                "scout_climb": next((c.get("scout_climb") for c in reversed(group) if c.get("scout_climb")), ""),
                "trust_carpet": next((c.get("trust_carpet") for c in reversed(group) if c.get("trust_carpet") is not None), None),
            }
        )
    return merged
