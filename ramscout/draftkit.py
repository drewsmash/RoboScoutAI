"""Role-weighted draft presets, alliance fit, and EPA blending."""

from __future__ import annotations

from typing import Any

from ramscout.picklist import aggregate_cards, draft_scores

ROLE_PRESETS: dict[str, dict[str, float]] = {
    "balanced": {
        "hubs": 22.0,
        "climb": 24.0,
        "defense": 0.35,
        "path": 2.5,
        "speed": 3.0,
        "collection": 0.4,
        "epa": 1.0,
    },
    "scorer": {
        "hubs": 36.0,
        "climb": 12.0,
        "defense": 0.1,
        "path": 2.0,
        "speed": 2.5,
        "collection": 0.8,
        "epa": 1.2,
    },
    "defender": {
        "hubs": 8.0,
        "climb": 10.0,
        "defense": 1.2,
        "path": 1.5,
        "speed": 4.0,
        "collection": 0.1,
        "epa": 0.6,
    },
    "climber": {
        "hubs": 10.0,
        "climb": 48.0,
        "defense": 0.15,
        "path": 1.0,
        "speed": 1.5,
        "collection": 0.2,
        "epa": 0.8,
    },
    "flexible": {
        "hubs": 18.0,
        "climb": 18.0,
        "defense": 0.5,
        "path": 3.0,
        "speed": 3.5,
        "collection": 0.6,
        "epa": 1.0,
    },
}


def score_with_role(
    cards: list[dict[str, Any]],
    *,
    role: str = "balanced",
    epa_by_team: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    weights = ROLE_PRESETS.get(role) or ROLE_PRESETS["balanced"]
    epa_by_team = epa_by_team or {}
    ranked: list[dict[str, Any]] = []
    for card in aggregate_cards(cards):
        matches = max(1, int(card.get("matches") or 1))
        hubs_pm = float(card.get("hub_score_candidates") or 0) / matches
        climb_rate = float(
            card.get("climb_rate")
            if card.get("climb_rate") is not None
            else (1.0 if card.get("climb_attempt") else 0.0)
        )
        defense_pm = float(card.get("defense_time_s") or 0) / matches
        path_pm = float(card.get("path_length_in") or 0) / matches
        speed = float(card.get("max_speed_in_s") or 0)
        collection_pm = float(card.get("collection_time_s") or 0) / matches
        team = str(card.get("team") or "").replace("frc", "")
        epa = float(epa_by_team.get(team) or card.get("epa") or 0)

        score = (
            hubs_pm * weights["hubs"]
            + climb_rate * weights["climb"]
            + min(defense_pm, 45.0) * weights["defense"]
            + min(path_pm / 100.0, 8.0) * weights["path"]
            + min(speed / 50.0, 3.0) * weights["speed"]
            + min(collection_pm, 20.0) * weights["collection"]
            + epa * weights["epa"]
        )
        ranked.append(
            {
                "team": int(team) if team.isdigit() else team,
                "nickname": card.get("nickname") or "",
                "role": role,
                "score": round(score, 2),
                "matches": matches,
                "hubs_per_match": round(hubs_pm, 2),
                "climb_rate": round(climb_rate, 3),
                "defense_s_per_match": round(defense_pm, 1),
                "epa": round(epa, 2),
                "card": card,
            }
        )
    ranked.sort(key=lambda row: (-float(row["score"]), str(row["team"])))
    for i, row in enumerate(ranked, start=1):
        row["rank"] = i
    return ranked


def alliance_fit(
    cards: list[dict[str, Any]],
    locked: list[int],
    *,
    pool_limit: int = 12,
    epa_by_team: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Suggest partners that cover climb / defense / scoring gaps."""
    locked_set = {int(t) for t in locked}
    ranked = score_with_role(cards, role="balanced", epa_by_team=epa_by_team)
    locked_rows = [r for r in ranked if int(r["team"]) in locked_set]
    pool = [r for r in ranked if int(r["team"]) not in locked_set]

    have_climb = any(float(r["climb_rate"]) >= 0.5 for r in locked_rows)
    have_defense = any(float(r["defense_s_per_match"]) >= 10 for r in locked_rows)
    have_scoring = any(float(r["hubs_per_match"]) >= 2.5 for r in locked_rows)

    suggestions = []
    for row in pool:
        bonus = 0.0
        reasons = []
        if not have_climb and float(row["climb_rate"]) >= 0.5:
            bonus += 18
            reasons.append("covers climb")
        if not have_defense and float(row["defense_s_per_match"]) >= 10:
            bonus += 14
            reasons.append("covers defense")
        if not have_scoring and float(row["hubs_per_match"]) >= 2.5:
            bonus += 16
            reasons.append("covers scoring")
        if not reasons:
            reasons.append("overall fit")
        suggestions.append({**row, "fit_score": round(float(row["score"]) + bonus, 2), "fit_reasons": reasons})

    suggestions.sort(key=lambda r: (-r["fit_score"], r["team"]))
    return {
        "locked": locked_rows,
        "gaps": {
            "climb": not have_climb,
            "defense": not have_defense,
            "scoring": not have_scoring,
        },
        "suggestions": suggestions[:pool_limit],
    }


def blend_epa(
    cards: list[dict[str, Any]],
    epa_by_team: dict[str, float],
    *,
    role: str = "balanced",
) -> list[dict[str, Any]]:
    return score_with_role(cards, role=role, epa_by_team=epa_by_team)


def draft_board_state(
    cards: list[dict[str, Any]],
    *,
    picked: list[int] | None = None,
    do_not_pick: list[int] | None = None,
    role: str = "balanced",
) -> dict[str, Any]:
    picked = [int(t) for t in (picked or [])]
    dnp = {int(t) for t in (do_not_pick or [])}
    ranked = [r for r in score_with_role(cards, role=role) if int(r["team"]) not in dnp]
    available = [r for r in ranked if int(r["team"]) not in set(picked)]
    return {
        "role": role,
        "presets": list(ROLE_PRESETS),
        "picked": picked,
        "do_not_pick": sorted(dnp),
        "available": available,
        "first_round": available[:8],
        "second_round": available[8:16],
        "third_round": available[16:24],
        "baseline": draft_scores(cards)[:24],
    }
