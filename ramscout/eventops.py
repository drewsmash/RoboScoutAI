"""Event-wide batch analysis, team dossiers, and schedule helpers."""

from __future__ import annotations

import logging
from typing import Any

from ramscout.picklist import aggregate_cards
from ramscout.tba import TBAClient, TBAError

log = logging.getLogger(__name__)


def list_event_matches(client: TBAClient, event_key: str) -> list[dict[str, Any]]:
    matches = client.event_matches(event_key) or []
    rows = []
    for match in matches:
        alliances = match.get("alliances") or {}
        blue = [k.replace("frc", "") for k in (alliances.get("blue") or {}).get("team_keys") or []]
        red = [k.replace("frc", "") for k in (alliances.get("red") or {}).get("team_keys") or []]
        videos = match.get("videos") or []
        yt = next((v.get("key") for v in videos if v.get("type") == "youtube" and v.get("key")), None)
        rows.append(
            {
                "key": match.get("key"),
                "comp_level": match.get("comp_level"),
                "match_number": match.get("match_number"),
                "set_number": match.get("set_number"),
                "blue": blue,
                "red": red,
                "youtube_key": yt,
                "youtube_url": f"https://www.youtube.com/watch?v={yt}" if yt else None,
                "predicted_time": match.get("predicted_time") or match.get("time"),
            }
        )
    rows.sort(key=lambda r: (str(r.get("comp_level") or ""), int(r.get("set_number") or 0), int(r.get("match_number") or 0)))
    return rows


def watchlist_schedule(
    client: TBAClient,
    event_key: str,
    watch_teams: list[int],
) -> dict[str, Any]:
    wanted = {int(t) for t in watch_teams}
    matches = list_event_matches(client, event_key)
    relevant = []
    for match in matches:
        teams = {int(t) for t in (match.get("blue") or []) + (match.get("red") or []) if str(t).isdigit()}
        hit = sorted(wanted & teams)
        if hit:
            relevant.append({**match, "watch_teams": hit})
    return {
        "event_key": event_key,
        "watch_teams": sorted(wanted),
        "matches": relevant,
        "count": len(relevant),
    }


def team_dossier(cards: list[dict[str, Any]], team: int | str) -> dict[str, Any]:
    team_s = str(team).replace("frc", "")
    matches = [c for c in cards if str(c.get("team") or "").replace("frc", "") == team_s]
    if not matches:
        return {"team": team_s, "matches": 0, "found": False}
    agg = aggregate_cards(matches)[0]
    hubs = [float(c.get("hub_score_candidates") or 0) for c in matches]
    defense = [float(c.get("defense_time_s") or 0) for c in matches]
    climbs = [1.0 if c.get("climb_attempt") else 0.0 for c in matches]
    return {
        "team": team_s,
        "found": True,
        "nickname": agg.get("nickname") or "",
        "matches": len(matches),
        "aggregate": agg,
        "hubs_avg": round(sum(hubs) / len(hubs), 2),
        "hubs_best": max(hubs),
        "hubs_worst": min(hubs),
        "defense_avg_s": round(sum(defense) / len(defense), 1),
        "climb_rate": round(sum(climbs) / len(climbs), 3),
        "consistency": round(1.0 - (max(hubs) - min(hubs)) / max(max(hubs), 1.0), 3),
        "cards": matches,
    }


def batch_plan_from_event(
    tba_key: str,
    event_key: str,
    *,
    only_with_video: bool = True,
    limit: int = 50,
) -> dict[str, Any]:
    try:
        with TBAClient(tba_key) as client:
            matches = list_event_matches(client, event_key)
    except TBAError as exc:
        return {"ok": False, "error": str(exc), "jobs": []}

    jobs = []
    for match in matches:
        if only_with_video and not match.get("youtube_url"):
            continue
        jobs.append(
            {
                "match_key": match.get("key"),
                "event_key": event_key,
                "url": match.get("youtube_url"),
                "label": f"{match.get('comp_level')}{match.get('match_number')}",
                "blue": match.get("blue"),
                "red": match.get("red"),
            }
        )
        if len(jobs) >= limit:
            break
    return {
        "ok": True,
        "event_key": event_key,
        "planned": len(jobs),
        "jobs": jobs,
        "note": "Enqueue each job with POST /api/jobs using url + match_key + event_key.",
    }
