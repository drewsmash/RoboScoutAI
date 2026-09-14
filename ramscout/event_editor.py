"""Manual scout-event editing and timeline overrides."""

from __future__ import annotations

from typing import Any
from uuid import uuid4


EDITABLE_TYPES = {
    "hub_score_candidate",
    "climb_attempt",
    "defense",
    "collect",
    "foul",
    "note",
    "climb_success",
    "climb_fail",
}


def _eid(event: dict[str, Any]) -> str:
    if event.get("id"):
        return str(event["id"])
    return (
        f"{event.get('team')}-{event.get('type')}-{event.get('t')}-"
        f"{event.get('detail', '')[:24]}"
    )


def ensure_event_ids(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for event in events or []:
        row = dict(event)
        row["id"] = str(row.get("id") or _eid(row))
        out.append(row)
    return out


def apply_event_edits(
    events: list[dict[str, Any]],
    *,
    add: list[dict[str, Any]] | None = None,
    remove_ids: list[str] | None = None,
    update: list[dict[str, Any]] | None = None,
    reject_ids: list[str] | None = None,
    confirm_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Return a new event list after applying human edits."""
    rows = ensure_event_ids(events)
    by_id = {str(e["id"]): dict(e) for e in rows}

    for eid in remove_ids or []:
        by_id.pop(str(eid), None)

    for eid in reject_ids or []:
        if str(eid) in by_id:
            by_id[str(eid)]["rejected"] = True
            by_id[str(eid)]["confirmed"] = False

    for eid in confirm_ids or []:
        if str(eid) in by_id:
            by_id[str(eid)]["confirmed"] = True
            by_id[str(eid)]["rejected"] = False

    for patch in update or []:
        eid = str(patch.get("id") or "")
        if not eid or eid not in by_id:
            continue
        row = by_id[eid]
        for key in ("type", "t", "team", "zone", "detail", "confidence", "duration_s", "period"):
            if key in patch and patch[key] is not None:
                row[key] = patch[key]
        row["edited"] = True

    for item in add or []:
        row = dict(item)
        etype = str(row.get("type") or "note")
        if etype not in EDITABLE_TYPES:
            etype = "note"
        row["type"] = etype
        row["id"] = str(row.get("id") or f"manual-{uuid4().hex[:10]}")
        row["manual"] = True
        row["confirmed"] = True
        row.setdefault("confidence", 0.9)
        row.setdefault("detail", "Manual scout entry")
        row.setdefault("zone", "")
        row.setdefault("period", "")
        row.setdefault("t", 0.0)
        by_id[row["id"]] = row

    out = list(by_id.values())
    out.sort(key=lambda e: (float(e.get("t") or 0), str(e.get("team") or "")))
    return out


def cards_from_events(
    cards: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recompute card tallies from an edited event list (paths untouched)."""
    active = [e for e in events if not e.get("rejected")]
    by_team: dict[str, list[dict[str, Any]]] = {}
    for event in active:
        team = str(event.get("team") or "")
        if team:
            by_team.setdefault(team, []).append(event)

    updated = []
    for card in cards or []:
        row = dict(card)
        team = str(row.get("team") or "")
        team_events = by_team.get(team, [])
        row["events"] = team_events
        row["hub_score_candidates"] = sum(
            1 for e in team_events if e.get("type") == "hub_score_candidate"
        )
        row["climb_attempt"] = any(
            e.get("type") in {"climb_attempt", "climb_success"} for e in team_events
        )
        row["climb_success"] = any(e.get("type") == "climb_success" for e in team_events)
        row["defense_time_s"] = round(
            sum(float(e.get("duration_s") or 0) for e in team_events if e.get("type") == "defense"),
            2,
        )
        row["foul_count"] = sum(1 for e in team_events if e.get("type") == "foul")
        updated.append(row)
    return updated
