"""The Blue Alliance client and match resolution from a YouTube VOD."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from ramscout.titles import TitleHints, normalize_event_name

log = logging.getLogger(__name__)

TBA_BASE = "https://www.thebluealliance.com/api/v3"
USER_AGENT = "RamScoutAI/0.2 (team 59; match auto-scout)"


class TBAError(RuntimeError):
    pass


class TBAClient:
    def __init__(self, auth_key: str, timeout: float = 20.0):
        key = (auth_key or "").strip()
        if not key:
            raise TBAError("A TBA auth key is required to look up match data.")
        self._client = httpx.Client(
            base_url=TBA_BASE,
            headers={
                "X-TBA-Auth-Key": key,
                "User-Agent": USER_AGENT,
                "Accept": "application/json",
            },
            timeout=timeout,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "TBAClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get(self, path: str) -> Any:
        response = self._client.get(path)
        if response.status_code == 401:
            raise TBAError("TBA rejected the auth key (401).")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    def events_for_year(self, year: int) -> list[dict[str, Any]]:
        data = self.get(f"/events/{year}/simple")
        return data or []

    def event_matches(self, event_key: str) -> list[dict[str, Any]]:
        data = self.get(f"/event/{event_key}/matches")
        return data or []

    def match(self, match_key: str) -> dict[str, Any] | None:
        data = self.get(f"/match/{match_key}")
        return data

    def team(self, team_key: str) -> dict[str, Any] | None:
        return self.get(f"/team/{team_key}/simple")

    def zebra(self, match_key: str) -> dict[str, Any] | None:
        return self.get(f"/match/{match_key}/zebra_motionworks")


def resolve_match(
    client: TBAClient,
    hints: TitleHints,
    event_key: str | None = None,
    match_key: str | None = None,
) -> dict[str, Any] | None:
    """Find a TBA match from explicit keys or parsed YouTube title hints."""
    if match_key:
        found = client.match(match_key)
        if found:
            return found

    year = hints.year
    if event_key:
        return _match_from_event(client, event_key, hints)

    if not year:
        return None

    events = client.events_for_year(year)
    ranked = _rank_events(events, hints.event_name)
    for event in ranked[:8]:
        found = _match_from_event(client, event["key"], hints)
        if found:
            return found
    return None


def enrich_match(client: TBAClient, match: dict[str, Any]) -> dict[str, Any]:
    """Attach nicknames and optional Zebra telemetry to a TBA match payload."""
    teams: dict[str, dict[str, Any]] = {}
    for color in ("blue", "red"):
        for key in match.get("alliances", {}).get(color, {}).get("team_keys", []):
            info = client.team(key) or {"key": key, "team_number": int(key.replace("frc", "")), "nickname": ""}
            teams[key] = {
                "key": key,
                "team_number": info.get("team_number"),
                "nickname": info.get("nickname") or "",
                "alliance": color,
            }

    zebra = None
    try:
        zebra = client.zebra(match["key"])
    except Exception as exc:  # noqa: BLE001 — optional bonus data
        log.info("Zebra lookup skipped: %s", exc)

    return {
        "key": match.get("key"),
        "event_key": match.get("event_key"),
        "comp_level": match.get("comp_level"),
        "set_number": match.get("set_number"),
        "match_number": match.get("match_number"),
        "time": match.get("actual_time") or match.get("time"),
        "winning_alliance": match.get("winning_alliance") or "",
        "alliances": match.get("alliances") or {},
        "score_breakdown": match.get("score_breakdown"),
        "videos": match.get("videos") or [],
        "teams": teams,
        "zebra": zebra,
    }


def _match_from_event(client: TBAClient, event_key: str, hints: TitleHints) -> dict[str, Any] | None:
    matches = client.event_matches(event_key)
    if hints.video_id:
        for match in matches:
            for video in match.get("videos") or []:
                if video.get("type") == "youtube" and video.get("key") == hints.video_id:
                    return match
    if hints.comp_level and hints.match_number is not None:
        for match in matches:
            if match.get("comp_level") != hints.comp_level:
                continue
            if int(match.get("match_number") or 0) != int(hints.match_number):
                continue
            if hints.set_number and int(match.get("set_number") or 1) != int(hints.set_number):
                continue
            return match
    return None


def _rank_events(events: list[dict[str, Any]], event_name: str | None) -> list[dict[str, Any]]:
    if not event_name:
        return events
    needle = normalize_event_name(event_name)
    tokens = [t for t in needle.split() if len(t) > 2]

    def score(event: dict[str, Any]) -> int:
        hay = normalize_event_name(
            " ".join(str(event.get(k) or "") for k in ("name", "short_name", "key"))
        )
        return sum(1 for token in tokens if token in hay)

    return sorted(events, key=score, reverse=True)
