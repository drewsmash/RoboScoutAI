"""REBUILT auto-scout event heuristics from field-space trajectories.

Scoring locations (Hub, Tower, depot) are 2026-specific. For a new season:
copy ramscout/games/2026.json to YYYY.json, add web/fields/YYYY.png, then
update the constants in ramscout/field.py and the dwell/radius rules here.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ramscout.field import (
    AUTO_END_S,
    DEFENSE_PROXIMITY,
    ENDGAME_START_S,
    MATCH_END_S,
    TOWER_RADIUS,
    distance,
    on_opponent_half,
    own_hub,
    own_tower,
    period_name,
    zone_name,
)

HUB_DWELL_S = 1.2
HUB_COOLDOWN_S = 4.0
HUB_SPEED_IN_S = 6.0
HUB_PARK_RADIUS = 36.0
CLIMB_DWELL_S = 2.8
CLIMB_SPEED_IN_S = 10.0
DEFENSE_DWELL_S = 2.0
# Do not accumulate path length across long gaps / teleport jumps (camera cuts,
# invalid projections). Peak FRC robot speed is ~20 ft/s; anything faster is a
# tracker teleport. Also reject single-step jumps longer than ~one robot length×2.
MAX_PATH_SEGMENT_S = 1.25
MAX_PATH_SEGMENT_SPEED_IN_S = 200.0
MAX_PATH_SEGMENT_IN = 96.0


@dataclass
class Pose:
    t: float
    x: float
    y: float
    team: str
    alliance: str
    track_id: int = -1
    speed: float = 0.0


@dataclass
class ScoutEvent:
    team: str
    type: str
    t: float
    zone: str
    confidence: float
    detail: str
    period: str = ""
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if not data["period"]:
            data["period"] = period_name(self.t)
        return data


@dataclass
class TeamCard:
    team: str
    alliance: str
    nickname: str = ""
    start_xy: list[float] = field(default_factory=list)
    path_length_in: float = 0.0
    avg_speed_in_s: float = 0.0
    max_speed_in_s: float = 0.0
    zone_time_s: dict[str, float] = field(default_factory=dict)
    auto_path: list[list[float]] = field(default_factory=list)
    teleop_path: list[list[float]] = field(default_factory=list)
    endgame_path: list[list[float]] = field(default_factory=list)
    hub_score_candidates: int = 0
    defense_time_s: float = 0.0
    climb_attempt: bool = False
    collection_time_s: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def with_speeds(poses: Iterable[Pose]) -> list[Pose]:
    rows = sorted(poses, key=lambda p: (p.team, p.t))
    prev: dict[str, Pose] = {}
    out: list[Pose] = []
    for pose in rows:
        last = prev.get(pose.team)
        if last and pose.t > last.t:
            pose.speed = distance((pose.x, pose.y), (last.x, last.y)) / (pose.t - last.t)
        out.append(pose)
        prev[pose.team] = pose
    return out


def detect_events(poses: list[Pose]) -> list[ScoutEvent]:
    """Latch hub / defense / climb candidates from smoothed trajectories."""
    poses = with_speeds(poses)
    by_team: dict[str, list[Pose]] = defaultdict(list)
    for pose in poses:
        by_team[pose.team].append(pose)

    events: list[ScoutEvent] = []
    for team, path in by_team.items():
        path.sort(key=lambda p: p.t)
        events.extend(_hub_events(path))
        events.extend(_climb_events(path))
        events.extend(_defense_events(path, poses))
        events.extend(_collect_events(path))
    events.sort(key=lambda e: e.t)
    return events


def build_cards(
    poses: list[Pose],
    events: list[ScoutEvent],
    nicknames: dict[str, str] | None = None,
) -> list[TeamCard]:
    poses = with_speeds(poses)
    nicknames = nicknames or {}
    by_team: dict[str, list[Pose]] = defaultdict(list)
    for pose in poses:
        by_team[pose.team].append(pose)

    cards: list[TeamCard] = []
    for team, path in by_team.items():
        path.sort(key=lambda p: p.t)
        alliance = path[0].alliance
        card = TeamCard(team=team, alliance=alliance, nickname=nicknames.get(team, ""))
        if path:
            card.start_xy = [round(path[0].x, 1), round(path[0].y, 1)]
        speeds = [p.speed for p in path if p.speed > 0]
        card.max_speed_in_s = round(max(speeds), 1) if speeds else 0.0
        card.avg_speed_in_s = round(sum(speeds) / len(speeds), 1) if speeds else 0.0
        length = 0.0
        zone_time: dict[str, float] = defaultdict(float)
        for prev, cur in zip(path, path[1:]):
            dt = max(cur.t - prev.t, 0.0)
            step = distance((prev.x, prev.y), (cur.x, cur.y))
            # Skip camera-cut / teleport segments so scout cards aren't inflated
            # by corner spiders and gap interpolations.
            if (
                dt <= MAX_PATH_SEGMENT_S
                and step <= MAX_PATH_SEGMENT_IN
                and (dt <= 1e-6 or step / dt <= MAX_PATH_SEGMENT_SPEED_IN_S)
            ):
                length += step
                zone_time[zone_name(cur.x, cur.y)] += dt
            pt = [round(cur.x, 1), round(cur.y, 1), round(cur.t, 2)]
            period = period_name(cur.t)
            if period == "auto":
                card.auto_path.append(pt)
            elif period == "endgame":
                card.endgame_path.append(pt)
            else:
                card.teleop_path.append(pt)
        card.path_length_in = round(length, 1)
        card.zone_time_s = {k: round(v, 2) for k, v in zone_time.items()}
        team_events = [e.as_dict() for e in events if e.team == team]
        card.events = team_events
        card.hub_score_candidates = sum(1 for e in team_events if e["type"] == "hub_score_candidate")
        card.climb_attempt = any(e["type"] == "climb_attempt" for e in team_events)
        card.defense_time_s = round(
            sum(float(e.get("duration_s") or 0) for e in team_events if e["type"] == "defense"),
            2,
        )
        # Defense events store duration in detail when t_end isn't present.
        if card.defense_time_s == 0:
            card.defense_time_s = round(sum(zone_time.get(k, 0) for k in zone_time if "alliance" in k and alliance not in k), 2)
        card.collection_time_s = round(zone_time.get("neutral", 0.0), 2)
        cards.append(card)
    alliance_rank = {"blue": 0, "red": 1}
    cards.sort(key=lambda c: (alliance_rank.get(c.alliance, 9), c.team))
    return cards


def _hub_events(path: list[Pose]) -> list[ScoutEvent]:
    if not path:
        return []
    alliance = path[0].alliance
    hub = own_hub(alliance)
    events: list[ScoutEvent] = []
    dwell_start: float | None = None
    fired = False
    last_event_t = -999.0
    for pose in path:
        near = distance((pose.x, pose.y), hub) <= HUB_PARK_RADIUS
        parked = near and pose.speed <= HUB_SPEED_IN_S
        if parked:
            dwell_start = pose.t if dwell_start is None else dwell_start
            if (
                not fired
                and pose.t - dwell_start >= HUB_DWELL_S
                and pose.t - last_event_t >= HUB_COOLDOWN_S
            ):
                confidence = 0.55 if pose.t < AUTO_END_S else 0.48
                if pose.t >= ENDGAME_START_S:
                    confidence = 0.42
                events.append(
                    ScoutEvent(
                        team=pose.team,
                        type="hub_score_candidate",
                        t=round(dwell_start, 2),
                        zone=f"{alliance}_hub",
                        confidence=confidence,
                        detail="Low-speed dwell next to own Hub (heuristic, not an official score).",
                        period=period_name(pose.t),
                    )
                )
                last_event_t = pose.t
                fired = True
        elif not near:
            dwell_start = None
            fired = False
    return events


def _climb_events(path: list[Pose]) -> list[ScoutEvent]:
    tower = own_tower(path[0].alliance) if path else (0, 0)
    dwell_start: float | None = None
    for pose in path:
        if pose.t < ENDGAME_START_S:
            continue
        near = distance((pose.x, pose.y), tower) <= TOWER_RADIUS and pose.speed <= CLIMB_SPEED_IN_S
        if near:
            dwell_start = pose.t if dwell_start is None else dwell_start
            if pose.t - dwell_start >= CLIMB_DWELL_S:
                return [
                    ScoutEvent(
                        team=pose.team,
                        type="climb_attempt",
                        t=round(dwell_start, 2),
                        zone=f"{pose.alliance}_tower",
                        confidence=0.5,
                        detail="Stationary in the Tower footprint during endgame. Rung level cannot be read from a wide shot.",
                        period="endgame",
                    )
                ]
        else:
            dwell_start = None
    return []


def _defense_events(path: list[Pose], all_poses: list[Pose]) -> list[ScoutEvent]:
    others = [p for p in all_poses if p.team != path[0].team and p.alliance != path[0].alliance]
    by_t: dict[int, list[Pose]] = defaultdict(list)
    for pose in others:
        by_t[int(pose.t * 10)].append(pose)

    events: list[ScoutEvent] = []
    run_start: float | None = None
    last_pose: Pose | None = None
    for pose in path:
        if pose.t < AUTO_END_S or pose.t > MATCH_END_S:
            run_start = None
            continue
        nearby = False
        bucket = by_t.get(int(pose.t * 10), [])
        for opp in bucket:
            if distance((pose.x, pose.y), (opp.x, opp.y)) <= DEFENSE_PROXIMITY:
                nearby = True
                break
        defending = on_opponent_half(pose.x, pose.alliance) and (nearby or pose.speed < 25)
        if defending:
            run_start = pose.t if run_start is None else run_start
            last_pose = pose
        elif run_start is not None and last_pose is not None:
            duration = last_pose.t - run_start
            if duration >= DEFENSE_DWELL_S:
                events.append(
                    ScoutEvent(
                        team=pose.team,
                        type="defense",
                        t=round(run_start, 2),
                        zone=zone_name(last_pose.x, last_pose.y),
                        confidence=0.4,
                        detail=f"Opponent-half pressure for {duration:.1f}s.",
                        period=period_name(run_start),
                        duration_s=round(duration, 2),
                    )
                )
            run_start = None
    return events


def _collect_events(path: list[Pose]) -> list[ScoutEvent]:
    """Mark long neutral-zone collection stretches once per robot."""
    start: float | None = None
    last: Pose | None = None
    for pose in path:
        if zone_name(pose.x, pose.y) == "neutral" and pose.t >= AUTO_END_S:
            start = pose.t if start is None else start
            last = pose
        elif start is not None and last is not None:
            if last.t - start >= 6.0:
                return [
                    ScoutEvent(
                        team=pose.team,
                        type="collect",
                        t=round(start, 2),
                        zone="neutral",
                        confidence=0.36,
                        detail="Extended time in the Neutral Zone (fuel collection candidate).",
                        period="teleop",
                    )
                ]
            start = None
    return []
