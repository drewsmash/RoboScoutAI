"""Persistent robot identity records — one physical robot, one stable ID.

Separates detection / tracklet / identity / team / alliance so the UI, field
map, and exports all read the same source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Alliance = Literal["red", "blue", "unknown"]
TrackState = Literal["tentative", "confirmed", "occluded", "lost", "retired"]
AssignmentSource = Literal["", "ocr", "user", "start_pose_unverified", "gallery", "cloud"]


@dataclass
class RobotIdentity:
    """Match-level identity that survives occlusion and association repairs."""

    identity_id: int
    alliance: Alliance = "unknown"
    alliance_conf: float = 0.0
    team: str = ""
    assignment_source: AssignmentSource = ""
    state: TrackState = "tentative"
    hits: int = 0
    last_t: float = 0.0
    first_t: float = 0.0
    color_votes: list[tuple[str, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    # Association-error flags (contradictory bumper color, etc.) — never auto-recolor.
    flags: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "identity_id": int(self.identity_id),
            "alliance": self.alliance if self.alliance in {"red", "blue", "unknown"} else "unknown",
            "alliance_conf": round(float(self.alliance_conf), 3),
            "team": str(self.team or ""),
            "assignment_source": self.assignment_source,
            "assignment_verified": self.assignment_source in {"ocr", "user", "gallery"},
            "state": self.state,
            "hits": int(self.hits),
            "last_t": round(float(self.last_t), 3),
            "first_t": round(float(self.first_t), 3),
            "flags": list(self.flags),
            "notes": list(self.notes),
        }


class IdentityBook:
    """Owns the mapping from local MOT tracklets → persistent identities."""

    def __init__(self) -> None:
        self._idents: dict[int, RobotIdentity] = {}
        self._tracklet_to_id: dict[int, int] = {}
        self._next_id = 1

    def bind(self, tracklet_id: int, *, t: float) -> RobotIdentity:
        tid = int(tracklet_id)
        if tid in self._tracklet_to_id:
            ident = self._idents[self._tracklet_to_id[tid]]
            ident.last_t = float(t)
            ident.hits += 1
            if ident.state in {"lost", "occluded"}:
                ident.state = "confirmed" if ident.hits >= 5 else "tentative"
            return ident
        iid = self._next_id
        self._next_id += 1
        ident = RobotIdentity(identity_id=iid, first_t=float(t), last_t=float(t), hits=1)
        self._idents[iid] = ident
        self._tracklet_to_id[tid] = iid
        return ident

    def observe_alliance(self, identity_id: int, label: str, confidence: float) -> RobotIdentity:
        """Accumulate bumper votes. Confirmed alliances never flip from ordinary evidence."""
        ident = self._idents[int(identity_id)]
        if label not in {"red", "blue"}:
            return ident
        conf = float(np_clip(confidence, 0.0, 1.0))
        ident.color_votes.append((label, conf))
        if len(ident.color_votes) > 60:
            ident.color_votes = ident.color_votes[-60:]

        if ident.alliance in {"red", "blue"} and ident.alliance_conf >= 0.55 and ident.hits >= 8:
            # Confirmed: contradictory color → flag, do not recolor or split ID.
            if label != ident.alliance and conf >= 0.55:
                flag = "alliance_conflict"
                if flag not in ident.flags:
                    ident.flags.append(flag)
                    ident.notes.append(f"bumper vote {label}@{conf:.2f} contradicts confirmed {ident.alliance}")
            return ident

        # Soft majority for unconfirmed identities.
        scores = {"red": 0.0, "blue": 0.0}
        for lab, c in ident.color_votes[-20:]:
            scores[lab] = scores.get(lab, 0.0) + c
        winner = max(scores, key=scores.get)
        total = scores["red"] + scores["blue"]
        if total >= 1.2 and scores[winner] / max(total, 1e-6) >= 0.6:
            ident.alliance = winner  # type: ignore[assignment]
            ident.alliance_conf = float(scores[winner] / total)
            if ident.hits >= 5 and ident.alliance_conf >= 0.55:
                ident.state = "confirmed"
        return ident

    def set_team(self, identity_id: int, team: str, *, source: AssignmentSource) -> RobotIdentity:
        ident = self._idents[int(identity_id)]
        ident.team = str(team or "")
        ident.assignment_source = source
        return ident

    def mark_occluded(self, identity_id: int, t: float) -> None:
        ident = self._idents.get(int(identity_id))
        if ident is None:
            return
        ident.state = "occluded"
        ident.last_t = float(t)

    def as_list(self) -> list[dict[str, Any]]:
        return [i.as_dict() for i in sorted(self._idents.values(), key=lambda r: r.identity_id)]

    def identity_for_tracklet(self, tracklet_id: int) -> RobotIdentity | None:
        iid = self._tracklet_to_id.get(int(tracklet_id))
        return self._idents.get(iid) if iid is not None else None


def np_clip(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def alliance_css(alliance: str) -> str:
    """Canonical CSS modifier: red | blue | unknown (never coerce unknown→blue)."""
    return alliance if alliance in {"red", "blue"} else "unknown"
