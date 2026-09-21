"""Cross-view correlation: link overview tracks with side / alt-pane detections.

Each broadcast pane is detected independently. This module stitches them into
shared identities so side-camera scoring/climb cues attach to the right team
instead of inventing alliance-level soft events.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ViewDetection:
    t: float
    role: str
    alliance: str
    x_norm: float
    y_norm: float
    confidence: float
    bbox: list[float] = field(default_factory=list)  # absolute frame pixels
    team: str = ""
    track_id: int | None = None
    source: str = "pane"

    def as_dict(self) -> dict[str, Any]:
        return {
            "t": round(self.t, 3),
            "role": self.role,
            "alliance": self.alliance if self.alliance in {"red", "blue"} else "",
            "x_norm": round(self.x_norm, 3),
            "y_norm": round(self.y_norm, 3),
            "confidence": round(self.confidence, 3),
            "bbox": [round(float(v), 1) for v in self.bbox[:4]],
            "team": self.team,
            "track_id": self.track_id,
            "source": self.source,
        }


def _alliance(sample: dict[str, Any]) -> str:
    a = str(sample.get("alliance") or "").lower()
    return a if a in {"red", "blue"} else ""


def samples_to_detections(samples: list[dict[str, Any]], *, role: str = "overview") -> list[ViewDetection]:
    out: list[ViewDetection] = []
    for s in samples:
        bbox = s.get("bbox") or s.get("box") or []
        if isinstance(bbox, dict):
            bbox = [bbox.get("x0", 0), bbox.get("y0", 0), bbox.get("x1", 0), bbox.get("y1", 0)]
        out.append(
            ViewDetection(
                t=float(s.get("t") or 0.0),
                role=str(s.get("view") or role),
                alliance=_alliance(s),
                x_norm=float(s.get("x_norm") or s.get("x") or 0.5),
                y_norm=float(s.get("y_norm") or s.get("y") or 0.5),
                confidence=float(s.get("confidence") or s.get("score") or 0.5),
                bbox=[float(v) for v in bbox[:4]] if bbox else [],
                team=str(s.get("team") or ""),
                track_id=int(s["track_id"]) if s.get("track_id") is not None else None,
                source=str(s.get("source") or "overview"),
            )
        )
    return out


def correlate_views(
    overview: list[dict[str, Any]],
    pane_dets: list[dict[str, Any]],
    *,
    time_tol: float = 1.25,
) -> dict[str, Any]:
    """Match side/alt detections to overview track IDs by time + alliance + position.

    Returns links, enriched pane detections, and team-attributed side cues.
    """
    ov = samples_to_detections(overview, role="overview")
    panes = [
        ViewDetection(
            t=float(d.get("t") or 0.0),
            role=str(d.get("role") or d.get("view") or "side"),
            alliance=str(d.get("alliance") or ""),
            x_norm=float(d.get("x_norm") or 0.5),
            y_norm=float(d.get("y_norm") or 0.5),
            confidence=float(d.get("confidence") or 0.4),
            bbox=list(d.get("bbox") or [])[:4],
            team=str(d.get("team") or ""),
            track_id=int(d["track_id"]) if d.get("track_id") is not None else None,
            source=str(d.get("source") or "pane"),
        )
        for d in pane_dets
    ]

    # Index overview samples by coarse time bins for O(n) matching.
    bins: dict[int, list[ViewDetection]] = defaultdict(list)
    for det in ov:
        bins[int(det.t * 2)].append(det)  # 0.5 s bins

    links: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []
    for det in panes:
        best: ViewDetection | None = None
        best_score = -1.0
        for b in range(int(det.t * 2) - 3, int(det.t * 2) + 4):
            for cand in bins.get(b, []):
                dt = abs(cand.t - det.t)
                if dt > time_tol:
                    continue
                score = 1.0 - dt / time_tol
                if det.alliance and cand.alliance:
                    if det.alliance != cand.alliance:
                        continue
                    score += 0.35
                elif det.alliance or cand.alliance:
                    score += 0.05
                # Side cams: blue pane prefers left-ish overview X, red right-ish.
                if det.role.startswith("blue") and cand.x_norm <= 0.62:
                    score += 0.15
                if det.role.startswith("red") and cand.x_norm >= 0.38:
                    score += 0.15
                # overview_alt: prefer similar normalized position.
                if "alt" in det.role:
                    score += 0.2 * (1.0 - min(1.0, abs(cand.x_norm - det.x_norm) + abs(cand.y_norm - det.y_norm)))
                score += 0.1 * min(cand.confidence, 1.0)
                if score > best_score:
                    best_score = score
                    best = cand
        row = det.as_dict()
        if best is not None and best_score >= 0.45:
            row["track_id"] = best.track_id
            row["team"] = best.team or row.get("team") or ""
            row["alliance"] = row.get("alliance") or best.alliance
            row["match_score"] = round(best_score, 3)
            links.append(
                {
                    "t": row["t"],
                    "role": row["role"],
                    "track_id": best.track_id,
                    "team": row["team"],
                    "alliance": row["alliance"],
                    "score": row["match_score"],
                }
            )
        enriched.append(row)

    # Build team-attributed cues from linked pane detections.
    cues: list[dict[str, Any]] = []
    for row in enriched:
        if not row.get("track_id") and not row.get("team"):
            continue
        role = str(row.get("role") or "")
        kind = "motion"
        if "side" in role or role in {"blue_side", "red_side", "sideline"}:
            # Upper third of a side crop → climb/tower; mid → hub.
            if float(row.get("y_norm") or 0.5) < 0.38:
                kind = "climb_activity"
            elif float(row.get("y_norm") or 0.5) < 0.78:
                kind = "hub_activity"
        cues.append(
            {
                "t": row["t"],
                "alliance": row.get("alliance") or "",
                "kind": kind,
                "confidence": row.get("confidence") or 0.4,
                "detail": f"correlated {role} → track {row.get('track_id')}",
                "role": role,
                "x_norm": row.get("x_norm"),
                "y_norm": row.get("y_norm"),
                "team": row.get("team") or "",
                "track_id": row.get("track_id"),
                "source": "correlated_pane",
            }
        )

    matched = sum(1 for r in enriched if r.get("track_id") is not None)
    return {
        "links": links,
        "pane_detections": enriched,
        "side_cues": cues,
        "stats": {
            "overview": len(ov),
            "pane_detections": len(panes),
            "matched": matched,
            "match_rate": round(matched / max(1, len(panes)), 3),
        },
    }


def attach_teams_to_cues(
    cues: list[dict[str, Any]],
    assignments: dict[str, str],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Fill missing team fields on cues using track_id → assignment."""
    team_by_tid = {str(k): str(v) for k, v in assignments.items()}
    for s in samples:
        tid = str(s.get("track_id"))
        if s.get("team") and tid:
            team_by_tid.setdefault(tid, str(s["team"]))
    out = []
    for cue in cues:
        row = dict(cue)
        tid = row.get("track_id")
        if tid is not None and not row.get("team"):
            row["team"] = team_by_tid.get(str(tid), "")
        out.append(row)
    return out
