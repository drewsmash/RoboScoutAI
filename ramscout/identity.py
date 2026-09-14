"""Alliance color from bumpers and team-number assignment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from ramscout.field import FIELD_LENGTH


def bumper_alliance(crop_bgr: np.ndarray) -> str:
    """Classify a robot crop as red / blue / unknown from bumper hue."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red_hi = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))
    red_count = int(np.count_nonzero(red_lo | red_hi))
    blue_count = int(np.count_nonzero(blue))
    if red_count > blue_count * 1.25 and red_count > 40:
        return "red"
    if blue_count > red_count * 1.25 and blue_count > 40:
        return "blue"
    return "unknown"


def ocr_digits(crop_bgr: np.ndarray) -> str:
    """Best-effort bumper OCR. Returns digits only; empty on failure."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = ""
    if hasattr(cv2, "text") and hasattr(cv2.text, "OCRTesseract_create"):
        try:
            ocr = cv2.text.OCRTesseract_create()
            text = ocr.run(th, 0)
        except Exception:  # noqa: BLE001
            text = ""
    digits = "".join(ch for ch in (text or "") if ch.isdigit())
    return digits


def constrain_to_teams(raw: str, team_numbers: Iterable[int | str]) -> str | None:
    """Map OCR digits onto the six match teams. Prefer an exact match."""
    allowed = {str(n) for n in team_numbers}
    if not raw:
        return None
    if raw in allowed:
        return raw
    hits = [team for team in allowed if team in raw]
    if len(hits) == 1:
        return hits[0]
    return None


def assign_by_start(
    tracks: list[dict[str, Any]],
    blue_teams: list[str],
    red_teams: list[str],
) -> dict[int, str]:
    """Map tracker ids to TBA teams using first-seen field pose.

    Blue robots are sorted by y (station 1–3), same for red. This is a
    starting guess — the UI lets a scout swap any assignment.
    """
    tracks = keep_top_tracks(tracks, max_tracks=max(len(blue_teams) + len(red_teams), 6) or 6)
    first: dict[int, dict[str, Any]] = {}
    for sample in tracks:
        tid = int(sample["track_id"])
        if tid not in first:
            first[tid] = sample

    blue_ids: list[tuple[float, int]] = []
    red_ids: list[tuple[float, int]] = []
    for tid, sample in first.items():
        alliance = sample.get("alliance") if sample.get("alliance") in {"red", "blue"} else None
        if alliance is None:
            alliance = "blue" if float(sample["x"]) < FIELD_LENGTH / 2 else "red"
        if alliance == "blue":
            blue_ids.append((float(sample["y"]), tid))
        else:
            red_ids.append((float(sample["y"]), tid))

    # If one side is empty/overfull, rebucket by field half using all first poses.
    if (len(blue_ids) < min(3, len(blue_teams)) or len(red_ids) < min(3, len(red_teams))) and first:
        blue_ids, red_ids = [], []
        for tid, sample in first.items():
            if float(sample["x"]) < FIELD_LENGTH / 2:
                blue_ids.append((float(sample["y"]), tid))
            else:
                red_ids.append((float(sample["y"]), tid))

    blue_ids.sort()
    red_ids.sort()
    mapping: dict[int, str] = {}
    for i, (_, tid) in enumerate(blue_ids[: len(blue_teams)]):
        mapping[tid] = str(blue_teams[i])
    for i, (_, tid) in enumerate(red_ids[: len(red_teams)]):
        mapping[tid] = str(red_teams[i])
    return mapping


def keep_top_tracks(samples: list[dict[str, Any]], max_tracks: int = 6) -> list[dict[str, Any]]:
    """Keep the longest-lived tracks so fragmented motion IDs do not flood scouting."""
    if not samples or max_tracks <= 0:
        return samples
    by_id: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_id.setdefault(int(sample["track_id"]), []).append(sample)
    if len(by_id) <= max_tracks:
        return samples

    def score(group: list[dict[str, Any]]) -> tuple[int, float]:
        times = [float(s.get("t") or 0.0) for s in group]
        span = (max(times) - min(times)) if times else 0.0
        return (len(group), span)

    ranked = sorted(by_id.items(), key=lambda item: score(item[1]), reverse=True)
    keep = {tid for tid, _ in ranked[:max_tracks]}
    return [s for s in samples if int(s["track_id"]) in keep]


def majority_alliance(samples: list[dict[str, Any]]) -> str:
    votes = Counter(s.get("alliance") for s in samples if s.get("alliance") in {"red", "blue"})
    if not votes:
        xs = [float(s["x"]) for s in samples]
        return "blue" if (sum(xs) / max(len(xs), 1)) < FIELD_LENGTH / 2 else "red"
    return votes.most_common(1)[0][0]


def linear_assignment(cost: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian assignment; greedy fallback if SciPy is missing."""
    if cost.size == 0:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist()))
    except Exception:  # noqa: BLE001
        pairs: list[tuple[int, int]] = []
        used_r: set[int] = set()
        used_c: set[int] = set()
        order = sorted(
            ((float(cost[i, j]), i, j) for i in range(cost.shape[0]) for j in range(cost.shape[1])),
        )
        for _, i, j in order:
            if i in used_r or j in used_c:
                continue
            used_r.add(i)
            used_c.add(j)
            pairs.append((i, j))
        return pairs


def stitch_occlusions(
    samples: list[dict[str, Any]],
    max_gap_s: float = 2.5,
    max_dist_in: float = 56.0,
) -> list[dict[str, Any]]:
    """Re-attach ByteTrack IDs that likely belong to the same robot after a gap."""
    if len(samples) < 2:
        return samples

    by_id: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_id.setdefault(int(sample["track_id"]), []).append(sample)
    for group in by_id.values():
        group.sort(key=lambda row: float(row["t"]))

    fragments: list[dict[str, Any]] = []
    for tid, group in by_id.items():
        fragments.append(
            {
                "tid": tid,
                "alliance": majority_alliance(group),
                "t0": float(group[0]["t"]),
                "t1": float(group[-1]["t"]),
                "start": (float(group[0]["x"]), float(group[0]["y"])),
                "end": (float(group[-1]["x"]), float(group[-1]["y"])),
            }
        )
    fragments.sort(key=lambda item: item["t0"])

    remap = {item["tid"]: item["tid"] for item in fragments}
    claimed: set[int] = set()
    for i, earlier in enumerate(fragments):
        costs: list[tuple[float, int]] = []
        for later in fragments[i + 1 :]:
            if later["tid"] in claimed:
                continue
            if (
                earlier["alliance"] in {"red", "blue"}
                and later["alliance"] in {"red", "blue"}
                and earlier["alliance"] != later["alliance"]
            ):
                continue
            gap = later["t0"] - earlier["t1"]
            if gap < 0 or gap > max_gap_s:
                continue
            dist = float(np.hypot(later["start"][0] - earlier["end"][0], later["start"][1] - earlier["end"][1]))
            if dist > max_dist_in:
                continue
            costs.append((dist + gap * 10.0, later["tid"]))
        if not costs:
            continue
        cost = np.array([[c[0] for c in costs]], dtype=float)
        pairs = linear_assignment(cost)
        if not pairs:
            continue
        _, col = pairs[0]
        chosen = costs[col][1]
        root = remap[earlier["tid"]]
        remap[chosen] = root
        claimed.add(chosen)
        for tid, mapped in list(remap.items()):
            if mapped == chosen:
                remap[tid] = root

    out: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        row["track_id"] = remap.get(int(sample["track_id"]), sample["track_id"])
        out.append(row)
    return out
