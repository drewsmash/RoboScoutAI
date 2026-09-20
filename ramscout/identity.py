"""Alliance color from bumpers and team-number assignment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from ramscout.field import ALLIANCE_DEPTH, FIELD_LENGTH, FIELD_WIDTH


def bumper_alliance(crop_bgr: np.ndarray, *, band_frac: float = 0.42) -> str:
    """Classify a robot crop as red / blue / unknown from bumper hue.

    Samples only the lower band of the crop (where bumpers are) and requires
    a clear saturated-hue majority; jerseys / carpet tape above the bumper
    line no longer vote.
    """
    from ramscout.trackers.alliance import bumper_band, chroma_features, hsv_alliance

    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    h, w = crop_bgr.shape[:2]
    band = bumper_band(crop_bgr, [0, 0, w, h], frac=band_frac)
    label, _conf = hsv_alliance(chroma_features(band if band is not None else crop_bgr))
    return label


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


def track_quality(group: list[dict[str, Any]]) -> float:
    """Score a track: long-lived, actually moving, and inside the field.

    Static blobs (walls, field elements) and perimeter clutter score low even
    when they persist for the whole match; a real robot path scores high.
    """
    if not group:
        return 0.0
    rows = sorted(group, key=lambda s: float(s.get("t") or 0.0))
    times = [float(s.get("t") or 0.0) for s in rows]
    span = max(times) - min(times) if len(times) > 1 else 0.0
    xs = [float(s.get("x") or 0.0) for s in rows]
    ys = [float(s.get("y") or 0.0) for s in rows]
    path = 0.0
    for i in range(1, len(rows)):
        path += float(np.hypot(xs[i] - xs[i - 1], ys[i] - ys[i - 1]))
    extent = 0.0
    if len(rows) > 1:
        extent = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
    # Fraction of samples comfortably inside the field (not on the perimeter).
    margin = 12.0
    inside = sum(
        1 for x, y in zip(xs, ys) if margin <= x <= FIELD_LENGTH - margin and margin <= y <= FIELD_WIDTH - margin
    )
    in_field = inside / max(len(rows), 1)
    # Movement factor: 0.35 for a never-moving blob → 1.0 once it has covered
    # a few robot-lengths of field.
    move = 0.35 + 0.65 * min(1.0, (0.5 * path + extent) / 240.0)
    return float(len(rows) * (0.5 + 0.5 * in_field) * move + 0.5 * span)


def keep_top_tracks(samples: list[dict[str, Any]], max_tracks: int = 6) -> list[dict[str, Any]]:
    """Keep the best tracks (long-lived, moving, in-field) so fragments and
    static clutter do not flood scouting."""
    if not samples or max_tracks <= 0:
        return samples
    by_id: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_id.setdefault(int(sample["track_id"]), []).append(sample)
    if len(by_id) <= max_tracks:
        return samples

    ranked = sorted(by_id.items(), key=lambda item: (track_quality(item[1]), len(item[1])), reverse=True)
    keep = {tid for tid, _ in ranked[:max_tracks]}
    return [s for s in samples if int(s["track_id"]) in keep]


def balance_alliances(
    samples: list[dict[str, Any]],
    *,
    n_red: int = 3,
    n_blue: int = 3,
    early_s: float = 30.0,
    min_tracks: int = 2,
) -> list[dict[str, Any]]:
    """Force exactly n_red / n_blue alliance labels across the strongest tracks.

    Per-track color evidence (weighted alliance votes) is combined with the
    starting-side prior and solved as a minimum-cost assignment onto 3 red +
    3 blue slots. Tracks beyond the slot count keep their own majority label.
    Returns new sample dicts; ``alliance_conf`` is attached per sample.
    """
    from ramscout.trackers.alliance import assign_alliance_slots, side_prior

    if not samples:
        return samples
    by_id: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        by_id.setdefault(int(sample["track_id"]), []).append(sample)
    if len(by_id) < min_tracks:
        return samples

    ranked = sorted(by_id.items(), key=lambda item: track_quality(item[1]), reverse=True)
    slots = n_red + n_blue
    assignable = ranked[:slots]

    p_color: list[float] = []
    p_prior: list[float] = []
    for _tid, group in assignable:
        red_w = blue_w = 0.0
        for s in group:
            label = s.get("alliance")
            if label not in {"red", "blue"}:
                continue
            w = float(s.get("alliance_conf") or 0.6)
            if label == "red":
                red_w += w
            else:
                blue_w += w
        total = red_w + blue_w
        p_color.append(red_w / total if total > 0 else 0.5)
        first = min(group, key=lambda s: float(s.get("t") or 0.0))
        label, weight = side_prior(
            float(first.get("x") or 0.0),
            float(first.get("t") or 0.0),
            field_length=FIELD_LENGTH,
            alliance_depth=ALLIANCE_DEPTH,
            early_s=early_s,
        )
        if label == "red":
            p_prior.append(0.5 + 0.5 * weight)
        elif label == "blue":
            p_prior.append(0.5 - 0.5 * weight)
        else:
            p_prior.append(0.5)

    labels = assign_alliance_slots(p_color, n_red=n_red, n_blue=n_blue, prior_red=p_prior, prior_weight=0.3)
    forced: dict[int, tuple[str, float]] = {}
    for (tid, _group), label, pc, pp in zip(assignable, labels, p_color, p_prior):
        p = 0.7 * pc + 0.3 * pp
        conf = p if label == "red" else 1.0 - p
        forced[tid] = (label, float(np.clip(conf, 0.0, 1.0)))

    out: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        tid = int(row["track_id"])
        if tid in forced:
            row["alliance"], row["alliance_conf"] = forced[tid]
        elif row.get("alliance") not in {"red", "blue"}:
            row["alliance"] = majority_alliance(by_id[tid])
        out.append(row)
    return out


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
    max_gap_s: float = 3.0,
    max_dist_in: float = 72.0,
) -> list[dict[str, Any]]:
    """Re-attach track IDs that likely belong to the same robot after a gap.

    Uses Hungarian matching on (distance + time gap + alliance mismatch).
    Wider defaults than before so SORT fragments from MOT still stitch.
    """
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
        if remap[earlier["tid"]] != earlier["tid"]:
            continue
        costs: list[tuple[float, int]] = []
        for later in fragments[i + 1 :]:
            if later["tid"] in claimed:
                continue
            if remap.get(later["tid"]) != later["tid"]:
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
            costs.append((dist + gap * 8.0, later["tid"]))
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
