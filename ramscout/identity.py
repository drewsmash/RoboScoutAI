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
    unknown_ids: list[tuple[float, int]] = []
    for tid, sample in first.items():
        alliance = sample.get("alliance") if sample.get("alliance") in {"red", "blue"} else None
        if alliance == "blue":
            blue_ids.append((float(sample["y"]), tid))
        elif alliance == "red":
            red_ids.append((float(sample["y"]), tid))
        else:
            # Keep unknown as unknown — do NOT invent alliance from field half.
            unknown_ids.append((float(sample["y"]), tid))

    # Do not soft-fill unknowns into red/blue slots from x-position. Roster
    # slots remain empty until bumper/OCR/user evidence assigns a team.
    # Unknown tracklets stay unmapped so the UI can show gray + "Correct team".
    _ = unknown_ids  # retained for callers / future gallery matching

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


def assemble_lanes(
    samples: list[dict[str, Any]],
    *,
    n_red: int = 3,
    n_blue: int = 3,
    overlap_s: float = 0.5,
    min_fragment_s: float = 0.6,
    max_dist_in: float = 48.0,
    speed_in_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Assemble tracklets into at most ``n_red + n_blue`` robot lanes.

    A match has exactly six robots, so every fragment that is *time-disjoint*
    from a lane of its alliance belongs to one of them; the cheapest
    (distance-plausible) lane wins, a jump farther than the robot could have
    driven during the gap is still accepted (marked ``lane_jump`` on the first
    sample) because coverage matters more than a rare wrong link. Fragments
    that overlap every lane of their alliance are a seventh simultaneous
    object — a plate, a referee, a cube — and are dropped.
    """
    if not samples:
        return samples
    by_id: dict[int, list[dict[str, Any]]] = {}
    for s in samples:
        by_id.setdefault(int(s["track_id"]), []).append(s)
    frags: list[dict[str, Any]] = []
    for tid, group in by_id.items():
        group.sort(key=lambda r: float(r["t"]))
        frags.append(
            {
                "tid": tid,
                "alliance": majority_alliance(group),
                "t0": float(group[0]["t"]),
                "t1": float(group[-1]["t"]),
                "start": (float(group[0]["x"]), float(group[0]["y"])),
                "end": (float(group[-1]["x"]), float(group[-1]["y"])),
                "dur": float(group[-1]["t"]) - float(group[0]["t"]),
                "n": len(group),
            }
        )
    # Long fragments first within the same start second so a solid track
    # claims the lane before a blip does.
    frags.sort(key=lambda f: (round(f["t0"], 0), -f["dur"]))
    quota = {"red": n_red, "blue": n_blue}
    lanes: list[dict[str, Any]] = []
    remap: dict[int, int] = {}
    cut_after: dict[int, float] = {}
    jump_at: dict[int, float] = {}
    dropped: set[int] = set()
    for f in frags:
        if f["dur"] < min_fragment_s and f["n"] < 4:
            dropped.add(f["tid"])
            continue
        allowed = [f["alliance"]] if f["alliance"] in quota else list(quota)
        free = [ln for ln in lanes if ln["alliance"] in allowed and ln["t1"] <= f["t0"] + overlap_s]
        if free:
            def cost(ln: dict[str, Any]) -> float:
                gap = max(f["t0"] - ln["t1"], 0.0)
                dist = float(np.hypot(f["start"][0] - ln["end"][0], f["start"][1] - ln["end"][1]))
                allow = max_dist_in + speed_in_s * gap
                return dist / max(allow, 1.0) + gap / 30.0

            lane = min(free, key=cost)
            gap = max(f["t0"] - lane["t1"], 0.0)
            dist = float(np.hypot(f["start"][0] - lane["end"][0], f["start"][1] - lane["end"][1]))
            if dist > max_dist_in + speed_in_s * gap:
                jump_at[f["tid"]] = f["t0"]
            if f["t0"] < lane["t1"]:
                cut_after[lane["last_tid"]] = f["t0"]
            remap[f["tid"]] = lane["root"]
            lane.update(t1=f["t1"], end=f["end"], last_tid=f["tid"])
            if lane["alliance"] not in quota and f["alliance"] in quota:
                lane["alliance"] = f["alliance"]
            continue
        # New lane if the alliance still has room.
        opened = False
        for alliance in allowed:
            used = sum(1 for ln in lanes if ln["alliance"] == alliance)
            if used < quota[alliance]:
                lanes.append({"alliance": alliance, "root": f["tid"], "t1": f["t1"], "end": f["end"], "last_tid": f["tid"]})
                remap[f["tid"]] = f["tid"]
                opened = True
                break
        if not opened:
            dropped.add(f["tid"])
    out: list[dict[str, Any]] = []
    for s in samples:
        tid = int(s["track_id"])
        if tid in dropped or tid not in remap:
            continue
        cut = cut_after.get(tid)
        if cut is not None and float(s["t"]) >= cut:
            continue
        row = dict(s)
        row["track_id"] = remap[tid]
        if jump_at.get(tid) is not None and float(s["t"]) == jump_at[tid]:
            row["lane_jump"] = True
        out.append(row)
    return out


def stitch_occlusions(
    samples: list[dict[str, Any]],
    max_gap_s: float = 3.0,
    max_dist_in: float = 48.0,
    *,
    overlap_s: float = 0.6,
    speed_in_s: float = 60.0,
) -> list[dict[str, Any]]:
    """Link tracklets that belong to the same robot into one path (chains).

    Fragments are sorted by start time; each chain end greedily claims the
    best later fragment by (distance − plausible travel) + time gap, with an
    alliance mismatch forbidden. Fragments may overlap the chain end by up to
    ``overlap_s`` (a new ID often spawns a few frames before the old one
    dies). Distance allowance grows with the gap at ``speed_in_s``.
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
                "n": len(group),
            }
        )
    fragments.sort(key=lambda item: item["t0"])
    by_tid = {f["tid"]: f for f in fragments}

    # chain root → (current end fragment)
    root_of: dict[int, int] = {f["tid"]: f["tid"] for f in fragments}
    chain_end: dict[int, dict[str, Any]] = {f["tid"]: f for f in fragments}
    claimed: set[int] = set()
    cut_after: dict[int, float] = {}  # fragment tid → drop its samples at/after this t
    for frag in fragments:
        if frag["tid"] in claimed:
            continue
        # Extend the chain that ends with ``frag`` (possibly just itself).
        root = root_of[frag["tid"]]
        end = chain_end[root]
        while True:
            costs: list[tuple[float, int]] = []
            for later in fragments:
                lt = later["tid"]
                if lt in claimed or root_of[lt] != lt or lt == root or lt == end["tid"]:
                    continue
                if (
                    end["alliance"] in {"red", "blue"}
                    and later["alliance"] in {"red", "blue"}
                    and end["alliance"] != later["alliance"]
                ):
                    continue
                gap = later["t0"] - end["t1"]
                if gap < -overlap_s or gap > max_gap_s:
                    continue
                dist = float(np.hypot(later["start"][0] - end["end"][0], later["start"][1] - end["end"][1]))
                allow = max_dist_in + speed_in_s * max(gap, 0.0)
                if dist > allow:
                    continue
                costs.append((dist / max(allow, 1.0) + max(gap, 0.0) / max_gap_s, lt))
            if not costs:
                break
            costs.sort()
            chosen = costs[0][1]
            claimed.add(chosen)
            root_of[chosen] = root
            nxt = by_tid[chosen]
            if nxt["t0"] <= end["t1"]:
                cut_after[end["tid"]] = nxt["t0"]
            end = nxt
            chain_end[root] = end

    out: list[dict[str, Any]] = []
    for sample in samples:
        tid = int(sample["track_id"])
        cut = cut_after.get(tid)
        if cut is not None and float(sample["t"]) >= cut:
            continue
        row = dict(sample)
        row["track_id"] = root_of.get(tid, tid)
        out.append(row)
    return out
