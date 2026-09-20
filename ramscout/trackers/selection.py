"""Auto-benchmark tracker selection.

``tracker_mode="auto"`` no longer means "hybrid with a different label". The
selector runs every *available* strategy set on the first
``benchmark_s`` seconds of the match (same crop / homography / field gate as
the real run), scores each on physically meaningful criteria and hands the
winner to the full run. The choice and every score are recorded in the job
as ``tracker_selection`` so users can see *why* a mode was picked and override
it manually.

Scoring (each term in [0, 1], weighted):

- ``robots``   — how close the mean number of simultaneously visible robots
                 is to six (and how often ≥ 5 are visible)
- ``balance``  — 3 red / 3 blue among the strongest tracks
- ``persist``  — mean confirmed-track lifetime relative to the window
- ``in_field`` — fraction of samples well inside the carpet (not the
                 perimeter band)
- ``speed``    — fraction of steps under the 20 ft/s physical limit and a
                 penalty for a fast p95
- ``moving``   — fraction of samples on tracks that actually moved (a wall
                 that is emitted for 10 s scores zero here)
- ``stability``— few alliance flips and few new IDs per second

Cloud strategies are only benchmarked when a key is present; YOLO only when
``ultralytics`` imports. Benchmark runs never call bumper OCR / smoothing.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.trackers.ensemble import TRACKER_MODES, list_strategies

ProgressFn = Callable[[str, float], None]

MAX_SPEED_IN_S = 240.0  # 20 ft/s
WEIGHTS: dict[str, float] = {
    "robots": 0.22,
    "balance": 0.14,
    "persist": 0.16,
    "in_field": 0.12,
    "speed": 0.12,
    "moving": 0.14,
    "stability": 0.10,
}


def candidate_modes(
    *,
    model_path: str | None = None,
    openai_key: str = "",
    google_key: str = "",
    include_cloud: bool = True,
) -> list[str]:
    """Modes worth benchmarking given what is installed / keyed."""
    avail = {row["name"]: bool(row["available"]) for row in list_strategies(model_path)}
    yolo_ok = avail.get("yolo", False)
    modes = ["motion", "color", "potato", "hybrid"]
    if yolo_ok:
        modes.append("yolo")
    if include_cloud:
        if google_key:
            modes.append("gemini")
        if openai_key:
            modes.append("openai")
    # Deduplicate strategy sets that resolve identically without cloud/YOLO.
    seen: set[tuple[str, ...]] = set()
    out: list[str] = []
    for mode in modes:
        wanted = tuple(
            s
            for s in TRACKER_MODES[mode]["strategies"]
            if s in {"motion", "color", "optical_flow"}
            or (s == "yolo" and yolo_ok)
            or (s == "gemini" and google_key)
            or (s == "openai" and openai_key)
        )
        key = (mode if TRACKER_MODES[mode].get("cascade") else "",) + tuple(sorted(wanted))
        if key in seen:
            continue
        seen.add(key)
        out.append(mode)
    return out


def score_samples(samples: list[dict[str, Any]], *, window_s: float, dt_s: float) -> dict[str, float]:
    """Physically grounded quality score of a tracking result (0–1 each)."""
    if not samples:
        return {k: 0.0 for k in WEIGHTS} | {"total": 0.0, "tracks": 0, "samples": 0}
    by_track: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for s in samples:
        by_track[int(s["track_id"])].append(s)
    window_s = max(float(window_s), dt_s)

    lengths: list[float] = []
    red = blue = 0
    flips = 0
    fast_steps = 0
    steps = 0
    speeds: list[float] = []
    moving_tracks = 0
    in_field = 0
    for rows in by_track.values():
        rows.sort(key=lambda r: float(r["t"]))
        lengths.append(float(rows[-1]["t"]) - float(rows[0]["t"]) + dt_s)
        labels = [r["alliance"] for r in rows if r.get("alliance") in {"red", "blue"}]
        if labels:
            maj = max(set(labels), key=labels.count)
            if maj == "red":
                red += 1
            else:
                blue += 1
            flips += sum(1 for a, b in zip(labels, labels[1:]) if a != b)
        xs = np.array([float(r["x"]) for r in rows])
        ys = np.array([float(r["y"]) for r in rows])
        if len(rows) >= 2:
            d = np.hypot(np.diff(xs), np.diff(ys))
            dts = np.diff(np.array([float(r["t"]) for r in rows]))
            ok = dts > 1e-6
            v = d[ok] / dts[ok]
            speeds.extend(v.tolist())
            steps += int(ok.sum())
            fast_steps += int((v > MAX_SPEED_IN_S).sum())
            span = float(np.hypot(xs.max() - xs.min(), ys.max() - ys.min()))
            if span >= 12.0 or any(bool(r.get("speed_in_s", 0) > 6) for r in rows):
                moving_tracks += 1
        m = 8.0
        in_field += int(np.sum((xs > m) & (xs < FIELD_LENGTH - m) & (ys > m) & (ys < FIELD_WIDTH - m)))

    n_tracks = len(by_track)
    per_t: dict[float, set[int]] = defaultdict(set)
    for s in samples:
        per_t[round(float(s["t"]) / max(dt_s, 1e-3)) * dt_s].add(int(s["track_id"]))
    counts = np.array([len(v) for v in per_t.values()], dtype=float)
    # Time bins with zero robots count too (the window may have empty frames).
    n_bins = max(int(round(window_s / dt_s)), len(counts), 1)
    if len(counts) < n_bins:
        counts = np.concatenate([counts, np.zeros(n_bins - len(counts))])
    mean_visible = float(counts.mean())
    ge5 = float(np.mean(counts >= 5))

    robots = float(np.clip(1.0 - abs(mean_visible - 6.0) / 6.0, 0.0, 1.0)) * 0.6 + 0.4 * ge5
    balance = 1.0 - min(abs(red - 3) + abs(blue - 3), 6) / 6.0
    if n_tracks > 8:
        balance *= 0.7
    persist = float(np.clip(np.mean(lengths) / window_s, 0.0, 1.0)) if lengths else 0.0
    in_field_r = in_field / max(len(samples), 1)
    p95 = float(np.percentile(speeds, 95)) if speeds else 0.0
    speed = (1.0 - fast_steps / max(steps, 1)) * float(np.clip(1.0 - max(0.0, p95 - 180.0) / 240.0, 0.2, 1.0))
    moving = moving_tracks / max(n_tracks, 1)
    births_per_s = n_tracks / window_s
    stability = float(np.clip(1.0 - flips / max(len(samples) / 20.0, 1.0), 0.0, 1.0)) * float(
        np.clip(1.0 - max(0.0, births_per_s - 0.6) / 1.5, 0.0, 1.0)
    )
    parts = {
        "robots": round(robots, 3),
        "balance": round(balance, 3),
        "persist": round(persist, 3),
        "in_field": round(in_field_r, 3),
        "speed": round(speed, 3),
        "moving": round(moving, 3),
        "stability": round(stability, 3),
    }
    total = sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
    parts["total"] = round(float(total), 4)
    parts["tracks"] = n_tracks
    parts["samples"] = len(samples)
    parts["mean_visible"] = round(mean_visible, 2)
    parts["red_blue"] = [red, blue]
    parts["p95_speed_in_s"] = round(p95, 1)
    return parts


def select_tracker_mode(
    video_path: Path,
    *,
    homography: np.ndarray | None = None,
    src_points: list[list[float]] | None = None,
    model_path: str | None = None,
    team_numbers: list[str] | None = None,
    frame_stride: int = 4,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    crop_left: float = 0.0,
    crop_right: float = 1.0,
    layout: Any = None,
    openai_key: str = "",
    google_key: str = "",
    openai_model: str = "",
    google_model: str = "",
    use_bev: bool = True,
    prefer_depth_neural: bool = True,
    field_gate: bool = True,
    game_year: int | None = None,
    start_s: float | None = None,
    benchmark_s: float = 12.0,
    on_progress: ProgressFn | None = None,
    include_cloud: bool = True,
    candidates: list[str] | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Benchmark every available strategy set on a short window; pick the best.

    ``runner`` defaults to :func:`ramscout.detect._track_video_impl` (injected
    for tests). Returns ``{"chosen", "scores", "candidates", "window_s", ...}``.
    """
    if runner is None:
        from ramscout.detect import _track_video_impl as runner  # type: ignore[assignment]

    modes = candidates or candidate_modes(
        model_path=model_path, openai_key=openai_key, google_key=google_key, include_cloud=include_cloud
    )
    t0 = float(start_s or 0.0)
    # Skip the very start (pre-match graphics / robots parked) when the
    # window is not pinned by the caller.
    if start_s is None:
        t0 = 3.0
    t1 = t0 + float(benchmark_s)
    scores: dict[str, dict[str, Any]] = {}
    timings: dict[str, float] = {}
    shared_h = homography
    shared_pts = src_points
    for i, mode in enumerate(modes):
        if on_progress:
            on_progress(f"Auto mode: benchmarking '{mode}' ({i + 1}/{len(modes)})…", 100.0 * i / max(len(modes), 1))
        started = time.time()
        try:
            res = runner(
                video_path,
                homography=shared_h,
                src_points=shared_pts,
                model_path=model_path,
                team_numbers=team_numbers,
                frame_stride=frame_stride,
                max_frames=None,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                on_progress=None,
                tracker_mode=mode,
                openai_key=openai_key,
                google_key=google_key,
                openai_model=openai_model,
                google_model=google_model,
                use_bev=use_bev,
                prefer_depth_neural=prefer_depth_neural,
                depth_stride=12,
                field_gate=field_gate,
                game_year=game_year,
                alliance_calibration_s=benchmark_s,
                crop_left=crop_left,
                crop_right=crop_right,
                layout=layout,
                start_s=t0,
                end_s=t1,
                smooth=False,
                bumper_ocr=False,
                field_lines=True,
            )
        except Exception as exc:  # noqa: BLE001
            scores[mode] = {k: 0.0 for k in WEIGHTS} | {"total": 0.0, "error": str(exc)}
            timings[mode] = round(time.time() - started, 2)
            continue
        timings[mode] = round(time.time() - started, 2)
        # Reuse the first successful BEV calibration for every other
        # candidate so the comparison is about *tracking*, not geometry.
        if shared_h is None and res.get("homography") and res.get("src_points"):
            shared_h = np.asarray(res["homography"], dtype=np.float64)
            shared_pts = [list(p) for p in res["src_points"]]
        fps = float(res.get("fps") or 30.0)
        dt_s = max(frame_stride, 1) / fps
        window = res.get("window") or [t0, t1]
        sc = score_samples(res.get("samples") or [], window_s=float(window[1]) - float(window[0]), dt_s=dt_s)
        sc["source_hits"] = dict(res.get("source_hits") or {})
        sc["strategies"] = list(res.get("strategies") or [])
        scores[mode] = sc

    if not scores:
        chosen = "hybrid"
    else:
        # Ties (identical strategy sets) resolve toward the cheaper local mode
        # by preferring earlier candidates.
        chosen = max(modes, key=lambda m: (scores.get(m, {}).get("total", 0.0), -modes.index(m)))
        if scores[chosen].get("total", 0.0) <= 0.0:
            chosen = "hybrid" if "hybrid" in modes else modes[0]
    ranking = sorted(scores, key=lambda m: scores[m].get("total", 0.0), reverse=True)
    if on_progress:
        on_progress(f"Auto mode chose '{chosen}'.", 100.0)
    return {
        "chosen": chosen,
        "scores": scores,
        "candidates": modes,
        "ranking": ranking,
        "window_s": float(benchmark_s),
        "window": [round(t0, 3), round(t1, 3)],
        "timings_s": timings,
        "weights": dict(WEIGHTS),
        "homography": np.asarray(shared_h).tolist() if shared_h is not None else None,
        "src_points": shared_pts,
    }
