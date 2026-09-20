"""Robot detection and tracking on a recorded match video.

Orchestrates multiple strategies via ramscout.trackers:
- Local: motion, bumper color, optical flow
- Neural: Ultralytics YOLO (optional)
- Cloud: OpenAI Vision / Google Gemini (optional API keys)
- Depth / BEV: Depth Anything V2 (optional) + bird's-eye projection for camera angle

Paths are produced even when neural/cloud backends are missing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.field import ALLIANCE_DEPTH, FIELD_LENGTH
from ramscout.geometry import crop_bounds, default_source_points, homography_from_corners, project_points
from ramscout.identity import balance_alliances, constrain_to_teams, ocr_digits
from ramscout.trackers.alliance import (
    AllianceCalibrator,
    AllianceVoter,
    bumper_band,
    chroma_features,
    side_prior,
)
from ramscout.trackers.ensemble import TRACKER_MODES, EnsembleTracker, list_strategies, resolve_strategies
from ramscout.trackers.field_gate import FieldGate, scorebug_bands_for_year
from ramscout.trackers.types import TrackerContext
from ramscout.trackers.yolo import (
    ensure_detector_weights,
    find_local_model,
    load_detector,
    robot_class_ids,
    ultralytics_available,
)

ProgressFn = Callable[[str, float], None]

__all__ = [
    "TRACKER_MODES",
    "ensure_detector_weights",
    "find_local_model",
    "list_strategies",
    "load_detector",
    "robot_class_ids",
    "save_jpeg",
    "track_video",
    "ultralytics_available",
]


def track_video(
    video_path: Path,
    homography: np.ndarray | None = None,
    src_points: list[list[float]] | None = None,
    model_path: str | None = None,
    team_numbers: list[str] | None = None,
    frame_stride: int = 4,
    max_frames: int | None = None,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    on_progress: ProgressFn | None = None,
    motion_only: bool = False,
    tracker_mode: str = "hybrid",
    openai_key: str = "",
    google_key: str = "",
    openai_model: str = "",
    google_model: str = "",
    use_bev: bool = True,
    prefer_depth_neural: bool = True,
    depth_stride: int = 12,
    field_gate: bool = True,
    game_year: int | None = None,
    alliance_calibration_s: float = 45.0,
) -> dict[str, Any]:
    """Run multi-strategy detection + tracking. Returns field-space samples.

    ``field_gate`` enables field-aware gating (perimeter / footprint / static
    rejection) and BEV speed-gated association. Alliance colors come from a
    per-match Lab-chroma calibration with per-track hysteresis voting and a
    final 3-red / 3-blue assignment.
    """
    import cv2

    from ramscout.bev import calibrate_bev
    from ramscout.depth import estimate_depth, refine_foot_point

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open uploaded video ({Path(video_path).name}). "
            "Confirm it is a real MP4/MKV (not an empty or corrupt download), then re-upload."
        )

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    cal_index = int(min(fps * 5.0, max(total * 0.08, 1))) if total else int(fps * 5.0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(cal_index, 0))
    ok, first = cap.read()
    if not ok or first is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Could not read a calibration frame from the video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    bev_info: dict[str, Any] | None = None
    depth_cal = None
    if use_bev and src_points is None and homography is None:
        try:
            if on_progress:
                on_progress("Estimating camera angle (depth → BEV)…", 2.0)
            bev_cal, depth_cal = calibrate_bev(
                first,
                crop_top=crop_top,
                crop_bottom=crop_bottom,
                prefer_neural=prefer_depth_neural,
            )
            pts = bev_cal.src_points
            homography = homography_from_corners(pts)
            src_points = pts
            bev_info = bev_cal.as_dict()
        except Exception as exc:  # noqa: BLE001
            bev_info = {"error": str(exc), "used_depth": False}

    if homography is None:
        pts = src_points or default_source_points(frame_w, frame_h, crop_top, crop_bottom).tolist()
        src_points = pts
        homography = homography_from_corners(pts)
    y0, y1 = crop_bounds(frame_h, crop_top, crop_bottom)
    crop_h = max(y1 - y0, 1)
    crop_w = max(frame_w, 1)
    depth_map = depth_cal.depth if depth_cal is not None else None
    depth_refresh = max(1, int(depth_stride))

    raw_mode = "motion" if motion_only else (tracker_mode or "hybrid")
    mode = "hybrid" if (raw_mode or "").strip().lower() == "auto" else (raw_mode or "hybrid")
    strategies, setup_warnings = resolve_strategies(
        mode,
        model_path=model_path,
        openai_key=openai_key,
        google_key=google_key,
    )
    cascade = bool(TRACKER_MODES.get(mode, {}).get("cascade")) or mode == "hybrid"
    ensemble = EnsembleTracker(strategies, cascade=cascade)
    warnings = list(setup_warnings)
    strategy_names = [getattr(s, "name", "?") for s in strategies]
    label = "hybrid cascade" if cascade else mode
    warnings.append(f"Tracking mode '{label}' using: {', '.join(strategy_names)} + SORT MOT.")

    # Field-aware gating + BEV association share one homography.
    dt_s = max(frame_stride, 1) / fps
    depth_holder: dict[str, Any] = {"depth": depth_map}

    def foot_fn(bbox: list[float]) -> tuple[float, float]:
        return refine_foot_point(bbox, depth_holder["depth"], crop_y0=float(y0))

    gate: FieldGate | None = None
    if field_gate:
        gate = FieldGate(
            homography,
            crop_y0=float(y0),
            crop_w=crop_w,
            crop_h=crop_h,
            frame_h=frame_h,
            scorebug_bands=scorebug_bands_for_year(game_year),
            foot_fn=foot_fn,
            dt_s=dt_s,
        )
    ensemble.configure_geometry(gate, dt_s=dt_s)

    calibrator = AllianceCalibrator()
    voter = AllianceVoter(window=15, flip_ratio=0.65, min_votes=3)
    seen_tracks: set[int] = set()
    # A confirmed alliance flip on a live track is the signature of an ID swap
    # (two robots crossing). Split the track there instead of carrying the
    # wrong robot forward under the old id.
    emitted_label: dict[int, str] = {}
    tid_alias: dict[int, int] = {}
    split_count = 0

    samples: list[dict[str, Any]] = []
    frame_index = 0
    processed = 0
    source_hits: dict[str, int] = {}

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if frame_index % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        t = frame_index / fps
        cropped = frame[y0:y1, :]
        if use_bev and processed % depth_refresh == 0:
            try:
                # Classical refresh is cheap; neural depth is calibration-only unless already loaded.
                depth_map = estimate_depth(cropped, prefer_neural=False).depth
                depth_holder["depth"] = depth_map
            except Exception:
                pass
        ctx = TrackerContext(
            frame_w=frame_w,
            frame_h=frame_h,
            crop_w=crop_w,
            crop_h=crop_h,
            t=t,
            frame_index=frame_index,
            team_numbers=list(team_numbers or []),
            openai_key=openai_key,
            google_key=google_key,
            openai_model=openai_model or "gpt-4o-mini",
            google_model=google_model or "gemini-3.5-flash",
        )
        detections = ensemble.detect(cropped, ctx)

        feet: list[list[float]] = []
        for det in detections:
            fx, fy = refine_foot_point(list(det.bbox), depth_map, crop_y0=float(y0))
            feet.append([fx, fy])
        mapped = project_points(feet, homography) if feet else np.zeros((0, 2), dtype=np.float32)

        calibrating = t <= alliance_calibration_s or not calibrator.calibrated
        for det, (fx, fy), (mx, my) in zip(detections, feet, mapped):
            x1, y1b, x2, y2 = det.bbox
            tid = int(det.track_id)
            roi = cropped[max(int(y1b), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)]

            # --- alliance: lower-band chroma → calibrated centroids → hysteresis vote
            band = bumper_band(cropped, det.bbox)
            feature = chroma_features(band) if band is not None else None
            moving = bool(det.meta.get("moving", True)) if det.meta else True
            hits = int(det.meta.get("hits", 0)) if det.meta else 0
            if calibrating and moving and hits >= 3:
                calibrator.add(feature)
                if calibrator.sample_count >= calibrator.min_samples and (
                    not calibrator.calibrated or processed % 30 == 0
                ):
                    calibrator.fit()
            if tid not in seen_tracks:
                seen_tracks.add(tid)
                prior_label, prior_w = side_prior(
                    float(mx), float(t), field_length=FIELD_LENGTH, alliance_depth=ALLIANCE_DEPTH
                )
                voter.set_prior(tid, prior_label, prior_w)
            label, conf, _method = calibrator.classify(feature)
            if label in {"red", "blue"}:
                voter.vote(tid, label, conf)
            if det.alliance in {"red", "blue"}:
                # Strategy-level label (color blob / cloud model) is a weaker vote.
                voter.vote(tid, det.alliance, 0.7 if det.source in {"gemini", "openai"} else 0.45)
            alliance, alliance_conf = voter.label(tid)
            prev_label = emitted_label.get(tid)
            if prev_label in {"red", "blue"} and alliance in {"red", "blue"} and alliance != prev_label:
                split_count += 1
                tid_alias[tid] = 50000 + split_count
            if alliance in {"red", "blue"}:
                emitted_label[tid] = alliance
            out_tid = tid_alias.get(tid, tid)
            if alliance == "unknown":
                alliance = "blue" if float(mx) < FIELD_LENGTH * 0.5 else "red"
                alliance_conf = 0.3

            team = det.team or ""
            if not team and team_numbers:
                raw = ocr_digits(roi)
                team = constrain_to_teams(raw, team_numbers) or ""
            source_hits[det.source] = source_hits.get(det.source, 0) + 1
            sample: dict[str, Any] = {
                "t": round(float(t), 3),
                "frame": frame_index,
                "track_id": out_tid,
                "x": float(mx),
                "y": float(my),
                "px": float(fx),
                "py": float(fy),
                "alliance": alliance,
                "alliance_conf": round(float(alliance_conf), 3),
                "team": team,
                "bbox": [x1, y1b + y0, x2, y2 + y0],
                "source": det.source,
                "view": "overview",
            }
            if det.meta and det.meta.get("speed_in_s") is not None:
                sample["speed_in_s"] = float(det.meta["speed_in_s"])
            samples.append(sample)

        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Tracking robots…", min(99.0, 100.0 * frame_index / total))

    cap.release()
    for w in ensemble.warnings:
        if w not in warnings:
            warnings.append(w)

    # Exactly 3 red + 3 blue across the strongest tracks (color votes + start side).
    samples = balance_alliances(samples)
    if gate is not None:
        warnings.append(gate.stats.summary())
    if calibrator.calibrated:
        warnings.append(
            f"Alliance colors calibrated to this broadcast (Lab chroma, {calibrator.sample_count} bumper samples, "
            f"separation {calibrator.separation:.0f})."
        )
    elif samples:
        warnings.append("Alliance colors from global HSV thresholds (not enough bumper samples to calibrate).")

    used_model = any(n in {"yolo", "openai", "gemini"} for n in source_hits)
    if not samples:
        warnings.append(
            "No robot tracks found in the video crop. Recalibrate the four field corners, "
            "widen the field crop (Advanced), try another tracking mode, or upload a clearer wide-angle VOD."
        )
    elif not used_model:
        warnings.append(
            "Tracked with local OpenCV methods only (no YOLO/cloud hits). Paths are approximate — confirm before pick lists."
        )

    if bev_info and bev_info.get("detail"):
        warnings.append(str(bev_info["detail"]))
    elif bev_info and bev_info.get("error"):
        warnings.append(f"BEV depth calibration skipped: {bev_info['error']}")
    elif use_bev:
        warnings.append("BEV camera-angle adjustment applied with classical depth cues.")

    return {
        "samples": samples,
        "warnings": warnings,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "first_frame": first,
        "homography": homography.tolist(),
        "src_points": src_points,
        "used_model": used_model,
        "crop": [crop_top, crop_bottom],
        "tracker_mode": mode,
        "strategies": strategy_names,
        "source_hits": source_hits,
        "yolo_hits": source_hits.get("yolo", 0),
        "motion_hits": source_hits.get("motion", 0),
        "bev": bev_info,
        "depth_source": (bev_info or {}).get("depth_source") or (depth_cal.source if depth_cal else None),
        "field_gate": {"accepted": gate.stats.accepted, "rejected": dict(gate.stats.rejected)} if gate else None,
        "alliance_calibration": calibrator.as_dict(),
    }


def save_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 85) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
