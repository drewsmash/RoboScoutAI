"""Robot detection and tracking on a recorded match video.

Orchestrates multiple strategies via ramscout.trackers:
- Local: motion, bumper color, optical flow (class-agnostic moving-object detector)
- Neural: Ultralytics YOLO (optional; FRC-tuned weights primary, COCO verify-only)
- Cloud: OpenAI Vision / Google Gemini (optional API keys)
- Depth / BEV: Depth Anything V2 (ONNX / torch) + bird's-eye projection for camera angle
- Field lines: carpet-boundary homography refinement + alliance orientation
- Layout: tracks only the *overview pane* of the broadcast decomposition and
  follows the director's layout switches (gaps instead of hallucinated paths)

Paths are produced even when neural/cloud backends are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.field import ALLIANCE_DEPTH, FIELD_LENGTH, FIELD_WIDTH
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


@dataclass
class _Segment:
    """A run of the video where one overview crop (or none) is valid."""

    t0: float
    t1: float
    box: tuple[float, float, float, float] | None  # normalized x0,y0,x1,y1
    index: int = 0


@dataclass
class _PaneSession:
    """Per-overview-crop tracking state (homography, gate, ensemble, alliance)."""

    seg: _Segment
    x0: int
    y0: int
    x1: int
    y1: int
    homography: np.ndarray
    src_points: list[list[float]]
    ensemble: EnsembleTracker
    gate: FieldGate | None
    calibrator: AllianceCalibrator
    voter: AllianceVoter
    depth: np.ndarray | None = None
    bev_info: dict[str, Any] | None = None
    depth_source: str | None = None
    field_mask: np.ndarray | None = None
    seen_tracks: set[int] = field(default_factory=set)
    emitted_label: dict[int, str] = field(default_factory=dict)
    tid_alias: dict[int, int] = field(default_factory=dict)
    frames: int = 0

    @property
    def crop_w(self) -> int:
        return max(self.x1 - self.x0, 1)

    @property
    def crop_h(self) -> int:
        return max(self.y1 - self.y0, 1)


def _segments_from_layout(
    layout: Any,
    duration: float,
    default_box: tuple[float, float, float, float],
) -> list[_Segment]:
    """Turn a CameraLayout timeline into contiguous segments with overview boxes."""
    timeline = list(getattr(layout, "timeline", None) or []) if layout is not None else []
    if not timeline:
        return [_Segment(0.0, max(duration, 0.0) or 1e9, default_box, 0)]
    segs: list[_Segment] = []
    boxes: list[tuple[float, float, float, float]] = []
    for raw in timeline:
        t0 = float(raw.get("t0") or 0.0)
        t1 = float(raw.get("t1") or duration)
        ov = raw.get("overview")
        box = tuple(float(v) for v in ov) if ov and len(ov) == 4 else None
        idx = 0
        if box is not None:
            match = next((i for i, b in enumerate(boxes) if all(abs(a - c) <= 0.03 for a, c in zip(a_b(b), a_b(box)))), None)
            if match is None:
                boxes.append(box)
                idx = len(boxes) - 1
            else:
                idx = match
                box = boxes[match]
        segs.append(_Segment(t0, t1, box, idx))
    if not segs:
        return [_Segment(0.0, duration, default_box, 0)]
    segs[-1].t1 = max(segs[-1].t1, duration)
    return segs


def a_b(box: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(v) for v in box)


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
    tracker_mode: str = "auto",
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
    *,
    crop_left: float = 0.0,
    crop_right: float = 1.0,
    layout: Any = None,
    start_s: float | None = None,
    end_s: float | None = None,
    smooth: bool = True,
    bumper_ocr: bool = True,
    field_lines: bool = True,
    benchmark_s: float = 12.0,
    benchmark: bool = True,
) -> dict[str, Any]:
    """Run multi-strategy detection + tracking. Returns field-space samples.

    ``layout`` (a :class:`ramscout.multicam.CameraLayout`) supplies the
    overview pane and its timeline; when the director drops the overview the
    tracker pauses and records a gap. ``tracker_mode="auto"`` benchmarks every
    available strategy set on the first ``benchmark_s`` seconds and keeps the
    best (see :mod:`ramscout.trackers.selection`).
    """
    mode_req = "motion" if motion_only else (tracker_mode or "auto").strip().lower()
    selection: dict[str, Any] | None = None
    if mode_req == "auto" and benchmark:
        from ramscout.trackers.selection import select_tracker_mode

        selection = select_tracker_mode(
            video_path,
            homography=homography,
            src_points=src_points,
            model_path=model_path,
            team_numbers=team_numbers,
            frame_stride=frame_stride,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_left=crop_left,
            crop_right=crop_right,
            layout=layout,
            openai_key=openai_key,
            google_key=google_key,
            openai_model=openai_model,
            google_model=google_model,
            use_bev=use_bev,
            prefer_depth_neural=prefer_depth_neural,
            field_gate=field_gate,
            game_year=game_year,
            start_s=start_s,
            benchmark_s=benchmark_s,
            on_progress=on_progress,
        )
        mode_req = selection["chosen"]
    result = _track_video_impl(
        video_path,
        homography=homography,
        src_points=src_points,
        model_path=model_path,
        team_numbers=team_numbers,
        frame_stride=frame_stride,
        max_frames=max_frames,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        on_progress=on_progress,
        tracker_mode=mode_req,
        openai_key=openai_key,
        google_key=google_key,
        openai_model=openai_model,
        google_model=google_model,
        use_bev=use_bev,
        prefer_depth_neural=prefer_depth_neural,
        depth_stride=depth_stride,
        field_gate=field_gate,
        game_year=game_year,
        alliance_calibration_s=alliance_calibration_s,
        crop_left=crop_left,
        crop_right=crop_right,
        layout=layout,
        start_s=start_s,
        end_s=end_s,
        smooth=smooth,
        bumper_ocr=bumper_ocr,
        field_lines=field_lines,
    )
    result["tracker_selection"] = selection
    if selection:
        result["warnings"].insert(
            0,
            f"Auto mode benchmarked {len(selection['candidates'])} tracker sets on the first "
            f"{selection['window_s']:.0f}s and chose '{selection['chosen']}' "
            f"(score {selection['scores'][selection['chosen']]['total']:.2f}).",
        )
    return result


def _track_video_impl(
    video_path: Path,
    *,
    homography: np.ndarray | None,
    src_points: list[list[float]] | None,
    model_path: str | None,
    team_numbers: list[str] | None,
    frame_stride: int,
    max_frames: int | None,
    crop_top: float,
    crop_bottom: float,
    on_progress: ProgressFn | None,
    tracker_mode: str,
    openai_key: str,
    google_key: str,
    openai_model: str,
    google_model: str,
    use_bev: bool,
    prefer_depth_neural: bool,
    depth_stride: int,
    field_gate: bool,
    game_year: int | None,
    alliance_calibration_s: float,
    crop_left: float,
    crop_right: float,
    layout: Any,
    start_s: float | None,
    end_s: float | None,
    smooth: bool,
    bumper_ocr: bool,
    field_lines: bool,
) -> dict[str, Any]:
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
    duration = (total / fps) if total > 0 else 0.0

    default_box = (float(crop_left), float(crop_top), float(crop_right), float(crop_bottom))
    segments = _segments_from_layout(layout, duration or 1e9, default_box)
    # A user-supplied homography / corners pins the geometry of the *primary* box only.
    primary = max((s for s in segments if s.box is not None), key=lambda s: s.t1 - s.t0, default=None)
    pinned_index = primary.index if primary is not None else 0

    t_start = float(start_s or 0.0)
    t_end = float(end_s) if end_s is not None else (duration if duration > 0 else float("inf"))
    first_frame_index = int(round(t_start * fps))
    if first_frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)

    # Calibration frame: a few seconds into the primary segment (robots
    # parked, lights on) rather than frame 0 (pre-match graphics).
    cal_seg = primary or segments[0]
    cal_t = max(t_start, cal_seg.t0) + min(5.0, max(0.0, (cal_seg.t1 - cal_seg.t0) * 0.08))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(cal_t * fps), 0))
    ok, first = cap.read()
    if not ok or first is None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, first = cap.read()
    if not ok or first is None:
        cap.release()
        raise RuntimeError("Could not read a calibration frame from the video.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_frame_index)

    mode = tracker_mode if tracker_mode in TRACKER_MODES else "hybrid"
    if mode == "auto":
        mode = "hybrid"
    dt_s = max(frame_stride, 1) / fps
    warnings: list[str] = []
    sessions: dict[int, _PaneSession] = {}
    next_track_id = 1000
    strategy_names: list[str] = []
    cascade_label = mode

    def build_session(seg: _Segment, frame: np.ndarray) -> _PaneSession:
        nonlocal next_track_id, strategy_names, cascade_label
        assert seg.box is not None
        bx0, by0, bx1, by1 = seg.box
        x0 = int(np.clip(round(bx0 * frame_w), 0, frame_w - 2))
        x1 = int(np.clip(round(bx1 * frame_w), x0 + 2, frame_w))
        y0, y1 = crop_bounds(frame_h, by0, by1)
        y1 = max(y1, y0 + 2)
        pane_frame = frame[y0:y1, x0:x1]

        H = None
        pts: list[list[float]] | None = None
        bev_info: dict[str, Any] | None = None
        depth_cal = None
        if seg.index == pinned_index and (homography is not None or src_points is not None):
            pts = src_points
            H = homography if homography is not None else homography_from_corners(pts)
            if pts is None:
                pts = default_source_points(frame_w, frame_h, by0, by1).tolist()
        elif use_bev:
            try:
                if on_progress:
                    on_progress("Estimating camera angle (depth → BEV)…", 2.0)
                extra = _sample_segment_frames(cap, seg, fps, t_start, skip=frame) if field_lines else []
                bev_cal, depth_cal = calibrate_bev(
                    frame,
                    crop_top=by0,
                    crop_bottom=by1,
                    crop_left=bx0,
                    crop_right=bx1,
                    prefer_neural=prefer_depth_neural,
                    field_lines=field_lines,
                    extra_frames=extra,
                )
                pts = bev_cal.src_points
                H = homography_from_corners(pts)
                bev_info = bev_cal.as_dict()
            except Exception as exc:  # noqa: BLE001
                bev_info = {"error": str(exc), "used_depth": False}
        if H is None:
            pts = pts or _default_points_for_box(frame_w, frame_h, seg.box)
            H = homography_from_corners(pts)

        strategies, setup_warnings = resolve_strategies(
            mode,
            model_path=model_path,
            openai_key=openai_key,
            google_key=google_key,
        )
        cascade = bool(TRACKER_MODES.get(mode, {}).get("cascade")) or mode == "hybrid"
        ensemble = EnsembleTracker(strategies, cascade=cascade)
        ensemble.mot.next_id = next_track_id
        for w in setup_warnings:
            if w not in warnings:
                warnings.append(w)
        strategy_names = [getattr(s, "name", "?") for s in strategies]
        cascade_label = "hybrid cascade" if cascade else mode

        depth_map = depth_cal.depth if depth_cal is not None else None
        holder: dict[str, Any] = {"depth": depth_map}

        def foot_fn(bbox: list[float]) -> tuple[float, float]:
            fx, fy = refine_foot_point(bbox, holder["depth"], crop_y0=float(y0))
            return fx + float(x0), fy

        gate: FieldGate | None = None
        if field_gate:
            gate = FieldGate(
                H,
                crop_y0=float(y0),
                crop_w=x1 - x0,
                crop_h=y1 - y0,
                frame_h=frame_h,
                scorebug_bands=_scorebug_bands(layout, seg, game_year, frame_h, y0, y1),
                foot_fn=foot_fn,
                dt_s=dt_s,
            )
            gate.crop_x0 = float(x0)
        ensemble.configure_geometry(gate, dt_s=dt_s)
        session = _PaneSession(
            seg=seg,
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            homography=H,
            src_points=[[round(float(p[0]), 2), round(float(p[1]), 2)] for p in (pts or [])],
            ensemble=ensemble,
            gate=gate,
            calibrator=AllianceCalibrator(),
            voter=AllianceVoter(window=15, flip_ratio=0.65, min_votes=3),
            depth=depth_map,
            bev_info=bev_info,
            depth_source=(bev_info or {}).get("depth_source") or (depth_cal.source if depth_cal else None),
            field_mask=field_mask_for_crop(H, x0, y0, x1 - x0, y1 - y0) if field_gate else None,
        )
        session._holder = holder  # type: ignore[attr-defined]
        return session

    ocr_engine = None
    if bumper_ocr and team_numbers:
        try:
            from ramscout.ocr import BumperOCR

            ocr_engine = BumperOCR(team_numbers, openai_key=openai_key)
            if not ocr_engine.available:
                ocr_engine = None
        except Exception:  # noqa: BLE001
            ocr_engine = None

    samples: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    switches: list[dict[str, Any]] = []
    frame_index = first_frame_index
    processed = 0
    source_hits: dict[str, int] = {}
    depth_refresh = max(1, int(depth_stride))
    split_count = 0
    active: _PaneSession | None = None
    gap_open: dict[str, Any] | None = None
    seg_iter = 0

    def segment_at(t: float) -> _Segment:
        nonlocal seg_iter
        while seg_iter + 1 < len(segments) and t >= segments[seg_iter + 1].t0:
            seg_iter += 1
        while seg_iter > 0 and t < segments[seg_iter].t0:
            seg_iter -= 1
        return segments[seg_iter]

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t = frame_index / fps
        if t >= t_end:
            break
        if max_frames is not None and processed >= max_frames:
            break
        if (frame_index - first_frame_index) % max(frame_stride, 1) != 0:
            frame_index += 1
            continue

        seg = segment_at(t)
        if seg.box is None:
            if gap_open is None:
                gap_open = {"t0": round(t, 3), "reason": "no overview pane in broadcast layout"}
            if active is not None:
                switches.append({"t": round(t, 3), "from": active.seg.index, "to": None})
                active = None
            frame_index += 1
            processed += 1
            continue
        if gap_open is not None:
            gap_open["t1"] = round(t, 3)
            gaps.append(gap_open)
            gap_open = None
        if active is None or active.seg is not seg:
            if active is not None and active.seg.index != seg.index:
                switches.append({"t": round(t, 3), "from": active.seg.index, "to": seg.index})
            if active is not None:
                next_track_id = active.ensemble.mot.next_id
            session = sessions.get(seg.index)
            if session is None or session.seg.index != seg.index:
                session = build_session(seg, frame)
                sessions[seg.index] = session
            else:
                # Same camera geometry as before: keep the homography but
                # start fresh tracks (the gap breaks continuity anyway).
                session.ensemble.mot.reset()
                session.ensemble.mot.next_id = next_track_id
            session.seg = seg
            active = session

        s = active
        cropped = frame[s.y0 : s.y1, s.x0 : s.x1]
        # Depth refresh must preserve the backend used at calibration. Silently
        # forcing classical (prefer_neural=False) was wiping neural depth.
        if use_bev and s.frames % depth_refresh == 0 and s.depth is not None:
            try:
                prefer_neural = True
                src = ""
                if isinstance(getattr(s, "bev_info", None), dict):
                    src = str(s.bev_info.get("depth_source") or "")
                src = src or str(getattr(s, "depth_source", "") or "")
                prefer_neural = src.startswith(("onnx", "torch", "neural")) or "classical" not in src
                refreshed = estimate_depth(cropped, prefer_neural=prefer_neural)
                s.depth = refreshed.depth
                s._holder["depth"] = s.depth  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
        ctx = TrackerContext(
            frame_w=frame_w,
            frame_h=frame_h,
            crop_w=s.crop_w,
            crop_h=s.crop_h,
            t=t,
            frame_index=frame_index,
            team_numbers=list(team_numbers or []),
            openai_key=openai_key,
            google_key=google_key,
            openai_model=openai_model or "gpt-4o-mini",
            google_model=google_model or "gemini-3.5-flash",
            extras={"field_mask": s.field_mask} if s.field_mask is not None else {},
        )
        detections = s.ensemble.detect(cropped, ctx)

        feet: list[list[float]] = []
        for det in detections:
            fx, fy = refine_foot_point(list(det.bbox), s.depth, crop_y0=float(s.y0))
            feet.append([fx + float(s.x0), fy])
        mapped = project_points(feet, s.homography) if feet else np.zeros((0, 2), dtype=np.float32)

        seg_elapsed = t - max(s.seg.t0, t_start)
        calibrating = seg_elapsed <= alliance_calibration_s or not s.calibrator.calibrated
        for det, (fx, fy), (mx, my) in zip(detections, feet, mapped):
            x1, y1b, x2, y2 = det.bbox
            tid = int(det.track_id)
            roi = cropped[max(int(y1b), 0) : max(int(y2), 0), max(int(x1), 0) : max(int(x2), 0)]

            band = bumper_band(cropped, det.bbox)
            feature = chroma_features(band) if band is not None else None
            moving = bool(det.meta.get("moving", True)) if det.meta else True
            hits = int(det.meta.get("hits", 0)) if det.meta else 0
            if calibrating and moving and hits >= 3:
                s.calibrator.add(feature)
                if s.calibrator.sample_count >= s.calibrator.min_samples and (
                    not s.calibrator.calibrated or s.frames % 30 == 0
                ):
                    s.calibrator.fit()
            if tid not in s.seen_tracks:
                s.seen_tracks.add(tid)
                # Soft start-side prior only — never hard-assign unknown from field half.
                prior_label, prior_w = side_prior(
                    float(mx), float(seg_elapsed), field_length=FIELD_LENGTH, alliance_depth=ALLIANCE_DEPTH
                )
                s.voter.set_prior(tid, prior_label, min(float(prior_w), 0.25))
            label, conf, _method = s.calibrator.classify(feature)
            if label in {"red", "blue"}:
                s.voter.vote(tid, label, conf)
            if det.alliance in {"red", "blue"}:
                s.voter.vote(tid, det.alliance, 0.7 if det.source in {"gemini", "openai"} else 0.45)
            alliance, alliance_conf = s.voter.label(tid)
            prev_label = s.emitted_label.get(tid)
            # Confirmed alliance flip: flag association conflict — do NOT mint a new track ID.
            if (
                prev_label in {"red", "blue"}
                and alliance in {"red", "blue"}
                and alliance != prev_label
                and float(alliance_conf) >= 0.55
            ):
                det.meta = dict(det.meta or {})
                det.meta["alliance_conflict"] = True
                det.meta["previous_alliance"] = prev_label
                # Keep the confirmed alliance until a scout corrects or evidence dominates.
                alliance = prev_label
                alliance_conf = max(float(alliance_conf), 0.55)
            if alliance in {"red", "blue"}:
                s.emitted_label[tid] = alliance
            out_tid = tid
            # Unknown stays unknown. Never invent alliance from field half.
            if alliance not in {"red", "blue"}:
                alliance = "unknown"
                alliance_conf = float(alliance_conf or 0.0)

            team = det.team or ""
            if not team and ocr_engine is not None and hits >= 4 and s.frames % 6 == 0:
                team = ocr_engine.observe(out_tid, roi) or ""
            elif not team and ocr_engine is None and team_numbers and hits >= 4 and s.frames % 12 == 0:
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
                "bbox": [x1 + s.x0, y1b + s.y0, x2 + s.x0, y2 + s.y0],
                "source": det.source,
                "view": "overview",
                "pane": s.seg.index,
            }
            if det.meta and det.meta.get("alliance_conflict"):
                sample["alliance_conflict"] = True
                sample["previous_alliance"] = det.meta.get("previous_alliance")
            if det.meta and det.meta.get("speed_in_s") is not None:
                sample["speed_in_s"] = float(det.meta["speed_in_s"])
            samples.append(sample)

        s.frames += 1
        processed += 1
        frame_index += 1
        if on_progress and total:
            on_progress("Tracking robots…", min(99.0, 100.0 * (frame_index - first_frame_index) / max(total - first_frame_index, 1)))

    cap.release()
    if gap_open is not None:
        gap_open["t1"] = round(frame_index / fps, 3)
        gaps.append(gap_open)

    if ocr_engine is not None:
        team_by_track = ocr_engine.assignments()
        for sample in samples:
            if not sample.get("team") and sample["track_id"] in team_by_track:
                sample["team"] = team_by_track[sample["track_id"]]

    for sess in sessions.values():
        for w in sess.ensemble.warnings:
            if w not in warnings:
                warnings.append(w)
    warnings.append(f"Tracking mode '{cascade_label}' using: {', '.join(strategy_names)} + BEV MOT (ByteTrack/OC-SORT association).")

    # Exactly 3 red + 3 blue across the strongest tracks (color votes + start side).
    samples = balance_alliances(samples)
    if smooth and samples:
        try:
            from ramscout.smoothing import smooth_samples

            samples = smooth_samples(samples, dt_hint=dt_s)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"Path smoothing skipped: {exc}")

    prim_session = sessions.get(pinned_index) or (next(iter(sessions.values())) if sessions else None)
    if prim_session is not None and prim_session.gate is not None:
        warnings.append(prim_session.gate.stats.summary())
    if prim_session is not None and prim_session.calibrator.calibrated:
        warnings.append(
            f"Alliance colors calibrated to this broadcast (Lab chroma, {prim_session.calibrator.sample_count} bumper samples, "
            f"separation {prim_session.calibrator.separation:.0f})."
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
    if gaps:
        gap_s = sum(float(g.get("t1", g["t0"])) - float(g["t0"]) for g in gaps)
        warnings.append(
            f"Overview camera absent for {gap_s:.0f}s across {len(gaps)} layout switch(es) — paths have gaps there instead of guesses."
        )
    if ocr_engine is not None and ocr_engine.assignments():
        warnings.append(f"Bumper OCR ({ocr_engine.backend}) read team numbers for {len(ocr_engine.assignments())} track(s).")

    bev_info = prim_session.bev_info if prim_session is not None else None
    if bev_info and bev_info.get("detail"):
        warnings.append(str(bev_info["detail"]))
    elif bev_info and bev_info.get("error"):
        warnings.append(f"BEV depth calibration skipped: {bev_info['error']}")
    elif use_bev:
        warnings.append("BEV camera-angle adjustment applied with classical depth cues.")

    H_out = prim_session.homography if prim_session is not None else (homography if homography is not None else homography_from_corners(src_points or default_source_points(frame_w, frame_h, crop_top, crop_bottom).tolist()))
    pts_out = prim_session.src_points if prim_session is not None else (src_points or default_source_points(frame_w, frame_h, crop_top, crop_bottom).tolist())
    crop_out = [crop_top, crop_bottom]
    box_out = list(default_box)
    if prim_session is not None:
        crop_out = [prim_session.y0 / max(frame_h, 1), prim_session.y1 / max(frame_h, 1)]
        box_out = [prim_session.x0 / max(frame_w, 1), prim_session.y0 / max(frame_h, 1), prim_session.x1 / max(frame_w, 1), prim_session.y1 / max(frame_h, 1)]

    gate_stats = None
    if prim_session is not None and prim_session.gate is not None:
        gate_stats = {"accepted": prim_session.gate.stats.accepted, "rejected": dict(prim_session.gate.stats.rejected)}

    return {
        "samples": samples,
        "warnings": warnings,
        "fps": fps,
        "frame_size": [frame_w, frame_h],
        "first_frame": first,
        "homography": np.asarray(H_out).tolist(),
        "src_points": pts_out,
        "used_model": used_model,
        "crop": crop_out,
        "crop_box": box_out,
        "tracker_mode": mode,
        "strategies": strategy_names,
        "source_hits": source_hits,
        "yolo_hits": source_hits.get("yolo", 0),
        "motion_hits": source_hits.get("motion", 0),
        "bev": bev_info,
        "depth_source": (bev_info or {}).get("depth_source") or (prim_session.depth_source if prim_session else None),
        "depth_backends": _depth_backend_status(),
        "field_gate": gate_stats,
        "alliance_calibration": prim_session.calibrator.as_dict() if prim_session is not None else {},
        "gaps": gaps,
        "layout_switches": switches,
        "segments": [
            {"t0": round(sg.t0, 3), "t1": round(sg.t1, 3), "box": list(sg.box) if sg.box else None, "index": sg.index}
            for sg in segments
        ],
        "processed_frames": processed,
        "window": [round(t_start, 3), round(min(t_end, frame_index / fps), 3)],
    }


def field_mask_for_crop(
    homography: np.ndarray,
    x0: int,
    y0: int,
    crop_w: int,
    crop_h: int,
    *,
    margin_in: float = 14.0,
    robot_height_in: float = 40.0,
) -> np.ndarray | None:
    """uint8 mask (255 inside) of where robots can *appear* in the crop.

    The carpet rectangle (plus ``margin_in``) is back-projected through the
    homography; the far edge is then lifted by a robot height so a robot at
    the far wall — whose body sits *above* its foot point in the image — is
    not clipped. Everything else (crowd, referees, driver stations, scorebug)
    can never become a motion / colour proposal.
    """
    import cv2

    try:
        Hinv = np.linalg.inv(np.asarray(homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    L, W, m = FIELD_LENGTH, FIELD_WIDTH, margin_in
    corners_in = np.array([[-m, -m], [L + m, -m], [L + m, W + m], [-m, W + m]], dtype=np.float64)
    pts = cv2.perspectiveTransform(corners_in.reshape(-1, 1, 2), Hinv).reshape(-1, 2)
    if not np.all(np.isfinite(pts)):
        return None
    pts[:, 0] -= x0
    pts[:, 1] -= y0
    # Lift the far edge: pixels per inch at the far wall from the far edge's
    # projected width, then robot_height_in above it.
    far = pts[:2]
    far_w_px = float(np.hypot(*(far[1] - far[0])))
    px_per_in = far_w_px / max(L + 2 * m, 1.0)
    lift = min(robot_height_in * px_per_in, 0.35 * crop_h)
    order = np.argsort(pts[:, 1])
    for i in order[:2]:
        pts[i, 1] -= lift
    mask = np.zeros((max(crop_h, 1), max(crop_w, 1)), dtype=np.uint8)
    poly = np.clip(pts, [-10 * crop_w, -10 * crop_h], [10 * crop_w, 10 * crop_h]).astype(np.int32)
    cv2.fillConvexPoly(mask, poly, 255)
    if np.count_nonzero(mask) < 0.05 * mask.size:
        return None
    return mask


def _sample_segment_frames(
    cap: Any,
    seg: _Segment,
    fps: float,
    t_start: float,
    *,
    skip: np.ndarray | None = None,
    count: int = 6,
    span_s: float = 60.0,
) -> list[np.ndarray]:
    """Grab a few frames spread over the segment (temporal median for the
    carpet model) and put the capture back where it was."""
    import cv2

    pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    t0 = max(seg.t0, t_start)
    t1 = min(seg.t1, t0 + span_s)
    if not np.isfinite(t1) or t1 <= t0 + 2.0:
        t1 = t0 + 20.0
    frames: list[np.ndarray] = []
    try:
        for t in np.linspace(t0 + 1.0, t1, count):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t * fps), 0))
            ok, fr = cap.read()
            if ok and fr is not None and (skip is None or fr.shape == skip.shape):
                frames.append(fr)
    except Exception:  # noqa: BLE001
        pass
    finally:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    return frames


def _default_points_for_box(frame_w: int, frame_h: int, box: tuple[float, float, float, float]) -> list[list[float]]:
    from ramscout.geometry import DEFAULT_SRC_NORM

    bx0, by0, bx1, by1 = box
    x0 = bx0 * frame_w
    y0 = by0 * frame_h
    w = max((bx1 - bx0) * frame_w, 1.0)
    h = max((by1 - by0) * frame_h, 1.0)
    return [[x0 + nx * w, y0 + ny * h] for nx, ny in DEFAULT_SRC_NORM]


def _scorebug_bands(layout: Any, seg: _Segment, game_year: int | None, frame_h: int, y0: int, y1: int) -> list[tuple[float, float]]:
    """Frame-normalized scorebug bands: detected box first, year defaults otherwise."""
    bands: list[tuple[float, float]] = []
    bug = getattr(layout, "scorebug", None) if layout is not None else None
    pane = None
    if layout is not None:
        for p in getattr(layout, "panes", []) or []:
            if getattr(p, "role", "") == "overview" and getattr(p, "scorebug", None):
                pane = p
                break
    if pane is not None and pane.scorebug:
        # pane-relative → frame-normalized
        top = y0 / frame_h + pane.scorebug[1] * (y1 - y0) / frame_h
        bot = y0 / frame_h + pane.scorebug[3] * (y1 - y0) / frame_h
        bands.append((float(top) - 0.01, float(bot) + 0.01))
    elif bug and seg.box is not None and seg.box[1] <= bug[1] <= seg.box[3]:
        bands.append((float(bug[1]) - 0.01, float(bug[3]) + 0.01))
    if bands:
        return bands
    if layout is not None and getattr(layout, "method", "") == "decomposition" and seg.box is not None:
        # The decomposition found no scorebug inside this pane: do not blank
        # out 12 % of a real field camera with year defaults.
        return []
    return scorebug_bands_for_year(game_year)


def _depth_backend_status() -> dict[str, Any] | None:
    try:
        from ramscout.depth import depth_backends

        status = depth_backends()
        return {
            "active": status.get("active"),
            "onnx": status["backends"]["onnx"]["importable"],
            "onnx_model_cached": status["backends"]["onnx"]["model_cached"],
            "torch": status["backends"]["torch"]["importable"],
            "missing": {k: v.get("missing") for k, v in status["backends"].items() if v.get("missing")},
        }
    except Exception:  # noqa: BLE001
        return None


def save_jpeg(frame_bgr: np.ndarray, path: Path, quality: int = 85) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
