"""End-to-end job orchestration for a YouTube match VOD."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ramscout.broadcast import match_from_overlay, merge_tba, parse_broadcast_text, read_overlay_from_video
from ramscout.detect import ensure_detector_weights, find_local_model, save_jpeg, track_video
from ramscout.events import Pose, build_cards, detect_events
from ramscout.firstevents import resolve_firstevents
from ramscout.gameconfig import public_game
from ramscout.geometry import default_source_points, reproject_samples
from ramscout.identity import assign_by_start, keep_top_tracks, majority_alliance, stitch_occlusions
from ramscout.ingest import download_video, fetch_video_info, store_uploaded_video
from ramscout.paths import jobs_dir, models_dirs
from ramscout.simulate import DEMO_MATCH, DEMO_VIDEO, demo_tracks
from ramscout.tba import TBAClient, TBAError, enrich_match, resolve_match
from ramscout.tba_scrape import enrich_from_html, resolve_match_html, scrape_match
from ramscout.titles import TitleHints, parse_match_title

log = logging.getLogger(__name__)

DATA = jobs_dir()


ProgressFn = Callable[[str, float], None]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Job:
    id: str
    url: str
    created_at: str
    status: str = "queued"
    progress: float = 0.0
    message: str = "Waiting to start…"
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    video_info: dict[str, Any] | None = None
    match: dict[str, Any] | None = None
    assignments: dict[str, str] = field(default_factory=dict)
    src_points: list[list[float]] | None = None
    samples: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    cards: list[dict[str, Any]] = field(default_factory=list)
    frame_size: list[int] = field(default_factory=lambda: [1280, 720])
    used_model: bool = False
    demo: bool = False
    tba_key: str = ""
    event_key: str = ""
    match_key: str = ""
    video_path: str | None = None
    frame_path: str | None = None
    overlay: dict[str, Any] | None = None
    game: dict[str, Any] | None = None
    crop_top: float = 0.10
    crop_bottom: float = 0.65
    user_calibrated: bool = False
    seeds: list[dict[str, Any]] = field(default_factory=list)
    tracker_mode: str = "auto"
    openai_key: str = ""
    google_key: str = ""
    tracker_strategies: list[str] = field(default_factory=list)
    source_hits: dict[str, int] = field(default_factory=dict)
    camera: dict[str, Any] = field(default_factory=dict)
    auto_multicam: bool = True
    edited_events: bool = False
    bev: dict[str, Any] = field(default_factory=dict)
    views: dict[str, Any] = field(default_factory=dict)
    side_cues: list[dict[str, Any]] = field(default_factory=list)
    thinking_stages: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "url": self.url,
            "created_at": self.created_at,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "warnings": self.warnings,
            "video_info": self.video_info,
            "match": _public_match(self.match),
            "zebra": _public_zebra((self.match or {}).get("zebra") if self.match else None),
            "overlay": self.overlay,
            "game": self.game or public_game(),
            "assignments": self.assignments,
            "src_points": self.src_points,
            "samples": self.samples,
            "events": self.events,
            "cards": self.cards,
            "seeds": self.seeds,
            "frame_size": self.frame_size,
            "used_model": self.used_model,
            "demo": self.demo,
            "has_video": bool(self.video_path),
            "has_frame": bool(self.frame_path),
            "user_calibrated": self.user_calibrated,
            "crop_top": self.crop_top,
            "crop_bottom": self.crop_bottom,
            "tracker_mode": self.tracker_mode,
            "tracker_strategies": self.tracker_strategies,
            "source_hits": self.source_hits,
            "camera": self.camera,
            "auto_multicam": self.auto_multicam,
            "edited_events": self.edited_events,
            "bev": self.bev,
            "views": self.views,
            "side_cues": self.side_cues,
            "thinking_stages": self.thinking_stages,
        }


def _public_match(match: dict[str, Any] | None) -> dict[str, Any] | None:
    if not match:
        return None
    copy = dict(match)
    copy.pop("zebra", None)
    return copy


def _public_zebra(zebra: dict[str, Any] | None) -> dict[str, Any] | None:
    if not zebra:
        return None
    times = list(zebra.get("times") or [])
    if not times:
        return {"alliances": zebra.get("alliances") or {}}
    stride = max(1, len(times) // 180)

    def thin(values: list[Any]) -> list[Any]:
        return values[::stride] if values else []

    alliances: dict[str, Any] = {}
    for color, robots in (zebra.get("alliances") or {}).items():
        slim = []
        for robot in robots or []:
            slim.append(
                {
                    "team_key": robot.get("team_key"),
                    "xs": thin(list(robot.get("xs") or [])),
                    "ys": thin(list(robot.get("ys") or [])),
                }
            )
        alliances[color] = slim
    return {"times": thin(times), "alliances": alliances}


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self, **kwargs: Any) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], created_at=_now(), **kwargs)
        with self._lock:
            self._jobs[job.id] = job
        (DATA / job.id).mkdir(parents=True, exist_ok=True)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(self, job: Job, **kwargs: Any) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(job, key, value)

    def set_progress(self, job: Job, status: str, message: str, progress: float) -> None:
        stages = list(job.thinking_stages or [])
        stages.append(
            {
                "status": status,
                "message": message,
                "progress": round(float(progress), 1),
                "at": _now(),
            }
        )
        # Keep the panel snappy — last ~12 stage ticks.
        if len(stages) > 12:
            stages = stages[-12:]
        self.update(job, status=status, message=message, progress=progress, thinking_stages=stages)


STORE = JobStore()


def start_job(
    url: str,
    tba_key: str = "",
    event_key: str = "",
    match_key: str = "",
    demo: bool = False,
    crop_top: float = 0.10,
    crop_bottom: float = 0.65,
    local_video: Path | str | None = None,
    tracker_mode: str = "auto",
    openai_key: str = "",
    google_key: str = "",
    auto_multicam: bool = True,
) -> Job:
    job = STORE.create(
        url=url,
        tba_key=tba_key,
        event_key=event_key,
        match_key=match_key,
        demo=demo,
        crop_top=float(crop_top),
        crop_bottom=float(crop_bottom),
        tracker_mode=(tracker_mode or "auto").strip().lower() or "auto",
        openai_key=openai_key or "",
        google_key=google_key or "",
        auto_multicam=bool(auto_multicam),
    )
    if local_video:
        dest = DATA / job.id
        stored = store_uploaded_video(Path(local_video), dest)
        STORE.update(job, video_path=str(stored))
    thread = threading.Thread(target=_run_job, args=(job.id,), daemon=True)
    thread.start()
    return job


def apply_calibration(job: Job, src_points: list[list[float]]) -> Job:
    STORE.update(job, src_points=src_points, user_calibrated=True)
    _reproject_and_scout(job)
    return job


def apply_assignments(job: Job, assignments: dict[str, str]) -> Job:
    merged = dict(job.assignments)
    merged.update({str(k): str(v) for k, v in assignments.items()})
    STORE.update(job, assignments=merged)
    _reproject_and_scout(job)
    return job


def _run_job(job_id: str) -> None:
    job = STORE.get(job_id)
    if job is None:
        return
    try:
        if job.demo:
            _run_demo(job)
            return
        _run_real(job)
    except Exception as exc:  # noqa: BLE001
        log.exception("Job %s failed", job_id)
        STORE.update(job, status="error", error=str(exc), message=str(exc), progress=0)


def _run_demo(job: Job) -> None:
    STORE.set_progress(job, "resolving", "Reading the sample broadcast overlay…", 15)
    reading = parse_broadcast_text(DEMO_VIDEO["title"], DEMO_VIDEO["description"])
    match = merge_tba(match_from_overlay(reading), DEMO_MATCH)
    match["key"] = DEMO_MATCH["key"]
    match["event_key"] = DEMO_MATCH["event_key"]
    STORE.update(
        job,
        video_info=DEMO_VIDEO,
        match=match,
        overlay=reading.as_dict(),
        game=public_game(reading.year),
    )
    STORE.set_progress(job, "views", "Sectioning camera angles…", 40)
    STORE.set_progress(job, "tracking", "Depth → BEV calibration (demo)…", 55)
    STORE.set_progress(job, "scouting", "Building robot paths…", 70)
    samples = demo_tracks()
    for sample in samples:
        sample.setdefault("view", "overview")
        sample.setdefault("bbox", [])
    assignments = {str(s["track_id"]): s["team"] for s in samples}
    STORE.update(
        job,
        samples=samples,
        assignments=assignments,
        used_model=False,
        bev={
            "pitch_deg": 38.0,
            "tilt_strength": 0.62,
            "depth_source": "classical",
            "detail": "Demo BEV — classical depth cues (no VOD).",
            "used_depth": True,
        },
        views={
            "mode": "stacked_sides",
            "panes": [
                {"role": "overview", "purpose": "Top wide-angle — general movement", "crop_top": 0.02, "crop_bottom": 0.48, "crop_left": 0, "crop_right": 1},
                {"role": "blue_side", "purpose": "Blue scoring / climb", "crop_top": 0.52, "crop_bottom": 0.98, "crop_left": 0, "crop_right": 0.5, "alliance_bias": "blue"},
                {"role": "red_side", "purpose": "Red scoring / climb", "crop_top": 0.52, "crop_bottom": 0.98, "crop_left": 0.5, "crop_right": 1, "alliance_bias": "red"},
            ],
            "side_cues": 0,
        },
        camera={"mode": "stacked_sides", "detail": "Demo multi-view layout.", "confidence": 0.9},
        warnings=[
            "Sample match — scores and teams were parsed from the broadcast title/overlay text. Paths are simulated.",
            "Demo includes sectional camera roles + BEV camera-angle metadata.",
        ],
    )
    _reproject_and_scout(job)
    STORE.set_progress(job, "ready", "Sample match is ready.", 100)


def _run_real(job: Job) -> None:
    STORE.set_progress(job, "resolving", "Reading YouTube metadata…", 8)
    info = fetch_video_info(job.url)
    STORE.update(job, video_info=info)
    if info.get("warning"):
        warning = str(info["warning"])
        if job.video_path and Path(job.video_path).is_file():
            warning = "YouTube metadata used oEmbed fallback (player API blocked). Analyzing the uploaded VOD."
        job.warnings.append(warning)
    if info.get("is_live"):
        raise RuntimeError("This looks like a live stream. Paste a recorded match video instead.")

    hints = parse_match_title(info.get("title") or "", job.url)
    # FIRST short titles like "Qualification 65 - South Florida Regional" omit the year.
    year = hints.year
    if year is None and job.event_key[:4].isdigit() if job.event_key else False:
        year = int(job.event_key[:4])
    if year is None:
        from ramscout.gameconfig import DEFAULT_YEAR

        year = DEFAULT_YEAR
    if year != hints.year:
        hints = TitleHints(
            year=year,
            event_name=hints.event_name,
            comp_level=hints.comp_level,
            set_number=hints.set_number,
            match_number=hints.match_number,
            video_id=hints.video_id,
            title=hints.title,
        )
    reading = parse_broadcast_text(info.get("title") or "", info.get("description") or "", job.url)
    video_match = match_from_overlay(reading)
    STORE.update(job, match=video_match, overlay=reading.as_dict(), game=public_game(hints.year or reading.year))

    tba_key = job.tba_key.strip()
    tba_match = None
    if tba_key:
        STORE.set_progress(job, "resolving", "Optional TBA match lookup…", 18)
        try:
            with TBAClient(tba_key) as client:
                found = resolve_match(client, hints, event_key=job.event_key or None, match_key=job.match_key or None)
                if found:
                    tba_match = enrich_match(client, found)
        except TBAError as exc:
            job.warnings.append(str(exc))
    else:
        job.warnings.append(
            "No TBA API key — trying public TBA website scrape, then FIRST Event Web / video overlay."
        )

    # HTML scrape fallback when API key is missing or the API call failed.
    if not tba_match:
        STORE.set_progress(job, "resolving", "Scraping TBA match pages (no API key)…", 19)
        try:
            html_match = None
            if job.match_key:
                html_match = scrape_match(job.match_key)
            if not html_match:
                html_match = resolve_match_html(
                    match_key=job.match_key or None,
                    event_key=job.event_key or (job.match_key.split("_")[0] if job.match_key else None),
                    comp_level=hints.comp_level,
                    match_number=hints.match_number,
                    set_number=hints.set_number,
                )
            if html_match:
                tba_match = enrich_from_html(html_match)
                job.warnings.append("TBA data loaded via public website scrape (API fallback).")
        except Exception as exc:  # noqa: BLE001
            job.warnings.append(f"TBA HTML scrape skipped ({exc}).")

    if not tba_match:
        STORE.set_progress(job, "resolving", "Looking up FIRST Event Web results…", 20)
        event_key = job.event_key or (job.match_key.split("_")[0] if job.match_key else None)
        fe = resolve_firstevents(hints, event_key=event_key)
        if fe:
            video_match = merge_tba(video_match, fe)
            video_match["source"] = fe.get("source") or video_match.get("source")
            video_match["key"] = fe.get("key") or video_match.get("key")
            video_match["event_key"] = fe.get("event_key") or video_match.get("event_key")
            STORE.update(job, match=video_match)

    dest = DATA / job.id
    if job.video_path and Path(job.video_path).is_file():
        STORE.set_progress(job, "downloading", "Using uploaded match video…", 45)
        video_path = Path(job.video_path)
    else:
        STORE.set_progress(job, "downloading", "Downloading the match video…", 25)

        def dl_progress(message: str, pct: float) -> None:
            STORE.set_progress(job, "downloading", message, 25 + pct * 0.25)

        try:
            video_path = download_video(job.url, dest, on_progress=dl_progress)
        except Exception as exc:  # noqa: BLE001
            # Keep scoreboard metadata even when YouTube blocks the file download.
            STORE.update(job, match=video_match, game=public_game(hints.year))
            raise RuntimeError(str(exc)) from exc
        STORE.update(job, video_path=str(video_path))

    STORE.set_progress(job, "resolving", "Reading the on-screen scorebug…", 52)
    reading = read_overlay_from_video(video_path, existing=reading, year=hints.year or reading.year)
    # Keep FIRST/TBA teams if overlay OCR did not recover them.
    video_match = merge_tba(match_from_overlay(reading), tba_match or video_match)
    STORE.update(job, match=video_match, overlay=reading.as_dict(), game=public_game(reading.year or hints.year))

    if getattr(job, "auto_multicam", True) and job.video_path:
        try:
            from ramscout.multicam import analyze_video, apply_layout

            STORE.set_progress(job, "views", "Sectioning camera angles…", 53)
            layout = analyze_video(job.video_path)
            top, bottom, chosen = apply_layout(
                layout,
                user_crop_top=job.crop_top,
                user_crop_bottom=job.crop_bottom,
                auto=True,
            )
            warnings = list(job.warnings or [])
            warnings.append(chosen.detail)
            views = {
                "mode": chosen.mode,
                "panes": [p.as_dict() for p in (chosen.panes or [])],
                "split_y": chosen.split_y,
                "split_x": chosen.split_x,
            }
            STORE.update(
                job,
                crop_top=top,
                crop_bottom=bottom,
                camera=chosen.as_dict(),
                views=views,
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            warnings = list(job.warnings or [])
            warnings.append(f"Multi-camera detect skipped: {exc}")
            STORE.update(job, warnings=warnings)

    STORE.set_progress(job, "tracking", "Depth → BEV calibration, then tracking overview…", 55)
    model = find_local_model(models_dirs()) or ensure_detector_weights()
    team_numbers = []
    if video_match:
        team_numbers = [str(v["team_number"]) for v in video_match.get("teams", {}).values() if v.get("team_number")]

    def track_progress(message: str, pct: float) -> None:
        STORE.set_progress(job, "tracking", message, 55 + pct * 0.25)

    result = track_video(
        video_path,
        src_points=job.src_points,
        model_path=str(model) if model else None,
        team_numbers=team_numbers or None,
        frame_stride=3,
        crop_top=job.crop_top,
        crop_bottom=job.crop_bottom,
        on_progress=track_progress,
        tracker_mode=job.tracker_mode or "auto",
        openai_key=job.openai_key,
        google_key=job.google_key,
        use_bev=not bool(job.user_calibrated),
    )
    frame_path = dest / "calibration.jpg"
    save_jpeg(result["first_frame"], frame_path)
    samples = stitch_occlusions(result["samples"])
    blue, red = _alliance_teams(video_match)
    samples = keep_top_tracks(samples, max_tracks=max(len(blue) + len(red), 6) or 6)
    warnings = list(job.warnings) + list(result.get("warnings") or [])
    if not samples:
        warnings.append(
            "Tracking produced no robot paths. Click the four field corners on the broadcast frame, "
            "widen the field crop, try another tracking mode, or upload a clearer wide-angle VOD."
        )
    elif not result.get("used_model"):
        warnings.append(
            "Neural/cloud detectors did not contribute — OpenCV local strategies filled the path overlay. "
            "For stronger detection: install ultralytics, or add an OpenAI / Google API key."
        )

    # Side / lower cameras: scoring & climb reasoning (separate from overview paths).
    side_cues: list[dict[str, Any]] = []
    try:
        from ramscout.multicam import CameraLayout, CameraPane
        from ramscout.multiview import process_side_views

        cam = job.camera or {}
        panes = []
        for raw in (cam.get("panes") or (job.views or {}).get("panes") or []):
            panes.append(CameraPane(**{k: raw[k] for k in raw if k in CameraPane.__dataclass_fields__}))
        layout_obj = CameraLayout(
            mode=str(cam.get("mode") or "single"),
            crop_top=float(cam.get("crop_top") or job.crop_top),
            crop_bottom=float(cam.get("crop_bottom") or job.crop_bottom),
            confidence=float(cam.get("confidence") or 0.0),
            detail=str(cam.get("detail") or ""),
            split_y=cam.get("split_y"),
            split_x=cam.get("split_x"),
            panes=panes,
        )
        if layout_obj.side_panes():
            STORE.set_progress(job, "side_views", "Reading side cameras for scoring & climbs…", 82)

            def side_progress(message: str, pct: float) -> None:
                STORE.set_progress(job, "side_views", message, 82 + pct * 0.08)

            mv = process_side_views(video_path, layout_obj, on_progress=side_progress)
            side_cues = list(mv.side_cues or [])
            warnings.extend(mv.warnings or [])
            views = dict(job.views or {})
            views["side_cues"] = len(side_cues)
            STORE.update(job, views=views, side_cues=side_cues)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Side-view processing skipped: {exc}")

    assignments = {str(k): v for k, v in assign_by_start(samples, blue, red).items()} if samples else {}
    # Prefer OCR team labels when present.
    for sample in samples:
        if sample.get("team"):
            assignments[str(sample["track_id"])] = str(sample["team"])

    src_points = job.src_points
    if src_points is None:
        src_points = result.get("src_points")
    if src_points is None:
        fw, fh = result["frame_size"]
        src_points = default_source_points(fw, fh, job.crop_top, job.crop_bottom).tolist()

    STORE.update(
        job,
        samples=samples,
        assignments=assignments,
        warnings=warnings,
        used_model=bool(result.get("used_model")),
        frame_size=result.get("frame_size") or [1280, 720],
        frame_path=str(frame_path),
        src_points=src_points,
        seeds=_seed_boxes(samples),
        tracker_strategies=list(result.get("strategies") or []),
        source_hits=dict(result.get("source_hits") or {}),
        bev=dict(result.get("bev") or {}),
        side_cues=side_cues,
    )
    _reproject_and_scout(job)
    STORE.set_progress(job, "ready", "Auto-scout complete.", 100)


def apply_browser_tracks(
    job: Job,
    raw_samples: list[dict[str, Any]],
    *,
    replace: bool = False,
) -> Job:
    """Merge browser potato (JS frame-diff) samples into a finished job."""
    from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
    from ramscout.geometry import homography_from_corners, project_points

    cleaned: list[dict[str, Any]] = []
    for row in raw_samples or []:
        try:
            px = float(row["px"])
            py = float(row["py"])
            t = float(row.get("t") or 0.0)
            tid = int(row.get("track_id") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        alliance = row.get("alliance") if row.get("alliance") in {"red", "blue"} else (
            "blue" if px < (job.frame_size[0] if job.frame_size else 1280) * 0.5 else "red"
        )
        cleaned.append(
            {
                "t": round(t, 3),
                "frame": int(row.get("frame") or 0),
                "track_id": tid,
                "x": float(row.get("x") or 0.0),
                "y": float(row.get("y") or 0.0),
                "px": px,
                "py": py,
                "alliance": alliance,
                "team": str(row.get("team") or ""),
                "bbox": list(row.get("bbox") or []),
                "source": "browser_potato",
            }
        )
    if not cleaned:
        return job

    fw = int(job.frame_size[0]) if job.frame_size else 1280
    fh = int(job.frame_size[1]) if job.frame_size else 720
    src = job.src_points or default_source_points(fw, fh, job.crop_top, job.crop_bottom).tolist()
    try:
        H = homography_from_corners(src)
        feet = [[s["px"], s["py"]] for s in cleaned]
        mapped = project_points(feet, H)
        for sample, (mx, my) in zip(cleaned, mapped):
            sample["x"] = float(np.clip(mx, 0, FIELD_LENGTH))
            sample["y"] = float(np.clip(my, 0, FIELD_WIDTH))
    except Exception as exc:  # noqa: BLE001
        job.warnings.append(f"Browser potato projection failed: {exc}")
        # Linear fallback so the UI still gets something.
        for sample in cleaned:
            sample["x"] = float(np.clip(sample["px"] / max(fw, 1) * FIELD_LENGTH, 0, FIELD_LENGTH))
            sample["y"] = float(np.clip(sample["py"] / max(fh, 1) * FIELD_WIDTH, 0, FIELD_WIDTH))

    if replace or not job.samples:
        merged = cleaned
    else:
        merged = list(job.samples) + cleaned

    hits = dict(job.source_hits or {})
    hits["browser_potato"] = hits.get("browser_potato", 0) + len(cleaned)
    warnings = list(job.warnings or [])
    note = f"Browser potato added {len(cleaned)} motion samples (no AI)."
    if note not in warnings:
        warnings.append(note)
    STORE.update(
        job,
        samples=merged,
        source_hits=hits,
        warnings=warnings,
        used_model=False if replace else job.used_model,
    )
    _reproject_and_scout(job)
    return job


def _reproject_and_scout(job: Job) -> None:
    samples = list(job.samples)
    if job.src_points and samples:
        try:
            samples = reproject_samples(samples, job.src_points)
        except Exception as exc:  # noqa: BLE001
            job.warnings.append(f"Calibration ignored: {exc}")

    blue, red = _alliance_teams(job.match)
    if blue or red:
        # Never drop browser-potato tracks — they are the intentional dumb fallback.
        potato = [s for s in samples if s.get("source") == "browser_potato"]
        pruned = keep_top_tracks(samples, max_tracks=max(len(blue) + len(red), 6))
        kept_ids = {int(s["track_id"]) for s in pruned}
        potato_ids = {int(s["track_id"]) for s in potato}
        missing = potato_ids - kept_ids
        if missing:
            pruned.extend(s for s in potato if int(s["track_id"]) in missing)
        samples = pruned

    for sample in samples:
        tid = str(sample.get("track_id"))
        if tid in job.assignments:
            sample["team"] = job.assignments[tid]

    if (blue or red) and (not job.assignments or not any(s.get("team") for s in samples)):
        guessed = assign_by_start(samples, blue, red)
        STORE.update(job, assignments={str(k): v for k, v in guessed.items()})
        for sample in samples:
            tid = int(sample["track_id"])
            if tid in guessed:
                sample["team"] = guessed[tid]
    elif blue or red:
        # Refresh assignments for any unmapped tracks after pruning.
        guessed = assign_by_start(samples, blue, red)
        merged = dict(job.assignments)
        for tid, team in guessed.items():
            merged.setdefault(str(tid), team)
        STORE.update(job, assignments=merged)
        for sample in samples:
            tid = str(sample["track_id"])
            if tid in merged:
                sample["team"] = merged[tid]

    poses: list[Pose] = []
    grouped: dict[int, list[dict[str, Any]]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["track_id"]), []).append(sample)
    for tid, group in grouped.items():
        alliance = majority_alliance(group)
        team = str(group[0].get("team") or job.assignments.get(str(tid)) or "")
        if not team:
            # Last resort: still scout unnamed tracks so the UI is not empty.
            team = f"T{tid}"
        for sample in group:
            sample["alliance"] = sample.get("alliance") if sample.get("alliance") in {"red", "blue"} else alliance
            sample["team"] = team
            poses.append(
                Pose(
                    t=float(sample["t"]),
                    x=float(sample["x"]),
                    y=float(sample["y"]),
                    team=team,
                    alliance=sample["alliance"],
                    track_id=tid,
                )
            )

    nicknames = {}
    if job.match:
        for info in job.match.get("teams", {}).values():
            nicknames[str(info.get("team_number"))] = info.get("nickname") or ""

    events = detect_events(poses)
    cards = build_cards(poses, events, nicknames=nicknames)
    event_dicts = [e.as_dict() for e in events]
    card_dicts = [c.as_dict() for c in cards]
    try:
        from ramscout.event_editor import ensure_event_ids
        from ramscout.multiview import merge_side_cues_into_events
        from ramscout.tba_stats import enrich_cards_with_tba

        event_dicts = ensure_event_ids(event_dicts)
        if job.side_cues:
            event_dicts = merge_side_cues_into_events(event_dicts, job.side_cues, cards=card_dicts)
            event_dicts = ensure_event_ids(event_dicts)
            # Rebuild cards so side-confirmed hub/climb counts stick.
            from ramscout.event_editor import cards_from_events

            card_dicts = cards_from_events(card_dicts, event_dicts)
        card_dicts = enrich_cards_with_tba(card_dicts, job.match)
    except Exception as exc:  # noqa: BLE001
        warnings = list(job.warnings or [])
        warnings.append(f"TBA card enrich skipped: {exc}")
        STORE.update(job, warnings=warnings)
    STORE.update(
        job,
        samples=samples,
        events=event_dicts,
        cards=card_dicts,
        seeds=_seed_boxes(samples) or job.seeds,
        status="ready" if job.status in {"ready", "scouting", "tracking", "side_views", "views"} else job.status,
    )
    _persist(job)


def _alliance_teams(match: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    if not match:
        return [], []
    alliances = match.get("alliances") or {}
    def nums(color: str) -> list[str]:
        keys = alliances.get(color, {}).get("team_keys") or []
        return [k.replace("frc", "") for k in keys]
    return nums("blue"), nums("red")


def _seed_boxes(samples: list[dict[str, Any]], t_max: float = 3.0) -> list[dict[str, Any]]:
    seeds: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if float(sample.get("t") or 0) > t_max:
            continue
        tid = str(sample.get("track_id"))
        if tid in seeds or not sample.get("bbox"):
            continue
        seeds[tid] = {
            "track_id": tid,
            "bbox": sample.get("bbox"),
            "alliance": sample.get("alliance") or "unknown",
            "team": str(sample.get("team") or ""),
        }
    return list(seeds.values())


def _persist(job: Job) -> None:
    dest = DATA / job.id / "result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = job.public()
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    try:
        from ramscout.history import record_job_summary

        record_job_summary(payload)
    except Exception:
        pass


def export_csv(job: Job) -> str:
    lines = ["team,alliance,nickname,hub_score_candidates,defense_time_s,climb_attempt,collection_time_s,path_length_in,avg_speed_in_s"]
    for card in job.cards:
        lines.append(
            ",".join(
                [
                    str(card.get("team", "")),
                    str(card.get("alliance", "")),
                    json.dumps(card.get("nickname") or ""),
                    str(card.get("hub_score_candidates", 0)),
                    str(card.get("defense_time_s", 0)),
                    str(card.get("climb_attempt", False)),
                    str(card.get("collection_time_s", 0)),
                    str(card.get("path_length_in", 0)),
                    str(card.get("avg_speed_in_s", 0)),
                ]
            )
        )
    lines.append("")
    lines.append("team,type,t,zone,confidence,detail")
    for event in job.events:
        detail = json.dumps(event.get("detail") or "")
        lines.append(
            f"{event.get('team')},{event.get('type')},{event.get('t')},{event.get('zone')},{event.get('confidence')},{detail}"
        )
    return "\n".join(lines) + "\n"
