"""RoboScoutAI local web server."""

from __future__ import annotations

import os
import re
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ramscout import __version__, managed
from ramscout.detect import TRACKER_MODES, list_strategies
from ramscout.gameconfig import public_game
from ramscout.ingest import is_youtube_url, resolve_cookies_path
from ramscout.paths import is_frozen, jobs_dir, web_dir
from ramscout.picklist import (
    aggregate_cards,
    alliance_summary,
    compare_teams,
    suggest_picks,
)
from ramscout.suite_api import register_suite_routes
from ramscout.pipeline import (
    STORE,
    apply_assignments,
    apply_browser_tracks,
    apply_calibration,
    export_csv,
    start_job,
)
from ramscout.updater import (
    apply_update_now,
    check_for_update,
    git_branch,
    git_remote,
    github_repo,
    last_check,
    platform_key,
)

WEB = web_dir()
_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}
_SAFE_UPLOAD_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_upload_name(filename: str | None, suffix: str) -> str:
    raw = Path(filename or f"upload{suffix}").name
    cleaned = _SAFE_UPLOAD_NAME.sub("_", raw).strip("._") or f"upload{suffix}"
    if not Path(cleaned).suffix:
        cleaned = f"{cleaned}{suffix}"
    return cleaned


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def worker() -> None:
        try:
            if managed.is_managed():
                managed.check_for_update()
            else:
                check_for_update()
        except Exception:  # noqa: BLE001
            pass
        try:
            from ramscout.detect import ensure_detector_weights

            ensure_detector_weights()
        except Exception:  # noqa: BLE001
            pass

    threading.Thread(target=worker, daemon=True).start()
    yield


app = FastAPI(title="RoboScoutAI", version=__version__, lifespan=lifespan)
register_suite_routes(app)
app.mount("/static", StaticFiles(directory=WEB), name="static")


class StartRequest(BaseModel):
    url: str = ""
    tba_key: str = ""
    event_key: str = ""
    match_key: str = ""
    demo: bool = False
    crop_top: float = 0.10
    crop_bottom: float = 0.65
    tracker_mode: str = "auto"
    openai_key: str = ""
    google_key: str = ""
    ai_gateway_key: str = ""
    auto_multicam: bool = True
    top_overview_only: bool = True
    crop_locked: bool = False



class CalibrateRequest(BaseModel):
    src_points: list[list[float]] = Field(min_length=4, max_length=4)


class AssignRequest(BaseModel):
    assignments: dict[str, str]


class BrowserTracksRequest(BaseModel):
    """Pixel-space feet from the browser potato tracker (no AI)."""

    samples: list[dict] = Field(default_factory=list)
    replace: bool = False


class PicklistRequest(BaseModel):
    cards: list[dict] = Field(default_factory=list)
    already_picked: list[int] = Field(default_factory=list)
    limit: int = 24


class CompareRequest(BaseModel):
    cards: list[dict] = Field(default_factory=list)
    teams: list[int] = Field(default_factory=list)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict:
    from ramscout.deps import check_deps

    deps = check_deps()
    return {
        "status": "ok" if deps.get("ok") else "degraded",
        "version": __version__,
        "deps_ok": bool(deps.get("ok")),
        "missing_required": list(deps.get("missing_required") or []),
        "missing_optional": list(deps.get("missing_optional") or []),
        "install": list(deps.get("install") or []),
        "detector": deps.get("detector") or {},
        "notes": list(deps.get("notes") or []),
        "scout_model": _scout_model_status(),
    }


@app.get("/api/version")
def version() -> dict:
    is_managed = managed.is_managed()
    cached = managed.last_check() if is_managed else last_check()
    cookies = resolve_cookies_path()
    return {
        "version": __version__,
        "frozen": is_frozen(),
        "platform": platform_key(),
        "repo": github_repo(),
        "git_remote": git_remote(),
        "git_branch": (managed.update_channel() or git_branch()) if is_managed else git_branch(),
        "managed": is_managed,
        "manager": managed.info(),
        "update": cached,
        "cookies": {
            "found": cookies is not None,
            "path": str(cookies) if cookies else None,
        },
    }


@app.get("/api/updates/check")
def updates_check() -> dict:
    if managed.is_managed():
        return managed.check_for_update()
    return check_for_update().as_dict()


@app.get("/api/updates/status")
def updates_status() -> dict:
    """Progress of a manager-driven update (side-by-side download/verify/install)."""
    return managed.status()


@app.post("/api/updates/relaunch")
def updates_relaunch() -> dict:
    """Ask the manager to restart us on the freshly installed version."""
    result = managed.request_relaunch()
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("message") or "not managed")
    return result


@app.post("/api/updates/download")
def updates_download() -> dict:
    """Fetch and apply an update: via the manager when managed, else the git updater."""
    if managed.is_managed():
        return managed.start_update()
    try:
        result = apply_update_now()
    except Exception as exc:  # noqa: BLE001
        # Last-resort manual fallback only — prefer newest release, never auto-open on success.
        from ramscout.updater import manual_download_url

        return {
            "ok": False,
            "open_url": manual_download_url(),
            "message": (
                f"Git update failed ({exc}). "
                "Open the newest RoboScoutAI-windows-x64.exe from Releases (or the local "
                "update-cache) and run it — More info → Run anyway if SmartScreen blocks."
            ),
            "error": str(exc),
            "restarting": False,
        }
    # Successful git apply must never include open_url (UI would browser-download).
    if result.get("ok"):
        result.pop("open_url", None)
        return result
    update = result.get("update") or {}
    if not result.get("open_url"):
        # Prefer a versioned/latest Releases link over a bare git browse URL.
        from ramscout.updater import manual_download_url

        latest = ""
        if isinstance(update, dict):
            latest = str(update.get("latest_version") or "")
        asset = ""
        if isinstance(update, dict):
            asset = str(update.get("asset_name") or "")
            if asset.startswith("git:"):
                asset = ""
        result["open_url"] = manual_download_url(version=latest, asset_name=asset)
    return result


@app.get("/api/game")
def game_config(year: int | None = None) -> dict:
    return public_game(year)


@app.get("/api/trackers")
def trackers() -> dict:
    """List tracking modes and which strategies are available on this machine."""
    strategies = list_strategies()
    avail = {row["name"]: bool(row["available"]) for row in strategies}
    modes = []
    for key, meta in TRACKER_MODES.items():
        wanted = list(meta["strategies"])
        missing = [s for s in wanted if not avail.get(s, False)]
        special = [s for s in wanted if s in {"yolo", "openai", "gemini"}]
        # "Full" when every special strategy the mode asks for is present;
        # potato strategies are always available so a mode never fully fails.
        status = "full" if not missing else ("partial" if len(missing) < len(special) or not special else "fallback")
        modes.append(
            {
                "id": key,
                "label": meta["label"],
                "description": meta["description"],
                "strategies": wanted,
                "missing": missing,
                "status": status,
                "default": key == "auto",
                "benchmark": bool(meta.get("benchmark")),
            }
        )
    try:
        from ramscout.depth import depth_backends

        depth = depth_backends()
    except Exception as exc:  # noqa: BLE001
        depth = {"error": str(exc)}
    try:
        from ramscout.onnx_detector import detector_status

        detector = detector_status()
    except Exception as exc:  # noqa: BLE001
        detector = {"error": str(exc), "frc_ready": False}
    try:
        from ramscout.deps import check_deps

        deps = check_deps()
    except Exception as exc:  # noqa: BLE001
        deps = {"ok": False, "error": str(exc)}
    return {
        "modes": modes,
        "strategies": strategies,
        "depth": depth,
        "detector": detector,
        "deps": {
            "ok": bool(deps.get("ok")),
            "missing_required": list(deps.get("missing_required") or []),
            "missing_optional": list(deps.get("missing_optional") or []),
            "install": list(deps.get("install") or []),
            "required": deps.get("required") or [],
            "optional": deps.get("optional") or [],
        },
        "scout_model": _scout_model_status(),
        "env_hints": {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY or GEMINI_API_KEY",
            "yolo": "pip install ultralytics (+ optional models/*.pt) — training only; runtime prefers ONNX",
            "onnx": "pip install onnxruntime (+ models/robot.onnx) — packaged local detector",
            "install": "pip install -r requirements.txt && python -m ramscout.deps --strict",
            "jev": "AI_GATEWAY_API_KEY (Vercel AI Gateway → typesafe-ai/jev)",
        },
    }


def _normalize_crop(crop_top: float, crop_bottom: float) -> tuple[float, float]:
    """Normalize overview crop. Bottom may be < 0.5 for a tight top widescreen lock."""
    if crop_bottom <= crop_top:
        raise HTTPException(400, "Crop bottom must be below crop top.")
    top = min(max(crop_top, 0.0), 0.45)
    bottom = min(max(crop_bottom, 0.25), 1.0)
    if bottom <= top:
        raise HTTPException(400, "Crop bottom must be below crop top.")
    return top, bottom


@app.post("/api/jobs")
def create_job(body: StartRequest) -> dict:
    if body.demo:
        job = start_job(url=body.url or "demo://sample", demo=True, tba_key=body.tba_key)
        return job.public()
    if not body.url.strip():
        raise HTTPException(400, "Paste a YouTube match video URL or upload a local file.")
    top, bottom = _normalize_crop(body.crop_top, body.crop_bottom)
    job = start_job(
        url=body.url.strip(),
        tba_key=body.tba_key.strip() or os.environ.get("TBA_AUTH_KEY", ""),
        event_key=body.event_key.strip(),
        match_key=body.match_key.strip(),
        crop_top=top,
        crop_bottom=bottom,
        tracker_mode=body.tracker_mode.strip() or "auto",
        openai_key=body.openai_key.strip() or os.environ.get("OPENAI_API_KEY", ""),
        google_key=body.google_key.strip()
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", ""),
        ai_gateway_key=body.ai_gateway_key.strip()
        or os.environ.get("AI_GATEWAY_API_KEY", "")
        or os.environ.get("VERCEL_AI_GATEWAY_API_KEY", ""),
        auto_multicam=bool(body.auto_multicam),
        top_overview_only=bool(body.top_overview_only),
        crop_locked=bool(body.crop_locked),
    )
    return job.public()


@app.post("/api/jobs/upload")
async def create_job_upload(
    file: UploadFile = File(...),
    url: str = Form(""),
    tba_key: str = Form(""),
    event_key: str = Form(""),
    match_key: str = Form(""),
    crop_top: float = Form(0.10),
    crop_bottom: float = Form(0.65),
    tracker_mode: str = Form("auto"),
    openai_key: str = Form(""),
    google_key: str = Form(""),
    ai_gateway_key: str = Form(""),
    auto_multicam: bool = Form(True),
    top_overview_only: bool = Form(True),
    crop_locked: bool = Form(False),
) -> dict:
    """Analyze an already-downloaded match VOD (bypasses YouTube bot checks)."""
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(400, "Upload an mp4/mkv/webm/mov match video.")
    top, bottom = _normalize_crop(crop_top, crop_bottom)

    staging = jobs_dir() / "_uploads"
    staging.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_upload_name(file.filename, suffix)
    tmp_path = staging / f"{uuid.uuid4().hex}_{safe_name}"
    size = 0
    try:
        with tmp_path.open("wb") as fh:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                fh.write(chunk)
        if size < 1024:
            raise HTTPException(
                400,
                "Uploaded file is empty or too small. Choose a real MP4/MKV match VOD.",
            )
        yt_url = (url or "").strip()
        # Prefer YouTube URL only for metadata; local file is the media source.
        meta_url = yt_url if is_youtube_url(yt_url) else str(tmp_path.resolve())
        job = start_job(
            url=meta_url,
            tba_key=(tba_key or "").strip() or os.environ.get("TBA_AUTH_KEY", ""),
            event_key=(event_key or "").strip(),
            match_key=(match_key or "").strip(),
            crop_top=top,
            crop_bottom=bottom,
            local_video=tmp_path,
            tracker_mode=(tracker_mode or "auto").strip() or "auto",
            openai_key=(openai_key or "").strip() or os.environ.get("OPENAI_API_KEY", ""),
            google_key=(google_key or "").strip()
            or os.environ.get("GOOGLE_API_KEY", "")
            or os.environ.get("GEMINI_API_KEY", ""),
            ai_gateway_key=(ai_gateway_key or "").strip()
            or os.environ.get("AI_GATEWAY_API_KEY", "")
            or os.environ.get("VERCEL_AI_GATEWAY_API_KEY", ""),
            auto_multicam=bool(auto_multicam),
            top_overview_only=bool(top_overview_only),
            crop_locked=bool(crop_locked),
        )
    except HTTPException:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    else:
        # start_job copies into the job folder synchronously; staging file can go.
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
    return job.public()


def _scout_model_status() -> dict:
    try:
        from ramscout.laya import scout_model_status

        return scout_model_status()
    except Exception as exc:  # noqa: BLE001
        return {
            "name": "local scout model",
            "installed": False,
            "state": "not installed",
            "install_command": "pip install -r requirements-laya.txt",
            "error": str(exc),
        }


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if job.status in {"ready", "error", "cancelled"}:
        return job.public()
    STORE.update(job, cancel_requested=True, message="Cancelling…")
    return job.public()


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job.public()


@app.post("/api/jobs/{job_id}/calibrate")
def calibrate(job_id: str, body: CalibrateRequest) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return apply_calibration(job, body.src_points).public()


@app.post("/api/jobs/{job_id}/assign")
def assign(job_id: str, body: AssignRequest) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return apply_assignments(job, body.assignments).public()


@app.post("/api/jobs/{job_id}/browser-tracks")
def browser_tracks(job_id: str, body: BrowserTracksRequest) -> dict:
    """Ingest dumb browser frame-diff tracks (potato fallback, no models)."""
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    if job.status not in {"ready", "tracking", "scouting"}:
        raise HTTPException(400, "Job is not ready for browser tracks yet.")
    return apply_browser_tracks(job, body.samples, replace=body.replace).public()


@app.get("/api/jobs/{job_id}/export.json")
def export_json(job_id: str) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return job.public()


@app.get("/api/jobs/{job_id}/export.csv")
def export_csv_file(job_id: str) -> PlainTextResponse:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return PlainTextResponse(export_csv(job), media_type="text/csv")


@app.get("/api/jobs/{job_id}/video")
def job_video(job_id: str) -> FileResponse:
    job = STORE.get(job_id)
    if job is None or not job.video_path:
        raise HTTPException(404, "No video for this job.")
    return FileResponse(job.video_path)


@app.get("/api/jobs/{job_id}/frame")
def job_frame(job_id: str) -> FileResponse:
    job = STORE.get(job_id)
    if job is None or not job.frame_path:
        raise HTTPException(404, "No calibration frame for this job.")
    return FileResponse(job.frame_path)


@app.post("/api/picklist")
def picklist(body: PicklistRequest) -> dict:
    cards = aggregate_cards(body.cards) if body.cards else []
    return suggest_picks(cards, already_picked=body.already_picked, limit=body.limit)


@app.post("/api/compare")
def compare(body: CompareRequest) -> dict:
    if not body.teams:
        raise HTTPException(400, "Provide at least one team number.")
    cards = aggregate_cards(body.cards) if body.cards else []
    return {
        "compare": compare_teams(cards, body.teams),
        "alliance": alliance_summary(cards, body.teams[:3]),
    }


@app.get("/api/jobs/{job_id}/picklist")
def job_picklist(job_id: str, limit: int = 12) -> dict:
    job = STORE.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job.")
    return suggest_picks(job.cards or [], limit=limit)


if __name__ == "__main__":
    from desktop.main import main

    raise SystemExit(main())
