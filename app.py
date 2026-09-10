"""RamScoutAI local web server."""

from __future__ import annotations

import os
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ramscout import __version__
from ramscout.detect import TRACKER_MODES, list_strategies
from ramscout.gameconfig import public_game
from ramscout.paths import is_frozen, web_dir
from ramscout.picklist import (
    aggregate_cards,
    alliance_summary,
    compare_teams,
    suggest_picks,
)
from ramscout.pipeline import (
    STORE,
    apply_assignments,
    apply_calibration,
    export_csv,
    start_job,
)
from ramscout.updater import (
    UpdateInfo,
    apply_downloaded_update,
    check_for_update,
    download_update,
    github_repo,
    last_check,
    platform_key,
)

WEB = web_dir()
_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".webm", ".mov", ".avi"}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    def worker() -> None:
        try:
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


app = FastAPI(title="RamScoutAI", version=__version__, lifespan=lifespan)
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


class CalibrateRequest(BaseModel):
    src_points: list[list[float]] = Field(min_length=4, max_length=4)


class AssignRequest(BaseModel):
    assignments: dict[str, str]


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
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/version")
def version() -> dict:
    cached = last_check()
    return {
        "version": __version__,
        "frozen": is_frozen(),
        "platform": platform_key(),
        "repo": github_repo(),
        "update": cached,
    }


@app.get("/api/updates/check")
def updates_check() -> dict:
    return check_for_update().as_dict()


@app.post("/api/updates/download")
def updates_download() -> dict:
    info = UpdateInfo(**(last_check() or check_for_update().as_dict()))
    if not info.available:
        raise HTTPException(400, info.error or "No update available.")
    if not info.asset_url:
        return {
            "ok": False,
            "open_url": info.release_url,
            "message": "Open the GitHub release page to download this update.",
            "update": info.as_dict(),
        }
    if not is_frozen():
        return {
            "ok": False,
            "open_url": info.release_url,
            "message": "Source installs update with git pull. Desktop builds can auto-apply.",
            "update": info.as_dict(),
        }
    try:
        package = download_update(info)
        message = apply_downloaded_update(package)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, str(exc)) from exc
    return {"ok": True, "message": message, "update": info.as_dict(), "restarting": True}


@app.get("/api/game")
def game_config(year: int | None = None) -> dict:
    return public_game(year)


@app.get("/api/trackers")
def trackers() -> dict:
    """List tracking modes and which strategies are available on this machine."""
    return {
        "modes": [
            {"id": key, "label": meta["label"], "description": meta["description"], "strategies": meta["strategies"]}
            for key, meta in TRACKER_MODES.items()
        ],
        "strategies": list_strategies(),
        "env_hints": {
            "openai": "OPENAI_API_KEY",
            "google": "GOOGLE_API_KEY or GEMINI_API_KEY",
            "yolo": "pip install ultralytics (+ optional models/*.pt)",
        },
    }


def _normalize_crop(crop_top: float, crop_bottom: float) -> tuple[float, float]:
    if crop_bottom <= crop_top:
        raise HTTPException(400, "Crop bottom must be below crop top.")
    top = min(max(crop_top, 0.0), 0.45)
    bottom = min(max(crop_bottom, 0.5), 1.0)
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
) -> dict:
    """Analyze an already-downloaded match VOD (bypasses YouTube bot checks)."""
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(400, "Upload an mp4/mkv/webm/mov match video.")
    top, bottom = _normalize_crop(crop_top, crop_bottom)
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    meta_url = (url or "").strip() or f"file://{tmp_path}"
    try:
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
        )
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
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
