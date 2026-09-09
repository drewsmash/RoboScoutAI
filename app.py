"""RamScoutAI local web server."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ramscout.gameconfig import public_game
from ramscout.pipeline import (
    STORE,
    apply_assignments,
    apply_calibration,
    export_csv,
    start_job,
)

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"

app = FastAPI(title="RamScoutAI", version="0.2.0")
app.mount("/static", StaticFiles(directory=WEB), name="static")


class StartRequest(BaseModel):
    url: str = ""
    tba_key: str = ""
    event_key: str = ""
    match_key: str = ""
    demo: bool = False


class CalibrateRequest(BaseModel):
    src_points: list[list[float]] = Field(min_length=4, max_length=4)


class AssignRequest(BaseModel):
    assignments: dict[str, str]


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/game")
def game_config(year: int | None = None) -> dict:
    return public_game(year)


@app.post("/api/jobs")
def create_job(body: StartRequest) -> dict:
    if body.demo:
        job = start_job(url=body.url or "demo://sample", demo=True, tba_key=body.tba_key)
        return job.public()
    if not body.url.strip():
        raise HTTPException(400, "Paste a YouTube match video URL.")
    job = start_job(
        url=body.url.strip(),
        tba_key=body.tba_key.strip() or os.environ.get("TBA_AUTH_KEY", ""),
        event_key=body.event_key.strip(),
        match_key=body.match_key.strip(),
    )
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), reload=True)
