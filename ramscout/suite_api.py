"""Extra scouting-suite API routes (draft, pit, history, multicam tools, etc.)."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from ramscout.gameconfig import available_years, public_game
from ramscout.pipeline import STORE, Job


class EventEditRequest(BaseModel):
    add: list[dict] = Field(default_factory=list)
    remove_ids: list[str] = Field(default_factory=list)
    update: list[dict] = Field(default_factory=list)
    reject_ids: list[str] = Field(default_factory=list)
    confirm_ids: list[str] = Field(default_factory=list)


class DraftRequest(BaseModel):
    cards: list[dict] = Field(default_factory=list)
    role: str = "balanced"
    picked: list[int] = Field(default_factory=list)
    do_not_pick: list[int] = Field(default_factory=list)
    locked: list[int] = Field(default_factory=list)
    epa_by_team: dict[str, float] = Field(default_factory=dict)


class EventBatchRequest(BaseModel):
    tba_key: str = ""
    event_key: str
    only_with_video: bool = True
    limit: int = 40


class ScheduleRequest(BaseModel):
    tba_key: str = ""
    event_key: str
    watch_teams: list[int] = Field(default_factory=list)


class DossierRequest(BaseModel):
    cards: list[dict] = Field(default_factory=list)
    team: int


class PitFormRequest(BaseModel):
    team: str
    drivetrain: str = ""
    weight_lb: float | None = None
    length_in: float | None = None
    width_in: float | None = None
    scoring: str = ""
    intake: str = ""
    climb: str = ""
    autos: str = ""
    defense: str = ""
    foul_risk: str = ""
    do_not_pick: bool = False
    photo_url: str = ""
    notes: str = ""
    scout: str = ""


class ScoutBookRequest(BaseModel):
    matches: list[dict] = Field(default_factory=list)
    cards: list[dict] = Field(default_factory=list)
    notes: dict = Field(default_factory=dict)
    watchlist: list = Field(default_factory=list)


class LiveRequest(BaseModel):
    device: int = 0


class EpaRequest(BaseModel):
    teams: list[int] = Field(default_factory=list)
    year: int = 2026


class SheetsCardsRequest(BaseModel):
    cards: list[dict] = Field(default_factory=list)


def register_suite_routes(app: FastAPI) -> None:
    @app.get("/api/games")
    def list_games() -> dict[str, Any]:
        years = available_years()
        return {"years": years, "games": [public_game(y) for y in years], "default": years[-1] if years else 2026}

    @app.post("/api/jobs/{job_id}/events")
    def edit_events(job_id: str, body: EventEditRequest) -> dict:
        from ramscout.event_editor import apply_event_edits, cards_from_events
        from ramscout.pipeline import _persist

        job = STORE.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        events = apply_event_edits(
            job.events or [],
            add=body.add,
            remove_ids=body.remove_ids,
            update=body.update,
            reject_ids=body.reject_ids,
            confirm_ids=body.confirm_ids,
        )
        cards = cards_from_events(job.cards or [], events)
        STORE.update(job, events=events, cards=cards, edited_events=True)
        try:
            _persist(job)
        except Exception:
            pass
        return job.public()

    @app.get("/api/jobs/{job_id}/heatmap")
    def job_heatmap(job_id: str, team: str | None = None, period: str | None = None) -> dict:
        from ramscout.heatmap import build_heatmap

        job = STORE.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job.")
        return build_heatmap(job.samples or [], team=team, period=period)

    @app.post("/api/jobs/{job_id}/annotate")
    def job_annotate(job_id: str, max_seconds: float = 45.0) -> dict:
        from ramscout.annotate import export_annotated_clip
        from ramscout.paths import jobs_dir

        job = STORE.get(job_id)
        if job is None or not job.video_path:
            raise HTTPException(404, "No video for this job.")
        out = jobs_dir() / job_id / "annotated.mp4"
        result = export_annotated_clip(
            job.video_path,
            out,
            job.samples or [],
            crop_top=getattr(job, "crop_top", 0.10),
            crop_bottom=getattr(job, "crop_bottom", 0.65),
            max_seconds=max_seconds,
        )
        return {**result, "url": f"/api/jobs/{job_id}/annotated"}

    @app.get("/api/jobs/{job_id}/annotated")
    def job_annotated_file(job_id: str) -> FileResponse:
        from ramscout.paths import jobs_dir

        path = jobs_dir() / job_id / "annotated.mp4"
        if not path.exists():
            raise HTTPException(404, "Annotated clip not generated yet.")
        return FileResponse(path, media_type="video/mp4", filename=f"{job_id}-annotated.mp4")

    @app.post("/api/draft")
    def draft_board(body: DraftRequest) -> dict:
        from ramscout.draftkit import alliance_fit, draft_board_state

        if body.locked:
            return alliance_fit(body.cards, body.locked, epa_by_team=body.epa_by_team)
        return draft_board_state(
            body.cards,
            picked=body.picked,
            do_not_pick=body.do_not_pick,
            role=body.role,
        )

    @app.post("/api/event/batch")
    def event_batch(body: EventBatchRequest) -> dict:
        from ramscout.eventops import batch_plan_from_event

        key = body.tba_key.strip() or os.environ.get("TBA_AUTH_KEY", "")
        if not key:
            raise HTTPException(400, "TBA auth key required for event batch planning.")
        return batch_plan_from_event(
            key,
            body.event_key.strip(),
            only_with_video=body.only_with_video,
            limit=body.limit,
        )

    @app.post("/api/event/schedule")
    def event_schedule(body: ScheduleRequest) -> dict:
        from ramscout.eventops import watchlist_schedule
        from ramscout.tba import TBAClient, TBAError

        key = body.tba_key.strip() or os.environ.get("TBA_AUTH_KEY", "")
        if not key:
            raise HTTPException(400, "TBA auth key required.")
        try:
            with TBAClient(key) as client:
                return watchlist_schedule(client, body.event_key.strip(), body.watch_teams)
        except TBAError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post("/api/dossier")
    def dossier(body: DossierRequest) -> dict:
        from ramscout.eventops import team_dossier

        return team_dossier(body.cards, body.team)

    @app.get("/api/history")
    def history(limit: int = 50) -> dict:
        from ramscout.history import list_history

        return {"jobs": list_history(limit=limit)}

    @app.get("/api/history/{job_id}")
    def history_job(job_id: str) -> dict:
        from ramscout.history import load_job_from_disk

        live = STORE.get(job_id)
        if live is not None:
            return live.public()
        data = load_job_from_disk(job_id)
        if not data:
            raise HTTPException(404, "Job not found on disk.")
        job = Job(id=job_id, url=data.get("url") or "", created_at=data.get("created_at") or "")
        for key, value in data.items():
            if hasattr(job, key) and key != "id":
                try:
                    setattr(job, key, value)
                except Exception:
                    continue
        STORE._jobs[job_id] = job  # noqa: SLF001
        return job.public()

    @app.get("/api/pit")
    def pit_list() -> dict:
        from ramscout.scoutbook import list_pit_forms

        return list_pit_forms()

    @app.post("/api/pit")
    def pit_upsert(body: PitFormRequest) -> dict:
        from ramscout.scoutbook import upsert_pit_form

        try:
            return upsert_pit_form(body.model_dump())
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.get("/api/scoutbook")
    def scoutbook_get() -> dict:
        from ramscout.scoutbook import load_scout_book

        return load_scout_book()

    @app.put("/api/scoutbook")
    def scoutbook_put(body: ScoutBookRequest) -> dict:
        from ramscout.scoutbook import save_scout_book

        return save_scout_book(body.model_dump())

    @app.post("/api/scoutbook/merge")
    def scoutbook_merge(body: ScoutBookRequest) -> dict:
        from ramscout.scoutbook import merge_scout_book

        return merge_scout_book(body.model_dump())

    @app.post("/api/sheets/cards.csv")
    def sheets_cards(body: SheetsCardsRequest) -> PlainTextResponse:
        from ramscout.picklist import aggregate_cards
        from ramscout.sheets import cards_to_sheets_csv

        cards = aggregate_cards(body.cards) if body.cards else []
        return PlainTextResponse(cards_to_sheets_csv(cards), media_type="text/csv")

    @app.post("/api/sheets/picklist.csv")
    def sheets_picklist(body: DraftRequest) -> PlainTextResponse:
        from ramscout.draftkit import score_with_role
        from ramscout.sheets import picklist_to_sheets_csv

        ranked = score_with_role(body.cards, role=body.role, epa_by_team=body.epa_by_team)
        return PlainTextResponse(picklist_to_sheets_csv(ranked), media_type="text/csv")

    @app.post("/api/epa")
    def epa_lookup(body: EpaRequest) -> dict:
        from ramscout.epa import fetch_epa_map

        return {"year": body.year, "epa": fetch_epa_map(body.teams, year=body.year)}

    @app.get("/api/live")
    def live_get() -> dict:
        from ramscout.live import live_status

        return live_status()

    @app.post("/api/live/start")
    def live_start(body: LiveRequest) -> dict:
        from ramscout.live import start_live

        return start_live(device=body.device)

    @app.post("/api/live/stop")
    def live_stop() -> dict:
        from ramscout.live import stop_live

        return stop_live()
