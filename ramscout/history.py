"""Persisted job history index (reopen past analyses)."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ramscout.paths import data_dir, jobs_dir

_LOCK = threading.Lock()


def _index_path() -> Path:
    path = data_dir() / "job_index.json"
    if not path.exists():
        path.write_text("[]", encoding="utf-8")
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_job_summary(job_public: dict[str, Any]) -> None:
    row = {
        "id": job_public.get("id"),
        "created_at": job_public.get("created_at") or _now(),
        "updated_at": _now(),
        "status": job_public.get("status"),
        "url": job_public.get("url"),
        "demo": bool(job_public.get("demo")),
        "match_key": (job_public.get("match") or {}).get("key"),
        "event_key": (job_public.get("match") or {}).get("event_key") or job_public.get("event_key"),
        "title": ((job_public.get("video_info") or {}).get("title") or ""),
        "warnings": len(job_public.get("warnings") or []),
        "teams": _teams(job_public),
        "camera": (job_public.get("camera") or {}).get("mode"),
        "crop_top": job_public.get("crop_top"),
        "crop_bottom": job_public.get("crop_bottom"),
    }
    with _LOCK:
        rows = json.loads(_index_path().read_text(encoding="utf-8"))
        rows = [r for r in rows if r.get("id") != row["id"]]
        rows.insert(0, row)
        _index_path().write_text(json.dumps(rows[:200], indent=2), encoding="utf-8")


def list_history(limit: int = 50) -> list[dict[str, Any]]:
    with _LOCK:
        rows = json.loads(_index_path().read_text(encoding="utf-8"))
    # Also surface on-disk result.json jobs not yet indexed.
    known = {r.get("id") for r in rows}
    for path in sorted(jobs_dir().glob("*/result.json"), reverse=True):
        job_id = path.parent.name
        if job_id in known:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        rows.append(
            {
                "id": job_id,
                "created_at": data.get("created_at"),
                "updated_at": data.get("created_at"),
                "status": data.get("status") or "ready",
                "url": data.get("url"),
                "demo": bool(data.get("demo")),
                "match_key": (data.get("match") or {}).get("key"),
                "event_key": (data.get("match") or {}).get("event_key"),
                "title": ((data.get("video_info") or {}).get("title") or ""),
                "teams": _teams(data),
                "from_disk": True,
            }
        )
    rows.sort(key=lambda r: str(r.get("updated_at") or r.get("created_at") or ""), reverse=True)
    return rows[: max(1, limit)]


def load_job_from_disk(job_id: str) -> dict[str, Any] | None:
    path = jobs_dir() / job_id / "result.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def _teams(job: dict[str, Any]) -> list[str]:
    match = job.get("match") or {}
    teams = []
    for color in ("blue", "red"):
        for key in (match.get("alliances") or {}).get(color, {}).get("team_keys") or []:
            teams.append(str(key).replace("frc", ""))
    if not teams:
        for card in job.get("cards") or []:
            if card.get("team"):
                teams.append(str(card["team"]))
    return teams
