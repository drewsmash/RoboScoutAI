"""Pit scouting forms and shared scout-book persistence."""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ramscout.paths import data_dir

_LOCK = threading.Lock()

PIT_FIELDS = (
    "team",
    "drivetrain",
    "weight_lb",
    "length_in",
    "width_in",
    "scoring",
    "intake",
    "climb",
    "autos",
    "defense",
    "foul_risk",
    "do_not_pick",
    "photo_url",
    "notes",
    "scout",
)


def _pit_path() -> Path:
    path = data_dir() / "pit_forms.json"
    if not path.exists():
        path.write_text("{}", encoding="utf-8")
    return path


def _book_path() -> Path:
    path = data_dir() / "scout_book.json"
    if not path.exists():
        path.write_text(
            json.dumps({"matches": [], "cards": [], "notes": {}, "watchlist": [], "updated_at": None}, indent=2),
            encoding="utf-8",
        )
    return path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def list_pit_forms() -> dict[str, Any]:
    with _LOCK:
        data = json.loads(_pit_path().read_text(encoding="utf-8"))
    return {"forms": data, "count": len(data)}


def upsert_pit_form(payload: dict[str, Any]) -> dict[str, Any]:
    team = str(payload.get("team") or "").replace("frc", "").strip()
    if not team:
        raise ValueError("team is required")
    row = {key: payload.get(key) for key in PIT_FIELDS}
    row["team"] = team
    row["do_not_pick"] = bool(payload.get("do_not_pick"))
    row["updated_at"] = _now()
    with _LOCK:
        data = json.loads(_pit_path().read_text(encoding="utf-8"))
        data[team] = row
        _pit_path().write_text(json.dumps(data, indent=2), encoding="utf-8")
    return row


def load_scout_book() -> dict[str, Any]:
    with _LOCK:
        return json.loads(_book_path().read_text(encoding="utf-8"))


def save_scout_book(book: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "matches": list(book.get("matches") or []),
        "cards": list(book.get("cards") or []),
        "notes": dict(book.get("notes") or {}),
        "watchlist": list(book.get("watchlist") or []),
        "updated_at": _now(),
    }
    with _LOCK:
        _book_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def merge_scout_book(incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge another device's export into the on-disk scout book."""
    current = load_scout_book()
    cards = list(current.get("cards") or [])
    seen = {
        (str(c.get("team")), str(c.get("match_key") or c.get("job_id") or len(cards)))
        for c in cards
    }
    for card in incoming.get("cards") or []:
        key = (str(card.get("team")), str(card.get("match_key") or card.get("job_id") or ""))
        if key in seen:
            continue
        cards.append(card)
        seen.add(key)
    matches = list(current.get("matches") or [])
    match_ids = {str(m.get("id") or m.get("job_id") or m.get("match_key")) for m in matches}
    for match in incoming.get("matches") or []:
        mid = str(match.get("id") or match.get("job_id") or match.get("match_key") or "")
        if mid and mid in match_ids:
            continue
        matches.append(match)
        if mid:
            match_ids.add(mid)
    notes = dict(current.get("notes") or {})
    notes.update(dict(incoming.get("notes") or {}))
    watch = sorted({str(t) for t in list(current.get("watchlist") or []) + list(incoming.get("watchlist") or [])})
    return save_scout_book(
        {"matches": matches, "cards": cards, "notes": notes, "watchlist": watch}
    )
