"""Remember the last local tracker that won auto selection on this machine."""

from __future__ import annotations

import json
from typing import Any

from ramscout.paths import data_dir

_NAME = "last_local_tracker.json"
_CLOUD = {"openai", "gemini", "cloud"}


def _path():
    return data_dir() / _NAME


def load_last_tracker() -> str | None:
    path = _path()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    mode = str((payload or {}).get("mode") or "").strip().lower()
    if not mode or mode in _CLOUD or mode == "auto":
        return None
    return mode


def save_last_tracker(mode: str, *, scores: dict[str, Any] | None = None) -> None:
    cleaned = str(mode or "").strip().lower()
    if not cleaned or cleaned in _CLOUD or cleaned == "auto":
        return
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"mode": cleaned}
    if scores:
        body["total"] = (scores.get(cleaned) or {}).get("total")
    path.write_text(json.dumps(body), encoding="utf-8")
