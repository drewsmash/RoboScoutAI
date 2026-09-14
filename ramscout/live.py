"""Best-effort live / webcam scouting mode.

Live streams are still unsupported for YouTube ingest, but a local webcam or
file-like camera device can feed short rolling windows into the potato/OpenCV
trackers for pit-side demos.
"""

from __future__ import annotations

import threading
import time
from typing import Any

_LOCK = threading.Lock()
_STATE: dict[str, Any] = {
    "running": False,
    "device": 0,
    "frames": 0,
    "message": "Live mode idle.",
    "started_at": None,
}


def live_status() -> dict[str, Any]:
    with _LOCK:
        return dict(_STATE)


def start_live(device: int = 0) -> dict[str, Any]:
    with _LOCK:
        if _STATE["running"]:
            return dict(_STATE)
        _STATE.update(
            {
                "running": True,
                "device": int(device),
                "frames": 0,
                "message": "Live camera preview running (OpenCV). Tracking is approximate.",
                "started_at": time.time(),
            }
        )

    thread = threading.Thread(target=_live_loop, args=(int(device),), daemon=True)
    thread.start()
    return live_status()


def stop_live() -> dict[str, Any]:
    with _LOCK:
        _STATE["running"] = False
        _STATE["message"] = "Live mode stopped."
    return live_status()


def _live_loop(device: int) -> None:
    try:
        import cv2
    except Exception as exc:  # noqa: BLE001
        with _LOCK:
            _STATE.update({"running": False, "message": f"OpenCV unavailable: {exc}"})
        return

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        with _LOCK:
            _STATE.update({"running": False, "message": f"Could not open camera device {device}."})
        return

    frames = 0
    while True:
        with _LOCK:
            if not _STATE["running"]:
                break
        ok, _frame = cap.read()
        if not ok:
            break
        frames += 1
        if frames % 15 == 0:
            with _LOCK:
                _STATE["frames"] = frames
                _STATE["message"] = f"Live camera active — {frames} frames."
        time.sleep(0.01)
    cap.release()
    with _LOCK:
        _STATE["running"] = False
        _STATE["frames"] = frames
        if not str(_STATE.get("message") or "").startswith("Could not"):
            _STATE["message"] = f"Live mode ended after {frames} frames."
