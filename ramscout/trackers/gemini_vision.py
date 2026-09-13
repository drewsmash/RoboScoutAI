"""Google Gemini vision tracker — samples keyframes for robot boxes."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Any

import numpy as np

from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import encode_jpeg_bgr, plausible_robot_size

log = logging.getLogger(__name__)

_PROMPT = """You are helping scout an FRC robotics match from a broadcast camera crop.
Return ONLY JSON (no markdown) as:
{"robots":[{"x1":0-1,"y1":0-1,"x2":0-1,"y2":0-1,"alliance":"red|blue|unknown","team":""}]}
Normalize coordinates to the image width/height (0-1). Include up to 6 robots on the field.
Ignore the scoreboard/HUD. If unsure, still guess approximate boxes for visible robots.
"""

# Current Google AI keys reject older flash IDs (404). Prefer latest stable flash,
# then fall back through aliases that still accept generateContent.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
GEMINI_MODEL_FALLBACKS = (
    "gemini-3.5-flash",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
    "gemini-2.5-flash",
)


class GeminiVisionTracker:
    name = "gemini"
    kind = "cloud"
    description = "Google Gemini Vision keyframe boxes — needs GOOGLE_API_KEY / GEMINI_API_KEY"

    def __init__(self, interval_s: float = 2.5, max_calls: int = 48) -> None:
        self.interval_s = interval_s
        self.max_calls = max_calls
        self._last_t = -999.0
        self._calls = 0
        self._cache: list[Detection] = []
        self._next_id = 6000
        self.warnings: list[str] = []
        self._resolved_model: str | None = None

    def available(self, ctx: TrackerContext | None = None) -> bool:
        return bool(_google_key(ctx).strip())

    def reset(self) -> None:
        self._last_t = -999.0
        self._calls = 0
        self._cache = []
        self._next_id = 6000
        self.warnings = []
        self._resolved_model = None

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        key = _google_key(ctx)
        if not key:
            return list(self._cache)
        if self._calls >= self.max_calls:
            return list(self._cache)
        if ctx.t - self._last_t < self.interval_s and self._cache:
            return list(self._cache)

        try:
            preferred = (ctx.google_model or DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
            robots = self._query(cropped, key, preferred)
            self._calls += 1
            self._last_t = ctx.t
            dets: list[Detection] = []
            h, w = cropped.shape[:2]
            for robot in robots:
                x1, y1, x2, y2 = _robot_box(robot, w, h)
                if x2 <= x1 or y2 <= y1:
                    continue
                if not plausible_robot_size(x2 - x1, y2 - y1, ctx.crop_w, ctx.crop_h):
                    continue
                alliance = str(robot.get("alliance") or robot.get("label") or "unknown").lower()
                if "red" in alliance:
                    alliance = "red"
                elif "blue" in alliance:
                    alliance = "blue"
                elif alliance not in {"red", "blue"}:
                    alliance = "unknown"
                team = str(robot.get("team") or "")
                if ctx.team_numbers and team and team not in ctx.team_numbers:
                    team = ""
                dets.append(
                    Detection(
                        track_id=self._next_id,
                        bbox=[x1, y1, x2, y2],
                        source=self.name,
                        confidence=0.55,
                        alliance=alliance,
                        team=team,
                    )
                )
                self._next_id += 1
            if dets:
                self._cache = dets
            return list(self._cache)
        except Exception as exc:  # noqa: BLE001
            msg = f"Gemini vision failed ({exc})."
            if msg not in self.warnings:
                self.warnings.append(msg)
            log.warning(msg)
            return list(self._cache)

    def _query(self, cropped: np.ndarray, api_key: str, model: str) -> list[dict[str, Any]]:
        import urllib.error
        import urllib.request

        jpeg = encode_jpeg_bgr(cropped, quality=65)
        b64 = base64.b64encode(jpeg).decode("ascii")
        body = {
            "contents": [
                {
                    "parts": [
                        {"text": _PROMPT},
                        {"inline_data": {"mime_type": "image/jpeg", "data": b64}},
                    ]
                }
            ],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        payload_bytes = json.dumps(body).encode("utf-8")

        candidates: list[str] = []
        if self._resolved_model:
            candidates.append(self._resolved_model)
        for name in (model, *GEMINI_MODEL_FALLBACKS):
            if name and name not in candidates:
                candidates.append(name)

        last_error: Exception | None = None
        for name in candidates:
            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{name}:generateContent?key={api_key}"
            )
            req = urllib.request.Request(
                url,
                data=payload_bytes,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                text = payload["candidates"][0]["content"]["parts"][0]["text"]
                self._resolved_model = name
                return _parse_robots_json(text)
            except urllib.error.HTTPError as exc:
                last_error = exc
                # Retry on missing/retired model ids or transient overload.
                if exc.code in {404, 429, 503}:
                    log.info("Gemini model %s unavailable (%s); trying fallback.", name, exc.code)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue
        raise RuntimeError(f"All Gemini models failed; last error: {last_error}")


def _google_key(ctx: TrackerContext | None) -> str:
    if ctx and ctx.google_key.strip():
        return ctx.google_key.strip()
    return (
        os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_AI_API_KEY")
        or ""
    ).strip()


def _robot_box(robot: dict[str, Any], width: float, height: float) -> tuple[float, float, float, float]:
    """Accept either normalized x1/y1/x2/y2 or Gemini box_2d [y0,x0,y1,x1] in 0–1000."""
    if "box_2d" in robot and isinstance(robot["box_2d"], (list, tuple)) and len(robot["box_2d"]) >= 4:
        y0, x0, y1, x1 = (float(v) for v in robot["box_2d"][:4])
        # box_2d is typically 0–1000; also tolerate 0–1.
        scale = 1000.0 if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5 else 1.0
        return (
            (x0 / scale) * width,
            (y0 / scale) * height,
            (x1 / scale) * width,
            (y1 / scale) * height,
        )
    return (
        float(robot.get("x1", 0)) * width,
        float(robot.get("y1", 0)) * height,
        float(robot.get("x2", 0)) * width,
        float(robot.get("y2", 0)) * height,
    )


def _parse_robots_json(content: str) -> list[dict[str, Any]]:
    text = (content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    data = json.loads(text)
    if isinstance(data, dict):
        robots = data.get("robots") or data.get("detections") or []
    elif isinstance(data, list):
        robots = data
    else:
        robots = []
    return [r for r in robots if isinstance(r, dict)]
