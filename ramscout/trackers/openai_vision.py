"""OpenAI Vision tracker with model fallbacks for retired GPT ids."""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import Any

import numpy as np

from ramscout.trackers.cloud_models import is_retryable_http, openai_model_candidates
from ramscout.trackers.types import Detection, TrackerContext
from ramscout.trackers.utils import encode_jpeg_bgr, plausible_robot_size

log = logging.getLogger(__name__)

_PROMPT = """You are helping scout an FRC robotics match from a broadcast camera crop.
Return ONLY JSON (no markdown) as:
{"robots":[{"x1":0-1,"y1":0-1,"x2":0-1,"y2":0-1,"alliance":"red|blue|unknown","team":""}]}
Normalize coordinates to the image width/height (0-1). Include up to 6 robots on the field.
Ignore the scoreboard/HUD. If unsure, still guess approximate boxes for visible robots.
"""

DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


class OpenAIVisionTracker:
    name = "openai"
    kind = "cloud"
    description = "OpenAI Vision (GPT) keyframe boxes — needs OPENAI_API_KEY"

    def __init__(self, interval_s: float = 2.5, max_calls: int = 48) -> None:
        self.interval_s = interval_s
        self.max_calls = max_calls
        self._last_t = -999.0
        self._calls = 0
        self._cache: list[Detection] = []
        self._next_id = 5000
        self.warnings: list[str] = []
        self._resolved_model: str | None = None
        self._fail_until = 0.0
        self._logged_fail = False

    def available(self, ctx: TrackerContext | None = None) -> bool:
        key = (ctx.openai_key if ctx else "") or os.environ.get("OPENAI_API_KEY", "")
        return bool(str(key).strip())

    def reset(self) -> None:
        self._last_t = -999.0
        self._calls = 0
        self._cache = []
        self._next_id = 5000
        self.warnings = []
        self._resolved_model = None
        self._fail_until = 0.0
        self._logged_fail = False

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        key = (ctx.openai_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            return list(self._cache)
        if self._calls >= self.max_calls:
            return list(self._cache)
        if time.time() < self._fail_until:
            return list(self._cache)
        if ctx.t - self._last_t < self.interval_s and self._cache:
            return list(self._cache)

        try:
            preferred = (ctx.openai_model or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
            robots = self._query(cropped, key, preferred)
            self._calls += 1
            self._last_t = ctx.t
            self._logged_fail = False
            dets: list[Detection] = []
            h, w = cropped.shape[:2]
            for robot in robots:
                x1 = float(robot.get("x1", 0)) * w
                y1 = float(robot.get("y1", 0)) * h
                x2 = float(robot.get("x2", 0)) * w
                y2 = float(robot.get("y2", 0)) * h
                if x2 <= x1 or y2 <= y1:
                    continue
                if not plausible_robot_size(x2 - x1, y2 - y1, ctx.crop_w, ctx.crop_h):
                    continue
                alliance = str(robot.get("alliance") or "unknown").lower()
                if alliance not in {"red", "blue"}:
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
            self._fail_until = time.time() + 20.0
            msg = f"OpenAI vision failed ({exc})."
            if msg not in self.warnings:
                self.warnings.append(msg)
            if not self._logged_fail:
                log.warning("%s Cooling down 20s before retrying.", msg)
                self._logged_fail = True
            return list(self._cache)

    def _query(self, cropped: np.ndarray, api_key: str, model: str) -> list[dict[str, Any]]:
        import urllib.error
        import urllib.request

        jpeg = encode_jpeg_bgr(cropped, quality=65)
        b64 = base64.b64encode(jpeg).decode("ascii")
        candidates: list[str] = []
        if self._resolved_model:
            candidates.append(self._resolved_model)
        for name in openai_model_candidates(preferred=model):
            if name not in candidates:
                candidates.append(name)

        last_error: Exception | None = None
        for name in candidates:
            body = {
                "model": name,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                            },
                        ],
                    }
                ],
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=45) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if self._resolved_model != name:
                    log.info("OpenAI vision using model %s", name)
                self._resolved_model = name
                return _parse_robots_json(content)
            except urllib.error.HTTPError as exc:
                last_error = exc
                # 400 often = model lacks vision / json_object support.
                if is_retryable_http(exc) or exc.code in {400, 404}:
                    log.debug("OpenAI model %s unavailable (%s); trying next.", name, exc.code)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if is_retryable_http(exc):
                    continue
                raise
        raise RuntimeError(f"All OpenAI vision models failed; last error: {last_error}")


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
