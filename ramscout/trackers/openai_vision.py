"""OpenAI Vision tracker with sparse keyframe sampling (never frame-by-frame)."""

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
# Thrifty defaults: sample ~every 2s (or every 30 pipeline frames), cap calls per match.
DEFAULT_INTERVAL_S = 2.0
DEFAULT_FRAME_STRIDE = 30
DEFAULT_MAX_CALLS = 24
DEFAULT_COOLDOWN_S = 30.0
MAX_COOLDOWN_S = 180.0


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class OpenAIVisionTracker:
    name = "openai"
    kind = "cloud"
    description = "OpenAI Vision (GPT) sparse keyframes — needs OPENAI_API_KEY"

    def __init__(
        self,
        interval_s: float | None = None,
        frame_stride: int | None = None,
        max_calls: int | None = None,
    ) -> None:
        self.interval_s = float(
            interval_s if interval_s is not None else _env_float("RAMSCOUT_OPENAI_INTERVAL_S", DEFAULT_INTERVAL_S)
        )
        self.frame_stride = max(
            1,
            int(
                frame_stride
                if frame_stride is not None
                else _env_int("RAMSCOUT_OPENAI_FRAME_STRIDE", DEFAULT_FRAME_STRIDE)
            ),
        )
        self.max_calls = max(
            1,
            int(max_calls if max_calls is not None else _env_int("RAMSCOUT_OPENAI_MAX_CALLS", DEFAULT_MAX_CALLS)),
        )
        self._last_t = -999.0
        self._last_frame = -10**9
        self._calls = 0
        self._cache: list[Detection] = []
        self._next_id = 5000
        self.warnings: list[str] = []
        self._resolved_model: str | None = None
        self._fail_until = 0.0
        self._cooldown_s = DEFAULT_COOLDOWN_S
        self._logged_fail = False

    def available(self, ctx: TrackerContext | None = None) -> bool:
        key = (ctx.openai_key if ctx else "") or os.environ.get("OPENAI_API_KEY", "")
        return bool(str(key).strip())

    def reset(self) -> None:
        self._last_t = -999.0
        self._last_frame = -10**9
        self._calls = 0
        self._cache = []
        self._next_id = 5000
        self.warnings = []
        self._resolved_model = None
        self._fail_until = 0.0
        self._cooldown_s = DEFAULT_COOLDOWN_S
        self._logged_fail = False

    def _should_sample(self, ctx: TrackerContext) -> bool:
        """True only on sparse keyframes — never every processed frame."""
        if self._calls >= self.max_calls:
            return False
        if time.time() < self._fail_until:
            return False
        # Hold last boxes between samples even when cache is empty (avoids hammering).
        if self._last_frame > -10**8:
            if (ctx.frame_index - self._last_frame) < self.frame_stride:
                return False
            if (ctx.t - self._last_t) < self.interval_s:
                return False
        return True

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]:
        key = (ctx.openai_key or os.environ.get("OPENAI_API_KEY", "")).strip()
        if not key:
            return list(self._cache)
        if not self._should_sample(ctx):
            return list(self._cache)

        # Reserve this keyframe before the HTTP call so failures/empty still pace calls.
        self._last_t = ctx.t
        self._last_frame = ctx.frame_index
        self._calls += 1

        try:
            preferred = (ctx.openai_model or DEFAULT_OPENAI_MODEL).strip() or DEFAULT_OPENAI_MODEL
            robots = self._query(cropped, key, preferred)
            self._logged_fail = False
            self._cooldown_s = DEFAULT_COOLDOWN_S
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
            if self._calls >= self.max_calls:
                msg = f"OpenAI vision hit max {self.max_calls} calls for this match; holding last boxes."
                if msg not in self.warnings:
                    self.warnings.append(msg)
            return list(self._cache)
        except _RateLimited as exc:
            self._arm_cooldown(exc, rate_limited=True)
            return list(self._cache)
        except Exception as exc:  # noqa: BLE001
            self._arm_cooldown(exc, rate_limited=False)
            return list(self._cache)

    def _arm_cooldown(self, exc: BaseException, *, rate_limited: bool) -> None:
        if rate_limited:
            self._cooldown_s = min(MAX_COOLDOWN_S, max(self._cooldown_s * 2.0, DEFAULT_COOLDOWN_S))
            label = "rate-limited (429)"
        else:
            self._cooldown_s = DEFAULT_COOLDOWN_S
            label = "failed"
        self._fail_until = time.time() + self._cooldown_s
        msg = f"OpenAI vision {label} ({exc}). Cooling down {self._cooldown_s:.0f}s."
        if msg not in self.warnings:
            self.warnings.append(msg)
        if not self._logged_fail:
            log.warning("%s Holding last boxes between samples.", msg)
            self._logged_fail = True

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
                # Never cascade across models on rate limits — that multiplies 429s.
                if exc.code == 429:
                    raise _RateLimited(f"HTTP 429 on model {name}") from exc
                # 400 often = model lacks vision / json_object support.
                if is_retryable_http(exc) or exc.code in {400, 404}:
                    log.debug("OpenAI model %s unavailable (%s); trying next.", name, exc.code)
                    continue
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "429" in str(exc):
                    raise _RateLimited(str(exc)) from exc
                if is_retryable_http(exc):
                    continue
                raise
        raise RuntimeError(f"All OpenAI vision models failed; last error: {last_error}")


class _RateLimited(RuntimeError):
    """OpenAI returned HTTP 429 — stop model cascade and back off."""


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
