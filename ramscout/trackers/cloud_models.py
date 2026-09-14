"""Runtime Gemini / OpenAI vision model discovery and fallbacks."""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request

log = logging.getLogger(__name__)

GEMINI_STATIC_FALLBACKS: tuple[str, ...] = (
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3-flash-preview",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-2.5-flash-lite",
)

OPENAI_VISION_FALLBACKS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "gpt-4.1-mini",
    "gpt-4.1",
)

_gemini_cache: dict[str, tuple[float, list[str]]] = {}
_CACHE_TTL_S = 30 * 60


def discover_gemini_models(api_key: str) -> list[str]:
    key = (api_key or "").strip()
    if not key:
        return list(GEMINI_STATIC_FALLBACKS)
    cached = _gemini_cache.get(key)
    now = time.time()
    if cached and cached[0] > now:
        return list(cached[1])

    discovered: list[str] = []
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "RamScoutAI/0.4"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        for row in payload.get("models") or []:
            methods = row.get("supportedGenerationMethods") or []
            if "generateContent" not in methods:
                continue
            name = str(row.get("name") or "").replace("models/", "").strip()
            if not name:
                continue
            lower = name.lower()
            if any(
                bad in lower
                for bad in (
                    "embed",
                    "tts",
                    "image",
                    "robotics",
                    "computer-use",
                    "lyria",
                    "deep-research",
                    "transcribe",
                )
            ):
                continue
            if "flash" in lower or name.startswith("gemini-3"):
                discovered.append(name)
    except Exception as exc:  # noqa: BLE001
        log.info("Gemini ListModels failed (%s); using static fallbacks.", exc)

    ordered = _prefer_order(discovered, GEMINI_STATIC_FALLBACKS)
    _gemini_cache[key] = (now + _CACHE_TTL_S, ordered)
    return ordered


def gemini_model_candidates(api_key: str, preferred: str | None = None) -> list[str]:
    live = discover_gemini_models(api_key)
    out: list[str] = []
    for name in ((preferred or "").strip(), *live, *GEMINI_STATIC_FALLBACKS):
        if name and name not in out:
            out.append(name)
    return out


def openai_model_candidates(preferred: str | None = None) -> list[str]:
    out: list[str] = []
    for name in ((preferred or "").strip(), *OPENAI_VISION_FALLBACKS):
        if name and name not in out:
            out.append(name)
    return out


def is_retryable_http(exc: BaseException) -> bool:
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in {404, 408, 429, 500, 502, 503, 504}
    text = str(exc).lower()
    return "timed out" in text or "temporarily" in text or "unavailable" in text


def _prefer_order(discovered: list[str], static: tuple[str, ...]) -> list[str]:
    out: list[str] = []
    for name in static:
        if name in discovered and name not in out:
            out.append(name)
    for name in discovered:
        if name not in out:
            out.append(name)
    return out or list(static)
