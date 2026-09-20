"""Vercel AI Gateway evaluation via TypeSafe Jev (`typesafe-ai/jev`).

Jev answers typed questions (boolean / choice / score) about shared state —
used here to verify heuristic scout events and optionally classify camera layout.
Docs: https://vercel.com/docs/ai-gateway/modalities/evaluation
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

JEV_MODEL = "typesafe-ai/jev"
EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_TIMEOUT_S = 45.0
MAX_EVENT_QUESTIONS = 20
TRUE_THRESHOLD = 0.55
FALSE_THRESHOLD = 0.35

_TRUTHY = {"1", "true", "yes", "on"}


def gateway_api_key(explicit: str | None = None) -> str:
    """Resolve AI Gateway key from job field or environment."""
    for value in (
        explicit,
        os.environ.get("AI_GATEWAY_API_KEY"),
        os.environ.get("VERCEL_AI_GATEWAY_API_KEY"),
        os.environ.get("ROBOSCOUT_AI_GATEWAY_KEY"),
        os.environ.get("RAMSCOUT_AI_GATEWAY_KEY"),
    ):
        text = (value or "").strip()
        if text:
            return text
    return ""


def is_available(explicit: str | None = None) -> bool:
    return bool(gateway_api_key(explicit))


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def _gateway_provider_options(
    *,
    zero_data_retention: bool | None = None,
    only: list[str] | None = None,
) -> dict[str, Any] | None:
    """Build optional providerOptions.gateway — omitted by default (safer).

    Strict ``only`` / ZDR filters commonly cause HTTP 403 when the key/plan
    cannot route to that provider set. Opt in via env or explicit args:

    - ``AI_GATEWAY_ZERO_DATA_RETENTION=1`` / ``JEV_ZERO_DATA_RETENTION=1``
    - ``AI_GATEWAY_ONLY=typesafe-ai`` / ``JEV_GATEWAY_ONLY=typesafe-ai``
      (comma-separated provider ids)
    """
    opts: dict[str, Any] = {}

    zdr = zero_data_retention
    if zdr is None:
        zdr = _env_truthy("AI_GATEWAY_ZERO_DATA_RETENTION") or _env_truthy(
            "JEV_ZERO_DATA_RETENTION"
        )
    if zdr:
        opts["zeroDataRetention"] = True

    providers = only
    if providers is None:
        raw = (
            os.environ.get("AI_GATEWAY_ONLY")
            or os.environ.get("JEV_GATEWAY_ONLY")
            or ""
        ).strip()
        if raw:
            providers = [p.strip() for p in raw.split(",") if p.strip()]
    if providers:
        opts["only"] = list(providers)

    return {"gateway": opts} if opts else None


def _response_detail(res: httpx.Response, *, limit: int = 240) -> str:
    """Best-effort short body snippet for error messages (no traceback noise)."""
    try:
        data = res.json()
        if isinstance(data, dict):
            for key in ("error", "message", "detail"):
                val = data.get(key)
                if isinstance(val, dict):
                    msg = val.get("message") or val.get("code") or str(val)
                else:
                    msg = val
                if msg:
                    return str(msg).strip()[:limit]
            return str(data)[:limit]
    except Exception:
        pass
    text = (res.text or "").strip()
    return text[:limit] if text else ""


def _denied_message(status_code: int, res: httpx.Response) -> str:
    detail = _response_detail(res)
    base = (
        f"AI Gateway HTTP {status_code} on Jev /v1/evaluate. "
        "Check AI_GATEWAY_API_KEY (Bearer) has evaluation access for typesafe-ai/jev. "
        "If you set AI_GATEWAY_ONLY / JEV_GATEWAY_ONLY or "
        "AI_GATEWAY_ZERO_DATA_RETENTION, clear them — strict only/ZDR often causes 403."
    )
    if detail:
        return f"{base} Gateway said: {detail}"
    return base


def evaluate(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,
    api_key: str | None = None,
    model: str = JEV_MODEL,
    timeout: float = DEFAULT_TIMEOUT_S,
    zero_data_retention: bool | None = None,
    only: list[str] | None = None,
) -> dict[str, Any]:
    """POST /v1/evaluate on AI Gateway. Returns the full JSON body.

    Request shape matches Vercel docs: ``model``, ``state``, ``questions``
    (boolean / choice / score). ``providerOptions.gateway`` is only sent when
    ZDR or provider ``only`` is explicitly enabled.
    """
    key = gateway_api_key(api_key)
    if not key:
        raise RuntimeError(
            "AI Gateway key missing. Set AI_GATEWAY_API_KEY or paste it in Tracking settings."
        )
    if not questions:
        return {"model": model, "answers": {}}

    payload: dict[str, Any] = {
        "model": model,
        "state": state,
        "questions": questions,
    }
    provider_options = _gateway_provider_options(
        zero_data_retention=zero_data_retention,
        only=only,
    )
    if provider_options:
        payload["providerOptions"] = provider_options

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "RoboScoutAI/jev",
    }
    # Avoid httpx's default INFO "HTTP Request: ... 403" drowning the actionable warning.
    httpx_log = logging.getLogger("httpx")
    prev_level = httpx_log.level
    try:
        httpx_log.setLevel(logging.WARNING)
        with httpx.Client(timeout=timeout) as client:
            res = client.post(EVALUATE_URL, headers=headers, json=payload)
            if res.status_code in {401, 403}:
                raise RuntimeError(_denied_message(res.status_code, res))
            if res.status_code >= 400:
                detail = _response_detail(res)
                msg = f"AI Gateway HTTP {res.status_code} on Jev /v1/evaluate."
                if detail:
                    msg = f"{msg} {detail}"
                raise RuntimeError(msg)
            data = res.json()
    finally:
        httpx_log.setLevel(prev_level)

    if not isinstance(data, dict):
        raise RuntimeError("Unexpected Jev response shape.")
    return data


def boolean_probability(answer: Any) -> float | None:
    """Normalize boolean / noul answers to a [0, 1] probability of true."""
    if not isinstance(answer, dict):
        return None
    if "probability" in answer:
        try:
            return float(answer["probability"])
        except (TypeError, ValueError):
            return None
    if "noul" in answer:
        try:
            return float(answer["noul"])
        except (TypeError, ValueError):
            return None
    return None


def choice_value(answer: Any) -> str | None:
    if not isinstance(answer, dict):
        return None
    choice = answer.get("choice")
    return str(choice) if choice is not None else None


def score_value(answer: Any) -> float | None:
    if not isinstance(answer, dict):
        return None
    if "score" not in answer:
        return None
    try:
        return float(answer["score"])
    except (TypeError, ValueError):
        return None


def verify_scout_events(
    events: list[dict[str, Any]],
    *,
    match: dict[str, Any] | None = None,
    side_cues: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ask Jev whether each heuristic scout event looks valid; adjust confidence.

    Returns (updated_events, warnings). No-ops when the gateway key is missing
    or there are no events. Failures are soft — heuristics are kept.
    """
    key = gateway_api_key(api_key)
    if not key or not events:
        return events, []

    # Prefer lower-confidence / scoring-related events — those need the most help.
    ranked = sorted(
        enumerate(events),
        key=lambda item: (
            0 if str(item[1].get("type") or "") in {"hub_score_candidate", "climb_attempt", "defense"} else 1,
            float(item[1].get("confidence") or 0.5),
        ),
    )
    selected = ranked[:MAX_EVENT_QUESTIONS]
    if not selected:
        return events, []

    questions: dict[str, dict[str, Any]] = {}
    for idx, event in selected:
        etype = str(event.get("type") or "event")
        team = str(event.get("team") or "?")
        t = float(event.get("t") or 0.0)
        zone = str(event.get("zone") or "")
        detail = str(event.get("detail") or "")
        questions[f"e{idx}"] = {
            "type": "boolean",
            "instructions": (
                f"Is this FRC auto-scout {etype} for team {team} at t={t:.1f}s "
                f"(zone={zone or 'unknown'}) a plausible real action, not a false positive?"
            ),
            "criteria": {
                "true": "Trajectory + timing + side-camera cues support the labeled action.",
                "false": "Likely noise, wrong zone, or unrelated dwell — discard or down-weight.",
            },
        }
        if detail:
            questions[f"e{idx}"]["instructions"] += f" Heuristic detail: {detail[:180]}"

    alliances = (match or {}).get("alliances") or {}
    state = {
        "sport": "FRC",
        "match": {
            "key": (match or {}).get("key"),
            "comp_level": (match or {}).get("comp_level"),
            "blue_score": (alliances.get("blue") or {}).get("score"),
            "red_score": (alliances.get("red") or {}).get("score"),
            "blue_teams": (alliances.get("blue") or {}).get("team_keys"),
            "red_teams": (alliances.get("red") or {}).get("team_keys"),
        },
        "side_cues": (side_cues or [])[:24],
        "events": [
            {
                "id": f"e{idx}",
                "type": ev.get("type"),
                "team": ev.get("team"),
                "t": ev.get("t"),
                "zone": ev.get("zone"),
                "confidence": ev.get("confidence"),
                "detail": (ev.get("detail") or "")[:240],
                "period": ev.get("period"),
            }
            for idx, ev in selected
        ],
    }

    try:
        result = evaluate(state, questions, api_key=key)
    except Exception as exc:  # noqa: BLE001
        log.warning("Jev event verification skipped: %s", exc)
        return events, [f"Jev event verification skipped: {exc}"]

    answers = result.get("answers") or {}
    out = [dict(ev) for ev in events]
    kept = 0
    dropped = 0
    for idx, _event in selected:
        prob = boolean_probability(answers.get(f"e{idx}"))
        if prob is None:
            continue
        row = out[idx]
        row["jev_probability"] = round(prob, 3)
        row["jev_model"] = JEV_MODEL
        base = float(row.get("confidence") or 0.5)
        if prob >= TRUE_THRESHOLD:
            row["confidence"] = round(min(0.98, max(base, 0.45) + 0.25 * (prob - TRUE_THRESHOLD) / max(1e-6, 1 - TRUE_THRESHOLD)), 3)
            detail = str(row.get("detail") or "")
            if "Jev" not in detail:
                row["detail"] = (detail + " · Jev confirmed").strip(" ·")
            kept += 1
        elif prob <= FALSE_THRESHOLD:
            row["confidence"] = round(min(base, 0.22), 3)
            row["jev_rejected"] = True
            detail = str(row.get("detail") or "")
            if "Jev" not in detail:
                row["detail"] = (detail + " · Jev likely false positive").strip(" ·")
            dropped += 1
        else:
            row["confidence"] = round(0.5 * base + 0.5 * prob, 3)

    # Soft-drop strongly rejected low-value noise, keep everything else for the editor.
    filtered = [ev for ev in out if not (ev.get("jev_rejected") and float(ev.get("confidence") or 0) < 0.2)]
    warning = (
        f"Jev ({JEV_MODEL}) reviewed {len(selected)} scout events "
        f"({kept} boosted, {dropped} down-weighted)."
    )
    return filtered, [warning]


def classify_camera_layout(
    summary: dict[str, Any],
    *,
    api_key: str | None = None,
) -> tuple[str | None, float, list[str]]:
    """Ask Jev to pick single / stacked_top / stacked_sides from a layout summary."""
    key = gateway_api_key(api_key)
    if not key:
        return None, 0.0, []
    questions = {
        "layout": {
            "type": "choice",
            "instructions": "Which broadcast camera layout best matches this FRC match VOD summary?",
            "criteria": {
                "single": "One wide field camera fills the frame.",
                "stacked_top": "Overview field camera on top, secondary feed below (vertical stack).",
                "stacked_sides": "Overview on top with blue/red side cameras below or beside.",
            },
        }
    }
    try:
        result = evaluate(summary, questions, api_key=key)
    except Exception as exc:  # noqa: BLE001
        log.warning("Jev camera layout skipped: %s", exc)
        return None, 0.0, [f"Jev camera layout skipped: {exc}"]
    answer = (result.get("answers") or {}).get("layout") or {}
    choice = choice_value(answer)
    probs = answer.get("probabilities") if isinstance(answer, dict) else None
    conf = 0.0
    if isinstance(probs, dict) and choice in probs:
        try:
            conf = float(probs[choice])
        except (TypeError, ValueError):
            conf = 0.0
    if choice not in {"single", "stacked_top", "stacked_sides"}:
        return None, 0.0, ["Jev camera layout returned an unknown choice."]
    return choice, conf, []
