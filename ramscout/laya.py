"""Local Laya decision engine (Jev replacement) + optional gateway fallback.

Laya (``pip install laya``) is an open-weight System-1 model: state + typed
questions → calibrated answers in one forward pass (~10× faster than cloud Jev).

Priority:
1. Local ``laya`` package (``LAYA_MODEL`` / ``convaiinnovations/laya``)
2. Optional Vercel AI Gateway ``typesafe-ai/jev`` when a key is present and
   local weights are unavailable

Scout verification and layout classification both go through this module.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

log = logging.getLogger(__name__)

LAYA_DEFAULT_REPO = "convaiinnovations/laya"
LAYA_TYPED_REPO = "convaiinnovations/laya"
LAYA_TYPED_SUBFOLDER = "typed-decisions"
JEV_MODEL = "typesafe-ai/jev"
EVALUATE_URL = "https://ai-gateway.vercel.sh/v1/evaluate"
DEFAULT_TIMEOUT_S = 45.0
MAX_EVENT_QUESTIONS = 24
TRUE_THRESHOLD = 0.55
FALSE_THRESHOLD = 0.35

_TRUTHY = {"1", "true", "yes", "on"}

_agent_lock = threading.Lock()
_agent: Any | None = None
_agent_error: str = ""
_agent_label: str = ""


def gateway_api_key(explicit: str | None = None) -> str:
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


def _env_truthy(name: str) -> bool:
    return (os.environ.get(name) or "").strip().lower() in _TRUTHY


def local_enabled() -> bool:
    """Local Laya is on by default; set ``LAYA_LOCAL=0`` to force gateway-only."""
    raw = (os.environ.get("LAYA_LOCAL") or "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def local_available() -> bool:
    if not local_enabled():
        return False
    try:
        import laya  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def is_available(explicit: str | None = None) -> bool:
    """True when local Laya can load or a gateway key is present."""
    return local_available() or bool(gateway_api_key(explicit))


def backend_status(explicit: str | None = None) -> dict[str, Any]:
    return {
        "local": local_available(),
        "local_enabled": local_enabled(),
        "gateway": bool(gateway_api_key(explicit)),
        "preferred": "laya-local" if local_available() else ("jev-gateway" if gateway_api_key(explicit) else "none"),
        "model": _agent_label or (os.environ.get("LAYA_MODEL") or LAYA_DEFAULT_REPO),
        "error": _agent_error,
    }


def _load_local_agent() -> Any:
    global _agent, _agent_error, _agent_label
    with _agent_lock:
        if _agent is not None:
            return _agent
        if _agent_error and not _env_truthy("LAYA_RETRY"):
            raise RuntimeError(_agent_error)
        try:
            import laya

            os.environ.setdefault("USE_TF", "0")
            repo = (os.environ.get("LAYA_MODEL") or LAYA_DEFAULT_REPO).strip()
            sub = (os.environ.get("LAYA_SUBFOLDER") or "").strip()
            # Prefer typed-decisions when available — closest to Jev workflows.
            if not sub and repo == LAYA_DEFAULT_REPO:
                try:
                    _agent = laya.load(LAYA_TYPED_REPO, subfolder=LAYA_TYPED_SUBFOLDER)
                    _agent_label = f"{LAYA_TYPED_REPO}/{LAYA_TYPED_SUBFOLDER}"
                except Exception:  # noqa: BLE001
                    _agent = laya.load(repo)
                    _agent_label = repo
            elif sub:
                _agent = laya.load(repo, subfolder=sub)
                _agent_label = f"{repo}/{sub}"
            else:
                _agent = laya.load(repo)
                _agent_label = repo
            _agent_error = ""
            log.info("Laya loaded: %s", _agent_label)
            return _agent
        except Exception as exc:  # noqa: BLE001
            _agent_error = str(exc)
            log.warning("Laya local load failed: %s", exc)
            raise


def _normalize_answers(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    answers = raw.get("answers") if "answers" in raw else raw
    if not isinstance(answers, dict):
        return {}
    return answers


def evaluate(
    state: Any,
    questions: dict[str, Any],
    *,
    api_key: str | None = None,
    model: str | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict[str, Any]:
    """Run typed questions against state. Prefers local Laya, then gateway Jev."""
    if not questions:
        return {"answers": {}, "backend": "none"}

    if local_available():
        try:
            agent = _load_local_agent()
            result = agent.predict(state, questions)
            answers = _normalize_answers(result)
            return {
                "model": _agent_label or LAYA_DEFAULT_REPO,
                "answers": answers,
                "backend": "laya-local",
                "routing": result.get("routing") if isinstance(result, dict) else None,
            }
        except Exception as exc:  # noqa: BLE001
            log.warning("Laya local predict failed (%s); trying gateway", exc)

    key = gateway_api_key(api_key)
    if not key:
        raise RuntimeError(
            "No decision backend available. Install Laya (`pip install laya`) for on-machine "
            "inference, or set AI_GATEWAY_API_KEY for cloud Jev."
        )
    return _evaluate_gateway(state, questions, api_key=key, model=model or JEV_MODEL, timeout_s=timeout_s)


def _evaluate_gateway(
    state: Any,
    questions: dict[str, Any],
    *,
    api_key: str,
    model: str,
    timeout_s: float,
) -> dict[str, Any]:
    import httpx

    payload: dict[str, Any] = {"model": model, "state": state, "questions": questions}
    opts: dict[str, Any] = {}
    if _env_truthy("AI_GATEWAY_ZERO_DATA_RETENTION") or _env_truthy("JEV_ZERO_DATA_RETENTION"):
        opts["zeroDataRetention"] = True
    only = (os.environ.get("AI_GATEWAY_ONLY") or os.environ.get("JEV_GATEWAY_ONLY") or "").strip()
    if only:
        opts["only"] = [p.strip() for p in only.split(",") if p.strip()]
    if opts:
        payload["providerOptions"] = {"gateway": opts}

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        with httpx.Client(timeout=timeout_s) as client:
            resp = client.post(EVALUATE_URL, json=payload, headers=headers)
            if resp.status_code == 403:
                detail = ""
                try:
                    body = resp.json()
                    detail = str(
                        (body.get("error") or {}).get("message")
                        if isinstance(body.get("error"), dict)
                        else body.get("error") or body.get("message") or resp.text
                    )
                except Exception:  # noqa: BLE001
                    detail = resp.text[:200]
                raise RuntimeError(
                    "AI Gateway returned 403 Forbidden for typesafe-ai/jev. "
                    "Check AI_GATEWAY_API_KEY, and avoid AI_GATEWAY_ONLY / "
                    "AI_GATEWAY_ZERO_DATA_RETENTION unless your plan supports them. "
                    f"Detail: {detail or 'Forbidden'}"
                )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise RuntimeError(f"AI Gateway HTTP {exc.response.status_code}: {exc}") from exc
    answers = _normalize_answers(data)
    return {"model": model, "answers": answers, "backend": "jev-gateway", "raw": data}


def boolean_probability(answer: dict[str, Any] | None) -> float | None:
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


def choice_value(answer: dict[str, Any] | None) -> str | None:
    if not isinstance(answer, dict):
        return None
    choice = answer.get("choice")
    return str(choice) if choice is not None else None


def verify_scout_events(
    events: list[dict[str, Any]],
    *,
    match: dict[str, Any] | None = None,
    cards: list[dict[str, Any]] | None = None,
    api_key: str | None = None,
    max_questions: int = MAX_EVENT_QUESTIONS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Verify heuristic scout events with Laya (or gateway Jev)."""
    if not is_available(api_key):
        return events, []

    candidates = [
        (i, e)
        for i, e in enumerate(events)
        if str(e.get("type") or "")
        in {
            "hub_score_candidate",
            "hub_score",
            "climb_attempt",
            "climb",
            "defense",
            "collection",
            "intake",
        }
    ]
    if not candidates:
        return events, []

    questions: dict[str, Any] = {}
    index_map: dict[str, int] = {}
    for n, (idx, event) in enumerate(candidates[: max(1, max_questions)]):
        key = f"e{n}"
        index_map[key] = idx
        etype = str(event.get("type") or "event")
        team = str(event.get("team") or "?")
        zone = str(event.get("zone") or "")
        detail = str(event.get("detail") or "")
        questions[key] = {
            "type": "noul",
            "instructions": (
                f"Given the match scouting state, is this {etype} for team {team} "
                f"in zone {zone} a real on-field action (not a false positive from walls/crowd)?"
            ),
        }
        # Keep a compact state slice per question via shared state document.
        _ = detail  # included in state below

    state = {
        "task": "FRC match auto-scout verification",
        "match": {
            "key": (match or {}).get("key"),
            "comp_level": (match or {}).get("comp_level"),
            "match_number": (match or {}).get("match_number"),
            "alliances": (match or {}).get("alliances"),
        },
        "teams": [
            {"team": c.get("team"), "alliance": c.get("alliance"), "nickname": c.get("nickname")}
            for c in (cards or [])[:12]
        ],
        "events": [
            {
                "id": key,
                "type": events[idx].get("type"),
                "team": events[idx].get("team"),
                "t": events[idx].get("t"),
                "zone": events[idx].get("zone"),
                "confidence": events[idx].get("confidence"),
                "detail": events[idx].get("detail"),
                "source": events[idx].get("source"),
            }
            for key, idx in index_map.items()
        ],
    }

    try:
        result = evaluate(state, questions, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return events, [f"Laya/Jev verify skipped: {exc}"]

    out = [dict(e) for e in events]
    answers = result.get("answers") or {}
    backend = str(result.get("backend") or "laya")
    model = str(result.get("model") or "")
    confirmed = rejected = 0
    for key, idx in index_map.items():
        prob = boolean_probability(answers.get(key) if isinstance(answers, dict) else None)
        if prob is None:
            continue
        event = out[idx]
        event["laya_probability"] = round(prob, 3)
        event["laya_backend"] = backend
        # Keep legacy key so older UI chips still light up.
        event["jev_probability"] = event["laya_probability"]
        conf = float(event.get("confidence") or 0.5)
        if prob >= TRUE_THRESHOLD:
            event["confidence"] = float(min(0.98, conf + 0.2 * prob))
            event["laya_confirmed"] = True
            event["jev_confirmed"] = True
            detail = str(event.get("detail") or "")
            tag = f"{backend} confirm"
            if tag not in detail:
                event["detail"] = f"{detail}; {tag}".strip("; ")
            confirmed += 1
        elif prob <= FALSE_THRESHOLD:
            event["confidence"] = float(min(conf, 0.22))
            event["laya_rejected"] = True
            event["jev_rejected"] = True
            rejected += 1
        else:
            event["confidence"] = float(0.5 * conf + 0.5 * prob)

    note = (
        f"Laya ({backend}{('/' + model) if model else ''}) verified {len(index_map)} scout event(s) "
        f"— {confirmed} confirmed, {rejected} soft-rejected."
    )
    return out, [note]


def classify_camera_layout(
    summary: dict[str, Any],
    *,
    api_key: str | None = None,
) -> tuple[str | None, float, list[str]]:
    """Classify broadcast layout from a text/feature summary."""
    if not is_available(api_key):
        return None, 0.0, []
    questions = {
        "layout": {
            "type": "choice",
            "instructions": "Which broadcast camera layout best matches this FRC VOD?",
            "criteria": {
                "single": "one full-frame overview camera",
                "stacked_top": "wide overview on top, one lower sidecar",
                "stacked_sides": "overview on top, blue and red angle shots split below",
                "side_by_side": "two cameras left/right",
                "grid": "three or more panes including graphics",
            },
        }
    }
    try:
        result = evaluate(summary, questions, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return None, 0.0, [f"Laya layout classify skipped: {exc}"]
    answers = result.get("answers") or {}
    ans = answers.get("layout") if isinstance(answers, dict) else None
    choice = choice_value(ans if isinstance(ans, dict) else None)
    conf = 0.0
    if isinstance(ans, dict):
        try:
            conf = float(ans.get("confidence") or ans.get("probability") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        if conf <= 0 and isinstance(ans.get("probabilities"), dict) and choice:
            try:
                conf = float(ans["probabilities"].get(choice) or 0.0)
            except (TypeError, ValueError):
                conf = 0.0
        if conf <= 0 and choice:
            conf = 0.55
    return choice, conf, []


def scout_team_actions(
    team_state: dict[str, Any],
    *,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Ask Laya what a team actually did this match (typed multi-question scout card)."""
    if not is_available(api_key):
        return {"available": False, "answers": {}}
    questions = {
        "auto_hub": {
            "type": "noul",
            "instructions": "Did this team score in the hub / speaker during autonomous?",
        },
        "teleop_hub": {
            "type": "score",
            "instructions": "How active was this team at scoring in teleop?",
            "criteria": ["idle / defense only", "occasional cycles", "consistent high scorer"],
        },
        "defense": {
            "type": "noul",
            "instructions": "Did this team play substantial defense?",
        },
        "climb": {
            "type": "choice",
            "instructions": "Endgame outcome for this team?",
            "criteria": {
                "none": "no climb attempt",
                "park": "parked / low rung",
                "climb": "successful climb / trap",
                "fail": "attempted but failed",
            },
        },
        "role": {
            "type": "choice",
            "instructions": "Primary role this match?",
            "criteria": {
                "scorer": "cycle hub/speaker",
                "defense": "guard / disrupt",
                "support": "feed / midfield",
                "broken": "disabled or barely moving",
            },
        },
    }
    try:
        result = evaluate(team_state, questions, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "error": str(exc), "answers": {}}
    return {
        "available": True,
        "backend": result.get("backend"),
        "model": result.get("model"),
        "answers": result.get("answers") or {},
    }
