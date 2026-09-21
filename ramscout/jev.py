"""Backward-compatible Jev helpers — implementation lives in ``ramscout.laya``.

New code should import from ``ramscout.laya``. This module re-exports the same
API so older call sites and tests keep working while Laya is the preferred
on-machine backend (~10× faster than cloud Jev).
"""

from __future__ import annotations

import httpx

from ramscout.laya import (  # noqa: F401
    DEFAULT_TIMEOUT_S,
    EVALUATE_URL,
    FALSE_THRESHOLD,
    JEV_MODEL,
    MAX_EVENT_QUESTIONS,
    TRUE_THRESHOLD,
    backend_status,
    boolean_probability,
    choice_value,
    classify_camera_layout,
    evaluate,
    gateway_api_key,
    is_available,
    local_available,
    scout_team_actions,
    verify_scout_events,
)

__all__ = [
    "DEFAULT_TIMEOUT_S",
    "EVALUATE_URL",
    "FALSE_THRESHOLD",
    "JEV_MODEL",
    "MAX_EVENT_QUESTIONS",
    "TRUE_THRESHOLD",
    "backend_status",
    "boolean_probability",
    "choice_value",
    "classify_camera_layout",
    "evaluate",
    "gateway_api_key",
    "httpx",
    "is_available",
    "local_available",
    "scout_team_actions",
    "verify_scout_events",
]
