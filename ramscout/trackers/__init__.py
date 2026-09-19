"""Pluggable robot-tracking strategies for RoboScoutAI.

Local strategies work offline with OpenCV. Optional cloud strategies
(OpenAI Vision, Google Gemini) can refine detections when API keys are set.
"""

from __future__ import annotations

from ramscout.trackers.ensemble import (
    POTATO_STRATEGIES,
    TRACKER_MODES,
    list_strategies,
    resolve_strategies,
)
from ramscout.trackers.types import Detection, TrackerContext

__all__ = [
    "POTATO_STRATEGIES",
    "TRACKER_MODES",
    "Detection",
    "TrackerContext",
    "list_strategies",
    "resolve_strategies",
]
