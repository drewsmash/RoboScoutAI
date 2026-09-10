"""Pluggable robot-tracking strategies for RamScoutAI.

Local strategies work offline with OpenCV. Optional cloud strategies
(OpenAI Vision, Google Gemini) can refine detections when API keys are set.
"""

from __future__ import annotations

from ramscout.trackers.ensemble import TRACKER_MODES, list_strategies, resolve_strategies
from ramscout.trackers.types import Detection, TrackerContext

__all__ = [
    "TRACKER_MODES",
    "Detection",
    "TrackerContext",
    "list_strategies",
    "resolve_strategies",
]
