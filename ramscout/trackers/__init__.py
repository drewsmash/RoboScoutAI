"""Pluggable robot-tracking strategies for RoboScoutAI.

Local strategies work offline with OpenCV. Optional cloud strategies
(OpenAI Vision, Google Gemini) can refine detections when API keys are set.
"""

from __future__ import annotations

from ramscout.trackers.alliance import AllianceCalibrator, AllianceVoter
from ramscout.trackers.ensemble import (
    POTATO_STRATEGIES,
    TRACKER_MODES,
    list_strategies,
    resolve_strategies,
)
from ramscout.trackers.field_gate import FieldGate
from ramscout.trackers.types import Detection, TrackerContext

__all__ = [
    "POTATO_STRATEGIES",
    "TRACKER_MODES",
    "AllianceCalibrator",
    "AllianceVoter",
    "Detection",
    "FieldGate",
    "TrackerContext",
    "list_strategies",
    "resolve_strategies",
]
