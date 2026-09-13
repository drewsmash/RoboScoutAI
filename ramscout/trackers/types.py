"""Shared types for robot trackers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


@dataclass
class Detection:
    track_id: int
    bbox: list[float]  # x1, y1, x2, y2 in crop coordinates
    source: str
    confidence: float = 0.5
    alliance: str = "unknown"
    team: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "track_id": int(self.track_id),
            "bbox": [float(v) for v in self.bbox],
            "source": self.source,
            "confidence": float(self.confidence),
            "alliance": self.alliance,
            "team": self.team,
        }


@dataclass
class TrackerContext:
    frame_w: int
    frame_h: int
    crop_w: int
    crop_h: int
    t: float = 0.0
    frame_index: int = 0
    team_numbers: list[str] = field(default_factory=list)
    openai_key: str = ""
    google_key: str = ""
    openai_model: str = "gpt-4o-mini"
    google_model: str = "gemini-3.5-flash"
    extras: dict[str, Any] = field(default_factory=dict)


class TrackerStrategy(Protocol):
    name: str
    kind: str  # local | neural | cloud
    description: str

    def available(self, ctx: TrackerContext | None = None) -> bool: ...

    def reset(self) -> None: ...

    def detect(self, cropped: np.ndarray, ctx: TrackerContext) -> list[Detection]: ...
