"""Alliance color from bumpers and team-number assignment."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np

from ramscout.field import FIELD_LENGTH


def bumper_alliance(crop_bgr: np.ndarray) -> str:
    """Classify a robot crop as red / blue / unknown from bumper hue."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    red_lo = cv2.inRange(hsv, (0, 80, 60), (12, 255, 255))
    red_hi = cv2.inRange(hsv, (165, 80, 60), (180, 255, 255))
    blue = cv2.inRange(hsv, (95, 70, 50), (135, 255, 255))
    red_count = int(np.count_nonzero(red_lo | red_hi))
    blue_count = int(np.count_nonzero(blue))
    if red_count > blue_count * 1.25 and red_count > 40:
        return "red"
    if blue_count > red_count * 1.25 and blue_count > 40:
        return "blue"
    return "unknown"


def ocr_digits(crop_bgr: np.ndarray) -> str:
    """Best-effort bumper OCR. Returns digits only; empty on failure."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return ""
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = ""
    if hasattr(cv2, "text") and hasattr(cv2.text, "OCRTesseract_create"):
        try:
            ocr = cv2.text.OCRTesseract_create()
            text = ocr.run(th, 0)
        except Exception:  # noqa: BLE001
            text = ""
    digits = "".join(ch for ch in (text or "") if ch.isdigit())
    return digits


def constrain_to_teams(raw: str, team_numbers: Iterable[int | str]) -> str | None:
    allowed = {str(n) for n in team_numbers}
    if raw in allowed:
        return raw
    for team in allowed:
        if team in raw or raw in team:
            return team
    return None


def assign_by_start(
    tracks: list[dict[str, Any]],
    blue_teams: list[str],
    red_teams: list[str],
) -> dict[int, str]:
    """Map tracker ids to TBA teams using first-seen field pose.

    Blue robots are sorted by y (station 1–3), same for red. This is a
    starting guess — the UI lets a scout swap any assignment.
    """
    first: dict[int, dict[str, Any]] = {}
    for sample in tracks:
        tid = int(sample["track_id"])
        if tid not in first:
            first[tid] = sample

    blue_ids: list[tuple[float, int]] = []
    red_ids: list[tuple[float, int]] = []
    for tid, sample in first.items():
        alliance = sample.get("alliance") or (
            "blue" if float(sample["x"]) < FIELD_LENGTH / 2 else "red"
        )
        if alliance == "blue":
            blue_ids.append((float(sample["y"]), tid))
        else:
            red_ids.append((float(sample["y"]), tid))

    blue_ids.sort()
    red_ids.sort()
    mapping: dict[int, str] = {}
    for i, (_, tid) in enumerate(blue_ids):
        if i < len(blue_teams):
            mapping[tid] = str(blue_teams[i])
    for i, (_, tid) in enumerate(red_ids):
        if i < len(red_teams):
            mapping[tid] = str(red_teams[i])
    return mapping


def majority_alliance(samples: list[dict[str, Any]]) -> str:
    votes = Counter(s.get("alliance") for s in samples if s.get("alliance") in {"red", "blue"})
    if not votes:
        xs = [float(s["x"]) for s in samples]
        return "blue" if (sum(xs) / max(len(xs), 1)) < FIELD_LENGTH / 2 else "red"
    return votes.most_common(1)[0][0]
