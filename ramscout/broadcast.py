"""Read match identity, teams, and scores from the video itself.

Primary sources, in order:
1. YouTube title + description (FIRST VODs usually name the match)
2. On-screen scorebug / alliance graphic via OCR of sampled frames
TBA is optional enrichment for nicknames, not required.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ramscout.titles import TitleHints, parse_match_title

log = logging.getLogger(__name__)

_BLUE_TEAMS = re.compile(
    r"\bblue(?:\s+alliance)?[:\s]+(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})",
    re.IGNORECASE,
)
_RED_TEAMS = re.compile(
    r"\bred(?:\s+alliance)?[:\s]+(\d{1,4})\D+(\d{1,4})\D+(\d{1,4})",
    re.IGNORECASE,
)
_BLUE_SCORE = re.compile(r"\bblue\D{0,16}(\d{1,3})\b", re.IGNORECASE)
_RED_SCORE = re.compile(r"\bred\D{0,16}(\d{1,3})\b", re.IGNORECASE)
_NAMED_SCORE = re.compile(
    r"(?:final\s+score|score)\s*[:\s]*blue\D{0,8}(\d{1,3})\D{1,12}red\D{0,8}(\d{1,3})",
    re.IGNORECASE,
)
_NAMED_SCORE_FLIP = re.compile(
    r"(?:final\s+score|score)\s*[:\s]*red\D{0,8}(\d{1,3})\D{1,12}blue\D{0,8}(\d{1,3})",
    re.IGNORECASE,
)
_PAIR_SCORE = re.compile(r"\b(\d{1,3})\s*[-–:]\s*(\d{1,3})\b")
_VS_TEAMS = re.compile(
    r"\b(\d{1,4})\D{1,8}(\d{1,4})\D{1,8}(\d{1,4})\s+vs\.?\s+(\d{1,4})\D{1,8}(\d{1,4})\D{1,8}(\d{1,4})\b",
    re.IGNORECASE,
)


@dataclass
class OverlayReading:
    blue_teams: list[str] = field(default_factory=list)
    red_teams: list[str] = field(default_factory=list)
    blue_score: int | None = None
    red_score: int | None = None
    match_label: str = ""
    year: int | None = None
    event_name: str | None = None
    comp_level: str | None = None
    match_number: int | None = None
    sources: list[str] = field(default_factory=list)
    raw_text: str = ""
    confidence: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_broadcast_text(title: str = "", description: str = "", url: str = "") -> OverlayReading:
    """Pull match/team/score hints from title + description (no TBA)."""
    hints = parse_match_title(title, url)
    blob = f"{title}\n{description}"
    reading = OverlayReading(
        year=hints.year,
        event_name=hints.event_name,
        comp_level=hints.comp_level,
        match_number=hints.match_number,
        match_label=_match_label(hints),
        raw_text=blob.strip(),
        sources=["youtube_title"] if title else [],
    )
    if description.strip():
        reading.sources.append("youtube_description")

    blue = _BLUE_TEAMS.search(blob)
    red = _RED_TEAMS.search(blob)
    vs = _VS_TEAMS.search(blob)
    if blue:
        reading.blue_teams = [blue.group(1), blue.group(2), blue.group(3)]
    if red:
        reading.red_teams = [red.group(1), red.group(2), red.group(3)]
    if (not reading.blue_teams or not reading.red_teams) and vs:
        reading.blue_teams = reading.blue_teams or [vs.group(1), vs.group(2), vs.group(3)]
        reading.red_teams = reading.red_teams or [vs.group(4), vs.group(5), vs.group(6)]

    named = _NAMED_SCORE.search(blob)
    named_flip = _NAMED_SCORE_FLIP.search(blob)
    if named:
        reading.blue_score = int(named.group(1))
        reading.red_score = int(named.group(2))
    elif named_flip:
        reading.red_score = int(named_flip.group(1))
        reading.blue_score = int(named_flip.group(2))
    else:
        reading.blue_score = _score_near(_BLUE_SCORE.findall(blob), reading.blue_teams)
        reading.red_score = _score_near(_RED_SCORE.findall(blob), reading.red_teams)

    reading.confidence = _confidence(reading)
    return reading


def read_overlay_from_video(
    video_path: Path,
    existing: OverlayReading | None = None,
    year: int | None = None,
) -> OverlayReading:
    """Sample scorebug bands from a VOD and merge into the title/description reading."""
    import cv2

    reading = existing or OverlayReading()
    top_band, bottom_band = _overlay_bands(year or reading.year)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        reading.sources.append("overlay_unreadable")
        return reading

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    texts: list[str] = []
    blue_scores: list[int] = []
    red_scores: list[int] = []
    fracs = (0.08, 0.25, 0.55, 0.78, 0.92)
    for frac in fracs:
        frame_i = int(total * frac) if total else int(fps * 20 * frac)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(frame_i, 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        text = _ocr_scorebug(frame, top_band=top_band, bottom_band=bottom_band)
        if text:
            texts.append(text)
        parsed = parse_broadcast_text(title=text, description="")
        if parsed.blue_score is not None:
            blue_scores.append(parsed.blue_score)
        if parsed.red_score is not None:
            red_scores.append(parsed.red_score)
        for team in parsed.blue_teams:
            if team not in reading.blue_teams:
                reading.blue_teams.append(team)
        for team in parsed.red_teams:
            if team not in reading.red_teams:
                reading.red_teams.append(team)
        region_blue, region_red = _scores_from_color_regions(frame)
        if region_blue is not None:
            blue_scores.append(region_blue)
        if region_red is not None:
            red_scores.append(region_red)

    cap.release()
    if texts:
        reading.raw_text = (reading.raw_text + "\n" + "\n".join(texts)).strip()
        reading.sources.append("video_scorebug")
    if blue_scores:
        reading.blue_score = _majority_or_last(blue_scores)
    if red_scores:
        reading.red_score = _majority_or_last(red_scores)
    reading.confidence = _confidence(reading)
    return reading


def match_from_overlay(reading: OverlayReading) -> dict[str, Any]:
    """Build a TBA-shaped match dict from video/title overlay data."""
    year = reading.year or 2026
    event = (reading.event_name or "unknown-event").lower()
    slug = re.sub(r"[^a-z0-9]+", "", event)[:8] or "video"
    level = reading.comp_level or "qm"
    number = reading.match_number or 0
    key = f"{year}{slug}_{level}{number}" if number else f"{year}{slug}_video"

    def alliance(color: str, teams: list[str], score: int | None) -> dict[str, Any]:
        keys = [f"frc{t}" for t in teams]
        return {"score": score if score is not None else None, "team_keys": keys}

    teams: dict[str, Any] = {}
    for color, nums in (("blue", reading.blue_teams), ("red", reading.red_teams)):
        for num in nums:
            teams[f"frc{num}"] = {
                "key": f"frc{num}",
                "team_number": int(num) if str(num).isdigit() else num,
                "nickname": "",
                "alliance": color,
            }

    blue_score = reading.blue_score
    red_score = reading.red_score
    winner = ""
    if blue_score is not None and red_score is not None:
        if blue_score > red_score:
            winner = "blue"
        elif red_score > blue_score:
            winner = "red"

    return {
        "key": key,
        "event_key": f"{year}{slug}",
        "comp_level": level,
        "set_number": 1,
        "match_number": number,
        "time": None,
        "winning_alliance": winner,
        "alliances": {
            "blue": alliance("blue", reading.blue_teams, blue_score),
            "red": alliance("red", reading.red_teams, red_score),
        },
        "score_breakdown": None,
        "videos": [],
        "teams": teams,
        "zebra": None,
        "source": {
            "scores": "video" if blue_score is not None or red_score is not None else "unknown",
            "teams": "video" if reading.blue_teams or reading.red_teams else "unknown",
            "channels": reading.sources,
        },
    }


def merge_tba(video_match: dict[str, Any], tba_match: dict[str, Any] | None) -> dict[str, Any]:
    """Video overlay wins for scores/teams; TBA fills nicknames and official extras."""
    if not tba_match:
        return video_match
    merged = dict(video_match)
    source = dict(video_match.get("source") or {})
    source["tba"] = True
    # Nicknames from TBA.
    for key, info in (tba_match.get("teams") or {}).items():
        slot = merged.setdefault("teams", {}).setdefault(key, {"key": key, "team_number": info.get("team_number"), "alliance": info.get("alliance")})
        if info.get("nickname"):
            slot["nickname"] = info["nickname"]
    # If the video never saw teams, take TBA alliances.
    for color in ("blue", "red"):
        video_keys = merged.get("alliances", {}).get(color, {}).get("team_keys") or []
        tba_keys = tba_match.get("alliances", {}).get(color, {}).get("team_keys") or []
        if not video_keys and tba_keys:
            merged["alliances"][color]["team_keys"] = tba_keys
            source["teams"] = "tba"
        video_score = merged.get("alliances", {}).get(color, {}).get("score")
        tba_score = tba_match.get("alliances", {}).get(color, {}).get("score")
        if video_score is None and tba_score is not None:
            merged["alliances"][color]["score"] = tba_score
            source["scores"] = "tba"
    if not merged.get("score_breakdown"):
        merged["score_breakdown"] = tba_match.get("score_breakdown")
    if tba_match.get("key"):
        merged["tba_key"] = tba_match.get("key")
    if tba_match.get("zebra") and not merged.get("zebra"):
        merged["zebra"] = tba_match.get("zebra")
    if tba_match.get("videos") and not merged.get("videos"):
        merged["videos"] = tba_match.get("videos")
    merged["source"] = source
    return merged


def _match_label(hints: TitleHints) -> str:
    if hints.comp_level == "qm" and hints.match_number:
        return f"Qualification Match {hints.match_number}"
    if hints.match_number:
        return f"{(hints.comp_level or 'match').upper()} {hints.match_number}"
    return hints.title


def _score_near(candidates: list[str], team_numbers: Iterable[str]) -> int | None:
    blocked = {int(t) for t in team_numbers if str(t).isdigit()}
    values = [int(c) for c in candidates if c.isdigit() and int(c) not in blocked]
    values = [v for v in values if v <= 400]
    return values[-1] if values else None


def _confidence(reading: OverlayReading) -> float:
    score = 0.15 if reading.match_number else 0.0
    if reading.blue_teams and reading.red_teams:
        score += 0.4
    if reading.blue_score is not None and reading.red_score is not None:
        score += 0.4
    if "video_scorebug" in reading.sources:
        score += 0.05
    return round(min(score, 0.95), 2)


def _majority_or_last(values: list[int]) -> int:
    common = Counter(values).most_common(1)
    return common[0][0] if common else values[-1]


def _overlay_bands(year: int | None) -> tuple[tuple[float, float], tuple[float, float]]:
    from ramscout.gameconfig import load_game

    overlay = (load_game(year).get("overlay") or {})
    top = overlay.get("top_band") or [0.0, 0.14]
    bottom = overlay.get("bottom_band") or [0.82, 1.0]
    return (float(top[0]), float(top[1])), (float(bottom[0]), float(bottom[1]))


def _ocr_scorebug(
    frame_bgr: np.ndarray,
    top_band: tuple[float, float] = (0.0, 0.14),
    bottom_band: tuple[float, float] = (0.82, 1.0),
) -> str:
    import cv2

    h, w = frame_bgr.shape[:2]
    slices = (
        frame_bgr[max(int(h * top_band[0]), 0) : max(int(h * top_band[1]), 1), :],
        frame_bgr[max(int(h * bottom_band[0]), 0) : max(int(h * bottom_band[1]), 1), :],
    )
    chunks: list[str] = []
    for band in slices:
        if band.size == 0:
            continue
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = ""
        if hasattr(cv2, "text") and hasattr(cv2.text, "OCRTesseract_create"):
            try:
                text = cv2.text.OCRTesseract_create().run(th, 0) or ""
            except Exception:  # noqa: BLE001
                text = ""
        if not text:
            text = _tesseract(th)
        if text:
            chunks.append(text)
    return " ".join(chunks)


def _tesseract(image: np.ndarray) -> str:
    try:
        import pytesseract
    except ImportError:
        return ""
    try:
        return pytesseract.image_to_string(image, config="--psm 6") or ""
    except Exception:  # noqa: BLE001
        return ""


def _scores_from_color_regions(frame_bgr: np.ndarray) -> tuple[int | None, int | None]:
    """Best-effort score digits from blue/red tinted scorebug corners."""
    import cv2

    h, w = frame_bgr.shape[:2]
    top = frame_bgr[0 : max(int(h * 0.16), 1), :]
    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, (95, 50, 50), (135, 255, 255))
    red = cv2.inRange(hsv, (0, 50, 50), (12, 255, 255)) | cv2.inRange(hsv, (165, 50, 50), (180, 255, 255))
    return _largest_digit_blob(top, blue), _largest_digit_blob(top, red)


def _largest_digit_blob(bgr: np.ndarray, color_mask: np.ndarray) -> int | None:
    import cv2

    if int(np.count_nonzero(color_mask)) < 40:
        return None
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = cv2.bitwise_and(th, th, mask=cv2.dilate(color_mask, np.ones((9, 9), np.uint8)))
    text = _tesseract(masked)
    nums = [int(n) for n in re.findall(r"\d{1,3}", text or "") if int(n) <= 400]
    return nums[-1] if nums else None
