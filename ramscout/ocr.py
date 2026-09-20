"""Bumper-number OCR → team identity for stable tracks.

FRC bumpers carry the team number in white ≥ 4-inch digits on a red or blue
band. Reading them from a broadcast is marginal (small, motion-blurred), so
this module is built around *voting*: every few frames a stable track's
bumper band is enhanced and read, the digits are matched against the six
teams in the match, and a per-track tally accumulates. A track only gets a
team when one candidate clearly leads and no other track already owns it.

Backends, first available wins:
- ``tesseract`` via ``pytesseract`` (needs the tesseract binary),
- ``cv2.text`` OCRTesseract (opencv-contrib),
- OpenAI Vision (when a key is set) — one small crop per call, sparse.
Absent all three, ``available`` is False and the tracker skips OCR.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter, defaultdict
from typing import Any, Callable, Iterable

import numpy as np

log = logging.getLogger(__name__)

_DIGITS = re.compile(r"\d{1,5}")


def _tesseract_reader() -> Callable[[np.ndarray], str] | None:
    try:
        import pytesseract  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    if not shutil.which("tesseract") and not getattr(pytesseract.pytesseract, "tesseract_cmd", ""):
        return None
    try:
        pytesseract.get_tesseract_version()
    except Exception:  # noqa: BLE001
        return None

    def _read(img: np.ndarray) -> str:
        try:
            return pytesseract.image_to_string(
                img, config="--psm 7 -c tessedit_char_whitelist=0123456789"
            )
        except Exception:  # noqa: BLE001
            return ""

    return _read


def _cv2_text_reader() -> Callable[[np.ndarray], str] | None:
    try:
        import cv2

        if not (hasattr(cv2, "text") and hasattr(cv2.text, "OCRTesseract_create")):
            return None
        ocr = cv2.text.OCRTesseract_create(None, "eng", "0123456789", 3, 7)
    except Exception:  # noqa: BLE001
        return None

    def _read(img: np.ndarray) -> str:
        try:
            return ocr.run(img, 0) or ""
        except Exception:  # noqa: BLE001
            return ""

    return _read


def _openai_reader(api_key: str, model: str = "gpt-4o-mini") -> Callable[[np.ndarray], str] | None:
    if not api_key:
        return None
    try:
        import base64

        import cv2
        import httpx
    except Exception:  # noqa: BLE001
        return None

    def _read(img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not ok:
            return ""
        b64 = base64.b64encode(bytes(buf)).decode("ascii")
        payload = {
            "model": model,
            "max_tokens": 12,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Read the FRC team number printed on this robot bumper. Reply with digits only, or NONE."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"}},
                    ],
                }
            ],
        }
        try:
            resp = httpx.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
                timeout=20.0,
            )
            resp.raise_for_status()
            return str(resp.json()["choices"][0]["message"]["content"])
        except Exception as exc:  # noqa: BLE001
            log.info("OpenAI bumper OCR failed: %s", exc)
            return ""

    return _read


def enhance_bumper(crop_bgr: np.ndarray, *, target_h: int = 64) -> np.ndarray | None:
    """Isolate the bumper band and boost the white digits for OCR."""
    import cv2

    if crop_bgr is None or crop_bgr.size == 0:
        return None
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 14:
        return None
    band = crop_bgr[int(h * 0.55) :, :]
    if band.shape[0] < 6:
        band = crop_bgr
    scale = target_h / max(band.shape[0], 1)
    band = cv2.resize(band, (max(int(band.shape[1] * scale), 16), target_h), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    # White digits: low saturation, high value.
    white = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    digits = cv2.bitwise_and(th, cv2.dilate(white, np.ones((3, 3), np.uint8)))
    if np.count_nonzero(digits) < 0.02 * digits.size:
        digits = th
    # Tesseract likes dark text on white.
    inv = cv2.bitwise_not(digits)
    inv = cv2.copyMakeBorder(inv, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return inv


_CONFUSABLE = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "|": "1", "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "g": "9", "T": "7"})
_ALNUM = re.compile(r"[0-9A-Za-z|]{3,6}")


def normalize_digits(raw: str) -> str:
    """Map OCR confusables (O→0, S→5, B→8 …) inside mostly-numeric tokens."""
    out = raw or ""
    for tok in _ALNUM.findall(out):
        n_digits = sum(ch.isdigit() for ch in tok)
        if n_digits >= 2 and n_digits >= len(tok) - 2:
            out = out.replace(tok, tok.translate(_CONFUSABLE))
    return out


def match_team(raw: str, teams: Iterable[str]) -> tuple[str | None, float]:
    """Best team match for an OCR string with a 0..1 confidence."""
    allowed = [str(t) for t in teams]
    tokens = _DIGITS.findall(normalize_digits(raw))
    if not tokens or not allowed:
        return None, 0.0
    best: tuple[str | None, float] = (None, 0.0)
    for tok in tokens:
        for team in allowed:
            if tok == team:
                return team, 1.0
            if len(tok) >= 3 and (tok in team or team in tok):
                score = 0.6 * min(len(tok), len(team)) / max(len(tok), len(team))
                if score > best[1]:
                    best = (team, score)
            elif len(tok) >= 3 and len(tok) == len(team):
                same = sum(1 for a, b in zip(tok, team) if a == b)
                if same >= len(team) - 1:
                    score = 0.45
                    if score > best[1]:
                        best = (team, score)
    return best


class BumperOCR:
    """Per-track digit voting on bumper crops."""

    def __init__(self, teams: Iterable[str], *, openai_key: str = "", reader: Callable[[np.ndarray], str] | None = None) -> None:
        self.teams = [str(t) for t in teams if str(t)]
        self.votes: dict[int, Counter] = defaultdict(Counter)
        self.attempts: dict[int, int] = defaultdict(int)
        self.backend = "none"
        self.reader = reader
        if self.reader is not None:
            self.backend = "custom"
        else:
            for name, factory in (("tesseract", _tesseract_reader), ("opencv-text", _cv2_text_reader)):
                r = factory()
                if r is not None:
                    self.reader, self.backend = r, name
                    break
            if self.reader is None and openai_key:
                r = _openai_reader(openai_key)
                if r is not None:
                    self.reader, self.backend = r, "openai-vision"
        self.max_attempts_per_track = 40 if self.backend != "openai-vision" else 6

    @property
    def available(self) -> bool:
        return self.reader is not None and bool(self.teams)

    def observe(self, track_id: int, crop_bgr: np.ndarray) -> str | None:
        """Read one crop; return the team only once the tally is decisive."""
        if not self.available or self.attempts[track_id] >= self.max_attempts_per_track:
            return self.current(track_id)
        img = enhance_bumper(crop_bgr)
        if img is None:
            return self.current(track_id)
        self.attempts[track_id] += 1
        raw = self.reader(img) if self.reader else ""
        team, score = match_team(raw, self.teams)
        if team is not None and score > 0:
            self.votes[track_id][team] += score
        return self.current(track_id)

    def current(self, track_id: int, *, min_score: float = 1.2, min_margin: float = 0.6) -> str | None:
        tally = self.votes.get(track_id)
        if not tally:
            return None
        ranked = tally.most_common(2)
        best, score = ranked[0]
        runner = ranked[1][1] if len(ranked) > 1 else 0.0
        if score >= min_score and score - runner >= min_margin:
            return best
        return None

    def assignments(self) -> dict[int, str]:
        """Decisive team per track, each team used at most once (highest tally wins)."""
        pairs: list[tuple[float, int, str]] = []
        for tid, tally in self.votes.items():
            team = self.current(tid)
            if team:
                pairs.append((tally[team], tid, team))
        pairs.sort(reverse=True)
        used: set[str] = set()
        out: dict[int, str] = {}
        for _score, tid, team in pairs:
            if team in used:
                continue
            used.add(team)
            out[tid] = team
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "available": self.available,
            "tracks_read": len(self.assignments()),
            "attempts": int(sum(self.attempts.values())),
        }
