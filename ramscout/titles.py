"""Parse FIRST match YouTube titles into TBA-friendly match hints."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

_YT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

_YEAR_RE = re.compile(r"\b(20[1-3]\d)\b")
_QM_RE = re.compile(
    r"\b(?:qualification(?:s)?\s+match|quals?\.?|qual(?:ification)?s?)\s*#?\s*(\d+)\b",
    re.IGNORECASE,
)
_PLAYOFF_MATCH_RE = re.compile(r"\bplayoff(?:s)?\s+match\s*#?\s*(\d+)\b", re.IGNORECASE)
_FINALS_RE = re.compile(r"\bfinals?\s+(?:match\s*)?#?\s*(\d+)\b", re.IGNORECASE)
_QF_RE = re.compile(
    r"\bquarter[- ]?finals?(?:\s+set)?\s*(\d+)?(?:\s+match\s*(\d+))?\b",
    re.IGNORECASE,
)
_SF_RE = re.compile(
    r"\bsemi[- ]?finals?(?:\s+set)?\s*(\d+)?(?:\s+match\s*(\d+))?\b",
    re.IGNORECASE,
)
_EVENT_DASH_RE = re.compile(
    r"(20[1-3]\d)\s+(.+?)\s+[-–—]\s+(qualification|quals?|playoff|quarter|semi|final).+",
    re.IGNORECASE,
)
_EVENT_SUFFIX_RE = re.compile(
    r"(?:at|from)\s+the\s+(.+?)(?:\s+FIRST Robotics.*)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TitleHints:
    year: int | None = None
    event_name: str | None = None
    comp_level: str | None = None
    set_number: int | None = None
    match_number: int | None = None
    video_id: str | None = None
    title: str = ""


def extract_youtube_id(url: str) -> str | None:
    """Return an 11-character YouTube video id from a URL or bare id."""
    raw = (url or "").strip()
    if _YT_ID_RE.match(raw):
        return raw

    parsed = urlparse(raw)
    host = (parsed.netloc or "").lower()
    path = parsed.path or ""

    if "youtu.be" in host:
        slug = path.lstrip("/").split("/")[0]
        return slug if _YT_ID_RE.match(slug) else None

    if "youtube.com" in host or "youtube-nocookie.com" in host:
        qs = parse_qs(parsed.query)
        if "v" in qs and _YT_ID_RE.match(qs["v"][0]):
            return qs["v"][0]
        parts = [p for p in path.split("/") if p]
        if parts and parts[0] in {"embed", "shorts", "live", "v"} and len(parts) > 1:
            return parts[1] if _YT_ID_RE.match(parts[1]) else None
        if parts and _YT_ID_RE.match(parts[-1]):
            return parts[-1]
    return None


def parse_match_title(title: str, url: str | None = None) -> TitleHints:
    """Extract year / event / match identity from a FIRST broadcast title."""
    text = (title or "").strip()
    video_id = extract_youtube_id(url or "") if url else None

    year = None
    year_match = _YEAR_RE.search(text)
    if year_match:
        year = int(year_match.group(1))

    comp_level = None
    set_number = 1
    match_number = None

    qm = _QM_RE.search(text)
    playoff = _PLAYOFF_MATCH_RE.search(text)
    finals = _FINALS_RE.search(text)
    qf = _QF_RE.search(text)
    sf = _SF_RE.search(text)

    if qm:
        comp_level = "qm"
        match_number = int(qm.group(1))
    elif playoff:
        # TBA 2023+ double-elim playoffs use sf / f depending on round.
        n = int(playoff.group(1))
        comp_level = "sf"
        match_number = n
    elif finals:
        comp_level = "f"
        match_number = int(finals.group(1))
    elif qf:
        comp_level = "qf"
        set_number = int(qf.group(1) or 1)
        match_number = int(qf.group(2) or 1)
    elif sf:
        comp_level = "sf"
        set_number = int(sf.group(1) or 1)
        match_number = int(sf.group(2) or 1)

    event_name = _extract_event_name(text)
    return TitleHints(
        year=year,
        event_name=event_name,
        comp_level=comp_level,
        set_number=set_number,
        match_number=match_number,
        video_id=video_id,
        title=text,
    )


def _extract_event_name(text: str) -> str | None:
    dash = _EVENT_DASH_RE.search(text)
    if dash:
        return _clean_event_name(dash.group(2))
    suffix = _EVENT_SUFFIX_RE.search(text)
    if suffix:
        return _clean_event_name(suffix.group(1))
    return None


def _clean_event_name(name: str) -> str:
    cleaned = re.sub(r"\s+", " ", name).strip(" -–—")
    cleaned = re.sub(r"\bFIRST Robotics Competition\b", "", cleaned, flags=re.I)
    return cleaned.strip(" -–—")


def normalize_event_name(name: str) -> str:
    text = name.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    for token in (
        "presented by",
        "sponsored by",
        "first",
        "frc",
        "district",
        "regional",
        "event",
        "championship",
        "the",
    ):
        text = text.replace(token, " ")
    return re.sub(r"\s+", " ", text).strip()
