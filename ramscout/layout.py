"""Broadcast signal decomposition: split an FRC webcast frame into its panes.

Event broadcasts are compositions: one or two field cameras, close-up
"alliance" cameras, a scorebug, sponsor / replay panes, letterbox bars. The
tracker must run on *the overview pane only*, side panes must feed the
scoring / climb reasoning, and the scorebug must be masked, so the first job
is to recover that composition from pixels alone:

1. **Pane boundaries** — persistent, straight, full-span edges. A pane edge is
   a row (or column) where the |Sobel| response is strong across ≥ ~80 % of
   the span in *every* sampled frame (temporal median of the coverage
   profile). Field walls / tape lines are straight too, but they never span
   the full frame and they are not perfectly axis aligned, so they drop out.
   Dark gutters and letterbox / pillarbox bars are found from the temporal
   max brightness.
2. **Static graphics** — scorebug / lower-third: pixels whose temporal std is
   ~0 while their edge density is high (digits, logos). The largest such
   blob in the top or bottom band is the scorebug.
3. **Pane roles** — per-pane content features from the temporal median
   frame (the background with the robots averaged away) and the temporal
   std map: carpet fraction (grey floor), saturated red / blue mass, live
   fraction (how much of the pane moves), global change (a panning close-up
   camera changes everywhere), size. A wide static camera on a grey floor
   with a handful of small moving blobs is the *overview*; a second such pane
   is ``overview_alt``; panning / high-live panes are alliance close-ups,
   biased blue / red by their colour cast; static high-contrast panes are
   graphics.
4. **Timeline** — the director switches layouts mid-match. Per-frame
   boundary signatures are grouped into runs; each run is analysed as its own
   segment. Consumers pause tracking when a segment has no overview pane.

Everything is OpenCV / numpy on ≤ 480-px-wide thumbnails, so a 3-minute match
segments in a few seconds.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ProgressFn = Callable[[str, float], None]

ANALYSIS_WIDTH = 480
EDGE_THRESHOLD = 34.0  # |Sobel| on 0..255 gray, 3x3 kernel, scaled by 1/4
CUT_COVERAGE = 0.78  # fraction of the span that must carry the edge
DARK_LEVEL = 26.0  # letterbox / gutter brightness (temporal max)
MIN_PANE_FRAC = 0.12  # panes smaller than this along an axis are noise
STATIC_STD = 4.5  # temporal std below which a pixel is "static graphics"
LIVE_STD = 7.0  # temporal std above which a pixel is "moving"

ROLE_OVERVIEW = "overview"
ROLE_OVERVIEW_ALT = "overview_alt"
ROLE_BLUE = "blue_side"
ROLE_RED = "red_side"
ROLE_SIDELINE = "sideline"
ROLE_GRAPHICS = "graphics"
ROLE_SCOREBUG = "scorebug"
ROLE_OTHER = "other"

ROLE_PURPOSE = {
    ROLE_OVERVIEW: "Wide field camera — drives robot tracking, depth / BEV and the top-down map",
    ROLE_OVERVIEW_ALT: "Second wide field camera — cross-checks positions; not tracked",
    ROLE_BLUE: "Blue-side close-up — blue scoring / climb cues",
    ROLE_RED: "Red-side close-up — red scoring / climb cues",
    ROLE_SIDELINE: "Sideline / close-up camera — scoring and climb cues",
    ROLE_GRAPHICS: "Static graphics (sponsor / score panel) — ignored",
    ROLE_SCOREBUG: "Scorebug overlay — OCR for time, score and teams; masked from tracking",
    ROLE_OTHER: "Unclassified pane — ignored",
}


@dataclass
class PaneBox:
    """Normalized [0,1] pane rectangle in the full frame."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def w(self) -> float:
        return max(self.x1 - self.x0, 0.0)

    @property
    def h(self) -> float:
        return max(self.y1 - self.y0, 0.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def as_list(self) -> list[float]:
        return [round(self.x0, 4), round(self.y0, 4), round(self.x1, 4), round(self.y1, 4)]

    def pixels(self, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        x0 = int(np.clip(round(self.x0 * frame_w), 0, frame_w - 1))
        x1 = int(np.clip(round(self.x1 * frame_w), x0 + 1, frame_w))
        y0 = int(np.clip(round(self.y0 * frame_h), 0, frame_h - 1))
        y1 = int(np.clip(round(self.y1 * frame_h), y0 + 1, frame_h))
        return x0, y0, x1, y1

    def slice(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = self.pixels(w, h)
        return frame[y0:y1, x0:x1]

    def same_as(self, other: "PaneBox", tol: float = 0.03) -> bool:
        return (
            abs(self.x0 - other.x0) <= tol
            and abs(self.y0 - other.y0) <= tol
            and abs(self.x1 - other.x1) <= tol
            and abs(self.y1 - other.y1) <= tol
        )


@dataclass
class DetectedPane:
    role: str
    box: PaneBox
    confidence: float
    features: dict[str, float] = field(default_factory=dict)
    alliance_bias: str | None = None
    scorebug: list[float] | None = None  # normalized box of a scorebug *inside* this pane

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "box": self.box.as_list(),
            "crop_left": round(self.box.x0, 4),
            "crop_top": round(self.box.y0, 4),
            "crop_right": round(self.box.x1, 4),
            "crop_bottom": round(self.box.y1, 4),
            "confidence": round(float(self.confidence), 3),
            "purpose": ROLE_PURPOSE.get(self.role, ""),
            "alliance_bias": self.alliance_bias,
            "features": {k: round(float(v), 4) for k, v in self.features.items()},
            "scorebug": self.scorebug,
        }


@dataclass
class FrameLayout:
    """Decomposition of one layout segment."""

    panes: list[DetectedPane]
    letterbox: PaneBox  # active picture area after removing black bars
    row_cuts: list[float]
    col_cuts: list[float]
    scorebug: list[float] | None
    signature: str
    confidence: float
    detail: str

    def overview(self) -> DetectedPane | None:
        for pane in self.panes:
            if pane.role == ROLE_OVERVIEW:
                return pane
        return None

    def by_role(self, *roles: str) -> list[DetectedPane]:
        return [p for p in self.panes if p.role in roles]

    @property
    def mode(self) -> str:
        cams = [p for p in self.panes if p.role not in {ROLE_GRAPHICS, ROLE_SCOREBUG, ROLE_OTHER}]
        if len(cams) <= 1:
            return "single"
        sides = self.by_role(ROLE_BLUE, ROLE_RED)
        if len(sides) >= 2 and self.row_cuts:
            return "stacked_sides"
        if self.row_cuts and not self.col_cuts:
            return "stacked_top" if self.overview() and self.overview().box.y0 < 0.3 else "stacked"
        if self.col_cuts and not self.row_cuts:
            return "side_by_side"
        return "grid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "panes": [p.as_dict() for p in self.panes],
            "letterbox": self.letterbox.as_list(),
            "row_cuts": [round(v, 4) for v in self.row_cuts],
            "col_cuts": [round(v, 4) for v in self.col_cuts],
            "scorebug": self.scorebug,
            "signature": self.signature,
            "confidence": round(float(self.confidence), 3),
            "detail": self.detail,
        }


@dataclass
class LayoutSegment:
    t0: float
    t1: float
    layout: FrameLayout
    sample_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        ov = self.layout.overview()
        return {
            "t0": round(self.t0, 3),
            "t1": round(self.t1, 3),
            "mode": self.layout.mode,
            "signature": self.layout.signature,
            "overview": ov.box.as_list() if ov else None,
            "pane_roles": [p.role for p in self.layout.panes],
            "samples": int(self.sample_count),
        }


@dataclass
class LayoutTimeline:
    segments: list[LayoutSegment]
    duration: float
    fps: float
    frame_size: tuple[int, int]

    def at(self, t: float) -> LayoutSegment | None:
        for seg in self.segments:
            if seg.t0 - 1e-6 <= t < seg.t1 + 1e-6:
                return seg
        return self.segments[-1] if self.segments and t >= self.segments[-1].t1 else None

    def primary(self) -> LayoutSegment | None:
        """Longest segment that has an overview pane (else longest overall)."""
        with_ov = [s for s in self.segments if s.layout.overview() is not None]
        pool = with_ov or self.segments
        if not pool:
            return None
        return max(pool, key=lambda s: s.t1 - s.t0)

    def overview_coverage(self) -> float:
        if self.duration <= 0:
            return 0.0
        cov = sum(s.t1 - s.t0 for s in self.segments if s.layout.overview() is not None)
        return float(np.clip(cov / self.duration, 0.0, 1.0))

    def as_dict(self) -> dict[str, Any]:
        prim = self.primary()
        return {
            "duration": round(self.duration, 3),
            "fps": round(self.fps, 3),
            "frame_size": list(self.frame_size),
            "segments": [s.as_dict() for s in self.segments],
            "switches": max(len(self.segments) - 1, 0),
            "overview_coverage": round(self.overview_coverage(), 3),
            "primary": prim.as_dict() if prim else None,
        }


# ----------------------------------------------------------------- sampling


def sample_frames(
    video_path: Path | str,
    times: Sequence[float],
    *,
    width: int | None = None,
) -> tuple[list[tuple[float, np.ndarray]], dict[str, Any]]:
    """Read frames at the given times (seconds). Returns [(t, frame)], meta."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    meta: dict[str, Any] = {"fps": 30.0, "duration": 0.0, "frame_size": (0, 0)}
    if not cap.isOpened():
        return [], meta
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    meta.update(fps=fps, duration=(total / fps) if total > 0 else 0.0, frame_size=(fw, fh))
    out: list[tuple[float, np.ndarray]] = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(round(t * fps)), 0))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if width and frame.shape[1] > width:
            scale = width / frame.shape[1]
            frame = cv2.resize(frame, (width, max(1, int(round(frame.shape[0] * scale)))), interpolation=cv2.INTER_AREA)
        out.append((float(t), frame))
    cap.release()
    return out, meta


def _thumb(frame: np.ndarray, width: int = ANALYSIS_WIDTH) -> np.ndarray:
    import cv2

    h, w = frame.shape[:2]
    if w <= width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)


# --------------------------------------------------------- boundary detection


WEAK_EDGE = 12.0
STRONG_EDGE = 34.0
WEAK_COVERAGE = 0.85
STRONG_COVERAGE = 0.50
FULL_COVERAGE = 0.78  # strong edge across this much of the span is a cut on its own
MAX_CONTINUITY = 0.20  # structure crossing the seam ⇒ same camera, not a seam
PEAK_PROMINENCE = 1.5


def edge_coverage_profiles(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-row |Sobel_y| strong-edge coverage and per-column |Sobel_x| coverage."""
    import cv2

    g = gray.astype(np.float32)
    sy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) * 0.25
    sx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) * 0.25
    row_cov = np.mean(sy > EDGE_THRESHOLD, axis=1)
    col_cov = np.mean(sx > EDGE_THRESHOLD, axis=0)
    return row_cov.astype(np.float32), col_cov.astype(np.float32)


def seam_profile(gray: np.ndarray, *, axis: int = 0) -> dict[str, np.ndarray]:
    """Per-line seam evidence along ``axis`` (0 → row seams, 1 → column seams).

    Returns weak / strong edge coverage (2-line max-pooled so a seam that
    straddles two thumbnail lines is not halved), the mean gradient, and the
    *continuity* of perpendicular structure across the line: the correlation
    between the |gradient| profiles a few lines above and below. Real camera
    content (robots, walls, field elements) is 3-D and crosses straight
    lines; a composition seam separates two unrelated images, so nothing
    crosses it.
    """
    import cv2

    g = gray.astype(np.float32)
    if axis == 1:
        g = np.ascontiguousarray(g.T)
    n = g.shape[0]
    along = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)) * 0.25  # gradient across lines
    perp = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)) * 0.25  # gradient along lines
    pooled = along.copy()
    pooled[:-1] = np.maximum(along[:-1], along[1:])
    weak = np.mean(pooled > WEAK_EDGE, axis=1)
    strong = np.mean(pooled > STRONG_EDGE, axis=1)
    mean = along.mean(axis=1)
    cont = np.ones(n, dtype=np.float32)
    gap, band = 2, 6
    for y in range(gap + band, n - gap - band - 1):
        a = perp[y - gap - band : y - gap].mean(axis=0)
        b = perp[y + gap + 2 : y + gap + 2 + band].mean(axis=0)
        sa, sb = float(a.std()), float(b.std())
        if sa < 1e-6 or sb < 1e-6:
            cont[y] = 0.0
            continue
        cont[y] = float(np.corrcoef(a, b)[0, 1])
    return {"weak": weak.astype(np.float32), "strong": strong.astype(np.float32), "mean": mean.astype(np.float32), "continuity": cont}


def seam_candidates(profile: dict[str, np.ndarray], *, margin: int, min_gap: int) -> list[int]:
    """Line indices that pass the seam rule, merged within ``min_gap``."""
    weak, strong, mean, cont = profile["weak"], profile["strong"], profile["mean"], profile["continuity"]
    n = len(weak)
    cands: list[int] = []
    for y in range(margin, n - margin):
        lo, hi = max(0, y - 14), min(n, y + 15)
        ring = np.concatenate([mean[lo : max(lo, y - 5)], mean[min(hi, y + 6) : hi]])
        base = float(np.median(ring)) if ring.size else 0.0
        prominent = mean[y] >= PEAK_PROMINENCE * max(base, 1.0)
        full = strong[y] >= FULL_COVERAGE and cont[y] <= MAX_CONTINUITY + 0.05
        seam = weak[y] >= WEAK_COVERAGE and strong[y] >= STRONG_COVERAGE and prominent and cont[y] <= MAX_CONTINUITY
        if full or seam:
            cands.append(y)
    if not cands:
        return []
    groups: list[list[int]] = [[cands[0]]]
    for i in cands[1:]:
        if i - groups[-1][-1] <= min_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    return [int(max(g, key=lambda i: mean[i])) for g in groups]


def _peaks(profile: np.ndarray, threshold: float, *, min_gap: int, margin: int) -> list[int]:
    """Indices of local maxima above threshold, merged within min_gap."""
    n = len(profile)
    cands = [i for i in range(margin, n - margin) if profile[i] >= threshold]
    if not cands:
        return []
    groups: list[list[int]] = [[cands[0]]]
    for i in cands[1:]:
        if i - groups[-1][-1] <= min_gap:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        best = max(g, key=lambda i: profile[i])
        out.append(int(best))
    return out


def _dark_runs(max_profile: np.ndarray, level: float) -> list[tuple[int, int]]:
    dark = max_profile < level
    runs: list[tuple[int, int]] = []
    start = None
    for i, d in enumerate(dark):
        if d and start is None:
            start = i
        elif not d and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(dark)))
    return runs


def detect_boundaries(
    frames: Sequence[np.ndarray],
) -> tuple[PaneBox, list[float], list[float], dict[str, Any]]:
    """Letterbox box + normalized row/column cuts from a stack of thumbnails."""
    import cv2

    grays = [cv2.cvtColor(_thumb(f), cv2.COLOR_BGR2GRAY) if f.ndim == 3 else _thumb(f) for f in frames]
    h, w = grays[0].shape[:2]
    grays = [g for g in grays if g.shape[:2] == (h, w)]
    stack = np.stack(grays).astype(np.float32)
    tmax = stack.max(axis=0)

    # Letterbox / pillarbox: dark bars touching the frame edges in every frame.
    row_max = tmax.max(axis=1)
    col_max = tmax.max(axis=0)
    y0, y1, x0, x1 = 0, h, 0, w
    for a, b in _dark_runs(row_max, DARK_LEVEL):
        if a == 0:
            y0 = max(y0, b)
        if b == h:
            y1 = min(y1, a)
    for a, b in _dark_runs(col_max, DARK_LEVEL):
        if a == 0:
            x0 = max(x0, b)
        if b == w:
            x1 = min(x1, a)
    if y1 - y0 < h * 0.4 or x1 - x0 < w * 0.4:
        y0, y1, x0, x1 = 0, h, 0, w
    letterbox = PaneBox(x0 / w, y0 / h, x1 / w, y1 / h)

    inner = stack[:, y0:y1, x0:x1]
    ih, iw = inner.shape[1:]
    # The temporal median wipes robots / people; seams and static structure stay.
    median = np.median(inner, axis=0) if inner.shape[0] > 1 else inner[0]

    # Interior dark gutters (black bars between panes) also count as cuts.
    inner_row_max = inner.max(axis=0).max(axis=1)
    inner_col_max = inner.max(axis=0).max(axis=0)
    gutter_rows = [(a + b) // 2 for a, b in _dark_runs(inner_row_max, DARK_LEVEL) if 0 < a and b < ih and (b - a) >= 2]
    gutter_cols = [(a + b) // 2 for a, b in _dark_runs(inner_col_max, DARK_LEVEL) if 0 < a and b < iw and (b - a) >= 2]

    margin_r = max(3, int(ih * MIN_PANE_FRAC))
    margin_c = max(3, int(iw * MIN_PANE_FRAC))
    row_prof = seam_profile(median, axis=0)
    row_cuts_px = seam_candidates(row_prof, margin=margin_r, min_gap=6)
    row_cuts_px = _confirm_per_frame(row_cuts_px, inner, axis=0)
    row_cuts_px = _reject_crossing(row_cuts_px, inner, axis=0)
    row_cuts_px += [r for r in gutter_rows if margin_r <= r < ih - margin_r]
    row_cuts_px = _merge_close(sorted(set(row_cuts_px)), min_gap=max(6, int(ih * MIN_PANE_FRAC)))

    # Column cuts are evaluated per row-band so a split lower pane does not
    # need the edge to run through the (unsplit) upper pane.
    bands = [0] + row_cuts_px + [ih]
    col_cuts_px: list[int] = []
    col_by_band: dict[tuple[int, int], list[int]] = {}
    for a, b in zip(bands[:-1], bands[1:]):
        if b - a < margin_r:
            continue
        band_prof = seam_profile(median[a:b], axis=1)
        cuts = seam_candidates(band_prof, margin=margin_c, min_gap=6)
        cuts = _confirm_per_frame(cuts, inner[:, a:b], axis=1)
        cuts = _reject_crossing(cuts, inner[:, a:b], axis=1)
        band_dark = inner[:, a:b].max(axis=0).max(axis=0)
        cuts += [(c0 + c1) // 2 for c0, c1 in _dark_runs(band_dark, DARK_LEVEL) if margin_c <= c0 and c1 < iw - margin_c and (c1 - c0) >= 2]
        cuts = _merge_close(sorted(set(cuts)), min_gap=max(6, int(iw * MIN_PANE_FRAC)))
        col_by_band[(a, b)] = cuts
        col_cuts_px.extend(cuts)
    col_cuts_px += [c for c in gutter_cols if margin_c <= c < iw - margin_c]
    col_cuts_px = _merge_close(sorted(set(col_cuts_px)), min_gap=max(6, int(iw * MIN_PANE_FRAC)))

    # Normalize cuts into full-frame coordinates.
    row_cuts = [letterbox.y0 + (r / ih) * letterbox.h for r in row_cuts_px]
    col_cuts = [letterbox.x0 + (c / iw) * letterbox.w for c in col_cuts_px]
    debug = {
        "row_profile": row_prof["weak"],
        "row_strength": [float(min(1.0, row_prof["weak"][r])) for r in row_cuts_px if r < ih],
        "col_by_band": {f"{letterbox.y0 + a / ih * letterbox.h:.3f}-{letterbox.y0 + b / ih * letterbox.h:.3f}": [letterbox.x0 + c / iw * letterbox.w for c in cuts] for (a, b), cuts in col_by_band.items()},
        "thumb_size": (w, h),
    }
    return letterbox, row_cuts, col_cuts, debug


MAX_CROSSING = 0.08
MIN_FRAMES_FOR_CROSSING = 6


def temporal_crossing(stack: np.ndarray, line: int, *, axis: int = 0, gaps: Sequence[int] = (2, 3)) -> float:
    """Do pixels just before / after ``line`` change *together* over time?

    ``stack`` is T×H×W gray. For every column we correlate the time series of
    the pixel ``gap`` lines above with the one ``gap`` lines below (only where
    something actually changes). Inside one camera, robots, people and
    lighting cross a straight line so the two sides co-vary (≥ 0.1). Across a
    composition seam the sides come from different sensors, so the
    correlation is ~0. Returns the max over ``gaps`` (worst case).
    """
    if stack.ndim != 3 or stack.shape[0] < 3:
        return 0.0
    arr = stack if axis == 0 else np.transpose(stack, (0, 2, 1))
    n = arr.shape[1]
    worst = 0.0
    for gap in gaps:
        lo, hi = line - gap, line + gap
        if lo < 0 or hi >= n:
            continue
        a = arr[:, lo, :]
        b = arr[:, hi, :]
        sa, sb = a.std(axis=0), b.std(axis=0)
        sel = (sa > 3.0) | (sb > 3.0)
        if int(sel.sum()) < 10:
            continue
        a = a[:, sel]
        b = b[:, sel]
        a = (a - a.mean(axis=0)) / (a.std(axis=0) + 1e-6)
        b = (b - b.mean(axis=0)) / (b.std(axis=0) + 1e-6)
        corr = float(np.mean((a * b).mean(axis=0)))
        worst = max(worst, corr)
    return worst


def _reject_crossing(cuts: list[int], inner: np.ndarray, *, axis: int) -> list[int]:
    if inner.shape[0] < MIN_FRAMES_FOR_CROSSING:
        return cuts
    return [c for c in cuts if temporal_crossing(inner, c, axis=axis) <= MAX_CROSSING]


def _confirm_per_frame(cuts: list[int], inner: np.ndarray, *, axis: int, min_fraction: float = 0.6) -> list[int]:
    """Keep cuts whose weak-edge coverage is present in most individual frames.

    A seam is a property of the composition, so it is in every frame; a
    structure that only shows in the temporal median (e.g. a wall line that
    robots keep crossing) is not.
    """
    if not cuts or inner.shape[0] < 2:
        return cuts
    kept: list[int] = []
    n_frames = inner.shape[0]
    idx = np.linspace(0, n_frames - 1, min(n_frames, 9)).astype(int)
    for c in cuts:
        hits = 0
        for i in idx:
            prof = seam_profile(inner[i], axis=axis)
            lo, hi = max(0, c - 1), min(len(prof["weak"]), c + 2)
            if float(prof["weak"][lo:hi].max()) >= WEAK_COVERAGE - 0.1:
                hits += 1
        if hits >= min_fraction * len(idx):
            kept.append(c)
    return kept


def _merge_close(values: list[int], *, min_gap: int) -> list[int]:
    out: list[int] = []
    for v in values:
        if out and v - out[-1] < min_gap:
            out[-1] = (out[-1] + v) // 2
        else:
            out.append(v)
    return out


def boxes_from_cuts(letterbox: PaneBox, row_cuts: Sequence[float], col_by_band: dict[str, list[float]] | None, col_cuts: Sequence[float]) -> list[PaneBox]:
    """Rectangular panes from row cuts and (per-band) column cuts."""
    ys = [letterbox.y0] + sorted(row_cuts) + [letterbox.y1]
    boxes: list[PaneBox] = []
    for a, b in zip(ys[:-1], ys[1:]):
        if b - a < MIN_PANE_FRAC * 0.5:
            continue
        band_cols: list[float] | None = None
        if col_by_band:
            for key, cuts in col_by_band.items():
                ka, kb = [float(v) for v in key.split("-")]
                if abs(ka - a) < 0.02 and abs(kb - b) < 0.02:
                    band_cols = cuts
                    break
        cols = band_cols if band_cols is not None else list(col_cuts)
        xs = [letterbox.x0] + sorted(cols) + [letterbox.x1]
        for c0, c1 in zip(xs[:-1], xs[1:]):
            if c1 - c0 < MIN_PANE_FRAC * 0.5:
                continue
            boxes.append(PaneBox(c0, a, c1, b))
    return boxes


# ------------------------------------------------------- content features


def _hsv_masks(bgr: np.ndarray) -> dict[str, np.ndarray]:
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    grey = (s < 60) & (v > 45) & (v < 210)
    red = ((hch <= 10) | (hch >= 165)) & (s > 110) & (v > 70)
    blue = (hch >= 95) & (hch <= 130) & (s > 110) & (v > 60)
    return {"grey": grey, "red": red, "blue": blue}


def pane_features(
    median_bgr: np.ndarray,
    std_map: np.ndarray,
    box: PaneBox,
    *,
    frame_area: float,
) -> dict[str, float]:
    """Content descriptors for one pane from the temporal median + std maps."""
    import cv2

    crop = box.slice(median_bgr)
    std_crop = box.slice(std_map)
    if crop.size == 0 or std_crop.size == 0:
        return {"area": 0.0}
    masks = _hsv_masks(crop)
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    )
    edge_density = float(np.mean(edges > 120.0))
    live = std_crop > LIVE_STD
    live_frac = float(np.mean(live))
    static_frac = float(np.mean(std_crop < STATIC_STD))
    global_change = float(np.median(std_crop))
    # Moving blobs: how many separate moving things (robots → several small ones).
    live_u8 = (live.astype(np.uint8) * 255)
    live_u8 = cv2.morphologyEx(live_u8, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n_blobs, _labels, stats, _cent = cv2.connectedComponentsWithStats(live_u8)
    blob_areas = stats[1:, cv2.CC_STAT_AREA] / float(max(h * w, 1)) if n_blobs > 1 else np.zeros(0)
    small_blobs = int(np.sum((blob_areas > 0.0004) & (blob_areas < 0.03)))
    big_blob = float(blob_areas.max()) if blob_areas.size else 0.0
    # Where is the grey floor? Overview cameras put it in the middle/bottom band.
    grey = masks["grey"]
    grey_rows = grey.mean(axis=1)
    mid = float(np.mean(grey_rows[int(h * 0.3) : int(h * 0.9)])) if h > 4 else 0.0
    return {
        "area": float(box.area),
        "aspect": float(box.w / max(box.h, 1e-6)),
        "carpet": float(np.mean(grey)),
        "carpet_mid": mid,
        "red": float(np.mean(masks["red"])),
        "blue": float(np.mean(masks["blue"])),
        "edge_density": edge_density,
        "live": live_frac,
        "static": static_frac,
        "global_change": global_change,
        "small_blobs": float(small_blobs),
        "big_blob": big_blob,
        "brightness": float(np.mean(gray)) / 255.0,
    }


def overview_score(f: dict[str, float]) -> float:
    """How much does this pane look like a wide, static field camera?"""
    if not f or f.get("area", 0.0) <= 0:
        return -1.0
    score = 0.0
    score += 2.2 * min(f.get("carpet", 0.0), 0.6) / 0.6
    score += 0.8 * min(f.get("carpet_mid", 0.0), 0.6) / 0.6
    score += 1.0 * min(f.get("area", 0.0), 0.6) / 0.6
    # Static camera: median temporal std small.
    gc = f.get("global_change", 0.0)
    score += 1.0 * float(np.clip(1.0 - gc / 18.0, 0.0, 1.0))
    live = f.get("live", 0.0)
    if 0.004 <= live <= 0.25:
        score += 0.6
    elif live > 0.4:
        score -= 1.0
    score += 0.15 * min(f.get("small_blobs", 0.0), 6.0)
    if f.get("big_blob", 0.0) > 0.12:
        score -= 0.8
    # Graphics panes are static *and* edge dense with no carpet.
    if f.get("static", 0.0) > 0.85 and f.get("carpet", 0.0) < 0.1:
        score -= 2.0
    # Both alliance colours present (tape, driver stations) is a field cue.
    if f.get("red", 0.0) > 0.002 and f.get("blue", 0.0) > 0.002:
        score += 0.4
    return float(score)


def _is_graphics(f: dict[str, float]) -> bool:
    return f.get("static", 0.0) > 0.8 and f.get("live", 0.0) < 0.05


def _bug_overlap(box: PaneBox, scorebug: list[float] | None) -> float:
    """Fraction of the scorebug area that lies inside ``box``."""
    if scorebug is None:
        return 0.0
    bx0, by0, bx1, by1 = scorebug
    ix = max(0.0, min(bx1, box.x1) - max(bx0, box.x0))
    iy = max(0.0, min(by1, box.y1) - max(by0, box.y0))
    return float(ix * iy / max((bx1 - bx0) * (by1 - by0), 1e-6))


def _classify_panes(
    boxes: list[PaneBox],
    median_bgr: np.ndarray,
    std_map: np.ndarray,
    scorebug: list[float] | None,
) -> list[DetectedPane]:
    frame_area = 1.0
    feats = [pane_features(median_bgr, std_map, b, frame_area=frame_area) for b in boxes]
    scores = []
    for b, f in zip(boxes, feats):
        sc = overview_score(f)
        if _is_graphics(f):
            sc = -5.0
        # Directors put the primary field camera on top; a scorebug burned
        # into a pane costs tracking area and OCR noise.
        sc += 0.35 * (1.0 - float(np.clip(b.y0 / 0.6, 0.0, 1.0)))
        sc -= 0.6 * _bug_overlap(b, scorebug)
        f["overview_score"] = sc
        scores.append(sc)
    panes: list[DetectedPane] = []
    order = sorted(range(len(boxes)), key=lambda i: scores[i], reverse=True)
    assigned: dict[int, tuple[str, float]] = {}
    if order:
        best = order[0]
        if scores[best] >= 2.0:
            conf = float(np.clip(0.45 + (scores[best] - 2.0) * 0.18, 0.45, 0.97))
            assigned[best] = (ROLE_OVERVIEW, conf)
            for i in order[1:]:
                if scores[i] >= 2.4 and feats[i].get("carpet", 0.0) >= 0.18:
                    assigned[i] = (ROLE_OVERVIEW_ALT, float(np.clip(0.4 + (scores[i] - 2.4) * 0.15, 0.4, 0.9)))
    for i, box in enumerate(boxes):
        f = feats[i]
        if i in assigned:
            role, conf = assigned[i]
            bias = None
        elif _is_graphics(f):
            role, conf, bias = ROLE_GRAPHICS, float(np.clip(0.5 + f["static"] * 0.4, 0.5, 0.95)), None
        elif f.get("live", 0.0) >= 0.02 or f.get("global_change", 0.0) > 6.0:
            red, blue = f.get("red", 0.0), f.get("blue", 0.0)
            if red > blue * 1.6 and red > 0.004:
                role, bias = ROLE_RED, "red"
            elif blue > red * 1.6 and blue > 0.004:
                role, bias = ROLE_BLUE, "blue"
            else:
                role, bias = ROLE_SIDELINE, None
            conf = float(np.clip(0.4 + min(f.get("live", 0.0), 0.3), 0.4, 0.85))
        else:
            role, conf, bias = ROLE_OTHER, 0.35, None
        bug_in = None
        if scorebug is not None:
            bx0, by0, bx1, by1 = scorebug
            ix = max(0.0, min(bx1, box.x1) - max(bx0, box.x0))
            iy = max(0.0, min(by1, box.y1) - max(by0, box.y0))
            if ix * iy > 0.5 * max((bx1 - bx0) * (by1 - by0), 1e-6):
                bug_in = [
                    round((bx0 - box.x0) / max(box.w, 1e-6), 4),
                    round((by0 - box.y0) / max(box.h, 1e-6), 4),
                    round((bx1 - box.x0) / max(box.w, 1e-6), 4),
                    round((by1 - box.y0) / max(box.h, 1e-6), 4),
                ]
        panes.append(DetectedPane(role=role, box=box, confidence=conf, features=f, alliance_bias=bias, scorebug=bug_in))
    # Two side cameras on the same band: fix left/right alliance by position when colour was ambiguous.
    sides = [p for p in panes if p.role in {ROLE_BLUE, ROLE_RED, ROLE_SIDELINE}]
    if len(sides) == 2 and all(p.role == ROLE_SIDELINE for p in sides):
        left, right = sorted(sides, key=lambda p: p.box.x0)
        left.role, left.alliance_bias = ROLE_BLUE, "blue"
        right.role, right.alliance_bias = ROLE_RED, "red"
        left.confidence = right.confidence = min(left.confidence, right.confidence, 0.55)
    return panes


# ------------------------------------------------------------ scorebug


def detect_scorebug(median_bgr: np.ndarray, std_map: np.ndarray) -> list[float] | None:
    """Largest static, edge-dense blob in the top / bottom 38 % of the frame."""
    import cv2

    h, w = std_map.shape[:2]
    gray = cv2.cvtColor(median_bgr, cv2.COLOR_BGR2GRAY)
    edges = np.abs(cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)) + np.abs(
        cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    )
    edge_density = cv2.blur((edges > 100.0).astype(np.float32), (15, 9))
    static = cv2.blur((std_map < STATIC_STD).astype(np.float32), (15, 9))
    cand = ((static > 0.75) & (edge_density > 0.12)).astype(np.uint8) * 255
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, np.ones((9, 25), np.uint8))
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(cand)
    best = None
    best_area = 0.0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        frac = (bw * bh) / float(h * w)
        cy = (y + bh * 0.5) / h
        if frac < 0.008 or frac > 0.3:
            continue
        if not (cy < 0.38 or cy > 0.62):
            continue
        if bw < w * 0.08 or bh < h * 0.03:
            continue
        if bw / max(bh, 1) < 1.2:
            continue
        if frac > best_area:
            best_area = frac
            best = [x / w, y / h, (x + bw) / w, (y + bh) / h]
    if best is None:
        return None
    return [round(float(v), 4) for v in best]


# ------------------------------------------------------------ per segment


def analyze_frames(frames: Sequence[np.ndarray]) -> FrameLayout:
    """Decompose a stack of frames (same layout) into labelled panes."""
    import cv2

    if not frames:
        return FrameLayout([], PaneBox(0, 0, 1, 1), [], [], None, "empty", 0.0, "No frames.")
    thumbs = [_thumb(f) for f in frames]
    h, w = thumbs[0].shape[:2]
    thumbs = [t for t in thumbs if t.shape[:2] == (h, w)]
    stack = np.stack(thumbs).astype(np.float32)
    median_bgr = np.median(stack, axis=0).astype(np.uint8)
    gray_stack = np.stack([cv2.cvtColor(t, cv2.COLOR_BGR2GRAY) for t in thumbs]).astype(np.float32)
    std_map = gray_stack.std(axis=0) if len(thumbs) > 1 else np.full((h, w), LIVE_STD + 1.0, dtype=np.float32)

    letterbox, row_cuts, col_cuts, debug = detect_boundaries(thumbs)
    boxes = boxes_from_cuts(letterbox, row_cuts, debug.get("col_by_band"), col_cuts)
    if not boxes:
        boxes = [letterbox]
    scorebug = detect_scorebug(median_bgr, std_map)
    panes = _classify_panes(boxes, median_bgr, std_map, scorebug)

    ov = next((p for p in panes if p.role == ROLE_OVERVIEW), None)
    strengths = debug.get("row_strength") or []
    cut_conf = float(np.mean(strengths)) if strengths else 1.0
    confidence = float(np.clip(0.5 * cut_conf + 0.5 * (ov.confidence if ov else 0.3), 0.05, 0.98))
    signature = _signature(letterbox, row_cuts, col_cuts)
    roles = ", ".join(p.role for p in panes)
    if len(boxes) == 1:
        detail = f"Single-pane broadcast ({roles})."
    else:
        detail = f"{len(boxes)} panes ({roles}); {len(row_cuts)} row cut(s), {len(col_cuts)} column cut(s)."
    if scorebug:
        detail += " Scorebug detected."
    if ov is None:
        detail += " No wide field camera in this layout — tracking pauses here."
    return FrameLayout(panes, letterbox, row_cuts, col_cuts, scorebug, signature, confidence, detail)


def _signature(letterbox: PaneBox, row_cuts: Sequence[float], col_cuts: Sequence[float]) -> str:
    parts = ["L" + ",".join(f"{v:.3f}" for v in (letterbox.x0, letterbox.y0, letterbox.x1, letterbox.y1))]
    parts.append("R" + ",".join(f"{v:.3f}" for v in sorted(row_cuts)))
    parts.append("C" + ",".join(f"{v:.3f}" for v in sorted(col_cuts)))
    return "|".join(parts)


def _parse_signature(sig: str) -> tuple[list[float], list[float], list[float]]:
    out: list[list[float]] = []
    for part in sig.split("|"):
        body = part[1:]
        out.append([float(v) for v in body.split(",") if v])
    while len(out) < 3:
        out.append([])
    return out[0], out[1], out[2]


def same_signature(a: str, b: str, tol: float = 0.04) -> bool:
    """Two layouts are the same when every cut agrees within ``tol`` (normalized)."""
    if a == b:
        return True
    try:
        la, ra, ca = _parse_signature(a)
        lb, rb, cb = _parse_signature(b)
    except Exception:  # noqa: BLE001
        return False
    for xs, ys in ((la, lb), (ra, rb), (ca, cb)):
        if len(xs) != len(ys):
            return False
        if any(abs(x - y) > tol for x, y in zip(xs, ys)):
            return False
    return True


def frame_signature(frames: np.ndarray | Sequence[np.ndarray]) -> str:
    """Cheap layout signature (boundaries only) for change detection."""
    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    letterbox, row_cuts, col_cuts, _debug = detect_boundaries(list(frames))
    return _signature(letterbox, row_cuts, col_cuts)


# ---------------------------------------------------------------- timeline


def analyze_video_layout(
    video_path: Path | str,
    *,
    step_s: float = 2.0,
    min_segment_s: float = 4.0,
    max_samples: int = 140,
    on_progress: ProgressFn | None = None,
) -> LayoutTimeline:
    """Segment a match VOD into layout runs and decompose each one."""
    probe, meta = sample_frames(video_path, [0.0], width=ANALYSIS_WIDTH)
    duration = float(meta.get("duration") or 0.0)
    fps = float(meta.get("fps") or 30.0)
    if duration <= 0:
        duration = 180.0
    step = max(step_s, duration / max_samples)
    times = list(np.arange(0.0, max(duration - 0.05, step), step))
    if on_progress:
        on_progress("Sampling broadcast frames for layout…", 5.0)
    frames, meta = sample_frames(video_path, times, width=ANALYSIS_WIDTH)
    fw, fh = meta.get("frame_size") or (0, 0)
    if not frames:
        empty = FrameLayout([], PaneBox(0, 0, 1, 1), [], [], None, "empty", 0.0, "No frames readable.")
        return LayoutTimeline([LayoutSegment(0.0, duration, empty)], duration, fps, (fw, fh))

    # 1) windowed signatures (5 neighbouring samples) → runs. A single frame
    #    cannot tell a seam from a straight wall; five frames two seconds
    #    apart usually can.
    sigs: list[str] = []
    half = 2
    for i in range(len(frames)):
        lo, hi = max(0, i - half), min(len(frames), i + half + 1)
        sigs.append(frame_signature([f for _t, f in frames[lo:hi]]))
    if on_progress:
        on_progress("Grouping layout runs…", 40.0)
    runs: list[list[int]] = [[0]]
    for i in range(1, len(sigs)):
        if same_signature(sigs[i], sigs[runs[-1][0]]):
            runs[-1].append(i)
        else:
            runs.append([i])
    # Smooth: absorb 1-sample blips into the neighbour they match.
    runs = _absorb_blips(runs, sigs)

    segments: list[LayoutSegment] = []
    for k, idxs in enumerate(runs):
        t0 = frames[idxs[0]][0]
        t1 = frames[idxs[-1] + 1][0] if idxs[-1] + 1 < len(frames) else duration
        layout = analyze_frames([frames[i][1] for i in idxs])
        segments.append(LayoutSegment(t0=t0, t1=t1, layout=layout, sample_count=len(idxs)))
        if on_progress:
            on_progress("Decomposing layout segments…", 40.0 + 55.0 * (k + 1) / max(len(runs), 1))

    # 2) merge adjacent segments whose decomposition agrees (same signature & roles)
    merged: list[LayoutSegment] = []
    for seg in segments:
        if merged and same_signature(merged[-1].layout.signature, seg.layout.signature):
            prev = merged[-1]
            keep = prev.layout if prev.sample_count >= seg.sample_count else seg.layout
            merged[-1] = LayoutSegment(prev.t0, seg.t1, keep, prev.sample_count + seg.sample_count)
        else:
            merged.append(seg)
    # 3) very short segments without an overview inherit nothing; keep them (they mark gaps)
    #    but drop sub-minimum blips *with* the same overview as a neighbour.
    final: list[LayoutSegment] = []
    for seg in merged:
        if final and (seg.t1 - seg.t0) < min_segment_s:
            prev = final[-1]
            po, so = prev.layout.overview(), seg.layout.overview()
            if po is not None and so is not None and po.box.same_as(so.box):
                final[-1] = LayoutSegment(prev.t0, seg.t1, prev.layout, prev.sample_count + seg.sample_count)
                continue
            if seg.sample_count <= 1 and prev.sample_count >= 3:
                # one odd sample between real layouts — a transition frame
                final[-1] = LayoutSegment(prev.t0, seg.t1, prev.layout, prev.sample_count + seg.sample_count)
                continue
        final.append(seg)
    # A second pass merges neighbours whose decomposition now agrees.
    squeezed: list[LayoutSegment] = []
    for seg in final:
        if squeezed and same_signature(squeezed[-1].layout.signature, seg.layout.signature):
            prev = squeezed[-1]
            keep = prev.layout if prev.sample_count >= seg.sample_count else seg.layout
            squeezed[-1] = LayoutSegment(prev.t0, seg.t1, keep, prev.sample_count + seg.sample_count)
        else:
            squeezed.append(seg)
    final = squeezed
    if final:
        final[0] = LayoutSegment(0.0, final[0].t1, final[0].layout, final[0].sample_count)
        last = final[-1]
        final[-1] = LayoutSegment(last.t0, max(last.t1, duration), last.layout, last.sample_count)
    if on_progress:
        on_progress("Layout timeline ready.", 100.0)
    return LayoutTimeline(final, duration, fps, (int(fw), int(fh)))


def _absorb_blips(runs: list[list[int]], sigs: list[str]) -> list[list[int]]:
    if len(runs) < 3:
        return runs
    out: list[list[int]] = []
    i = 0
    while i < len(runs):
        cur = runs[i]
        if len(cur) == 1 and out and i + 1 < len(runs) and same_signature(sigs[out[-1][0]], sigs[runs[i + 1][0]]):
            # a single odd frame between two identical runs → merge all three
            out[-1] = out[-1] + cur + runs[i + 1]
            i += 2
            continue
        out.append(cur)
        i += 1
    return out


# ------------------------------------------------------------- rendering


def render_decomposition(frame: np.ndarray, layout: FrameLayout, *, title: str = "") -> np.ndarray:
    """Draw pane boxes, roles and confidences onto a copy of the frame."""
    import cv2

    colors = {
        ROLE_OVERVIEW: (60, 220, 60),
        ROLE_OVERVIEW_ALT: (60, 200, 200),
        ROLE_BLUE: (255, 140, 40),
        ROLE_RED: (60, 60, 255),
        ROLE_SIDELINE: (200, 200, 60),
        ROLE_GRAPHICS: (160, 160, 160),
        ROLE_SCOREBUG: (255, 0, 255),
        ROLE_OTHER: (120, 120, 120),
    }
    out = frame.copy()
    h, w = out.shape[:2]
    for pane in layout.panes:
        x0, y0, x1, y1 = pane.box.pixels(w, h)
        color = colors.get(pane.role, (200, 200, 200))
        cv2.rectangle(out, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), color, 3)
        label = f"{pane.role} {pane.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(out, (x0 + 6, y0 + 6), (x0 + 14 + tw, y0 + 16 + th), (0, 0, 0), -1)
        cv2.putText(out, label, (x0 + 10, y0 + 10 + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if layout.scorebug:
        bx0, by0, bx1, by1 = layout.scorebug
        cv2.rectangle(out, (int(bx0 * w), int(by0 * h)), (int(bx1 * w), int(by1 * h)), colors[ROLE_SCOREBUG], 2)
        cv2.putText(out, "scorebug", (int(bx0 * w) + 4, int(by0 * h) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[ROLE_SCOREBUG], 2)
    if title:
        cv2.putText(out, title, (12, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return out
