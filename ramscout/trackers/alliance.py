"""Robust red / blue alliance classification for FRC broadcast crops.

Why the old approach failed: a global HSV threshold over the *whole* box picks
up jersey colors, carpet tape, LED wash, and glare, and it moves with the
broadcast white balance. Here we:

1. Sample only the **lower band** of the box, where bumpers live.
2. Learn the match's own red/blue **in CIE Lab chroma** (a*, b*) from saturated
   bumper pixels of confirmed, moving robots (self-supervised 2-means), then
   classify by nearest centroid with a margin. White balance shifts move both
   centroids together, so the decision stays put.
3. Vote per track over a sliding window with hysteresis so labels do not flap.
4. Solve a tiny assignment so a match ends with 3 red + 3 blue.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

RED = "red"
BLUE = "blue"
UNKNOWN = "unknown"

# Hue bands (OpenCV 0–180) wide enough to survive white-balance drift but still
# excluding yellow / green game pieces and carpet.
_RED_HUES = ((0, 22), (150, 180))
_BLUE_HUES = ((85, 140),)


def bumper_band(frame_bgr: np.ndarray, bbox: Sequence[float], frac: float = 0.42) -> np.ndarray | None:
    """Lower ``frac`` of the box (crop coordinates) where bumpers sit."""
    if frame_bgr is None or frame_bgr.size == 0:
        return None
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bh = max(y2 - y1, 1.0)
    band_top = y2 - bh * float(np.clip(frac, 0.15, 1.0))
    ix1 = int(np.clip(np.floor(x1), 0, w - 1))
    ix2 = int(np.clip(np.ceil(x2), ix1 + 1, w))
    iy1 = int(np.clip(np.floor(band_top), 0, h - 1))
    iy2 = int(np.clip(np.ceil(y2), iy1 + 1, h))
    if ix2 - ix1 < 2 or iy2 - iy1 < 2:
        return None
    return frame_bgr[iy1:iy2, ix1:ix2]


@dataclass
class ChromaFeature:
    a: float
    b: float
    saturated_fraction: float
    red_votes: int
    blue_votes: int
    n_pixels: int

    def as_point(self) -> np.ndarray:
        return np.array([self.a, self.b], dtype=np.float64)


def chroma_features(roi_bgr: np.ndarray, *, min_sat: int = 70, min_val: int = 50) -> ChromaFeature | None:
    """Mean Lab chroma of saturated pixels + HSV red/blue counts for fallback."""
    import cv2

    if roi_bgr is None or roi_bgr.size == 0 or roi_bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    sat_mask = (hsv[..., 1] >= min_sat) & (hsv[..., 2] >= min_val)
    n_total = int(sat_mask.size)
    n_sat = int(np.count_nonzero(sat_mask))
    hue = hsv[..., 0]
    red_mask = np.zeros_like(sat_mask)
    for lo, hi in _RED_HUES:
        red_mask |= (hue >= lo) & (hue <= hi)
    blue_mask = np.zeros_like(sat_mask)
    for lo, hi in _BLUE_HUES:
        blue_mask |= (hue >= lo) & (hue <= hi)
    red_votes = int(np.count_nonzero(red_mask & sat_mask))
    blue_votes = int(np.count_nonzero(blue_mask & sat_mask))
    if n_sat < 6:
        return ChromaFeature(0.0, 0.0, 0.0, red_votes, blue_votes, n_sat)

    # Only red-ish / blue-ish saturated pixels feed the chroma estimate so a
    # yellow game piece riding on the robot does not drag the centroid.
    bumperish = sat_mask & (red_mask | blue_mask)
    if np.count_nonzero(bumperish) < 6:
        bumperish = sat_mask
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    a = float(np.mean(lab[..., 1][bumperish])) - 128.0
    b = float(np.mean(lab[..., 2][bumperish])) - 128.0
    return ChromaFeature(a, b, n_sat / max(n_total, 1), red_votes, blue_votes, n_sat)


def hsv_alliance(feature: ChromaFeature | None, *, min_pixels: int = 12) -> tuple[str, float]:
    """Global-threshold fallback: which hue family dominates the band."""
    if feature is None:
        return UNKNOWN, 0.0
    r, b = feature.red_votes, feature.blue_votes
    if r + b < min_pixels:
        return UNKNOWN, 0.0
    share = abs(r - b) / float(r + b)
    if share < 0.2:
        return UNKNOWN, 0.0
    conf = float(np.clip(0.5 + share * 0.5, 0.5, 0.95))
    return (RED, conf) if r > b else (BLUE, conf)


def _two_means(points: np.ndarray, iterations: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic 2-means on (a, b) chroma; returns centroids, labels."""
    score = points[:, 0] + points[:, 1]  # red ≫ blue along a*+b*
    centroids = np.stack([points[int(np.argmax(score))], points[int(np.argmin(score))]])
    labels = np.zeros(len(points), dtype=int)
    for it in range(iterations):
        d = np.linalg.norm(points[:, None, :] - centroids[None, :, :], axis=2)
        new_labels = np.argmin(d, axis=1)
        if it > 0 and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(2):
            members = points[labels == k]
            if len(members):
                centroids[k] = members.mean(axis=0)
    return centroids, labels


@dataclass
class AllianceCalibrator:
    """Per-match adaptive red/blue centroids in Lab chroma."""

    min_samples: int = 24
    max_samples: int = 400
    min_separation: float = 14.0
    min_cluster_share: float = 0.2
    _points: list[np.ndarray] = field(default_factory=list)
    red_centroid: np.ndarray | None = None
    blue_centroid: np.ndarray | None = None
    separation: float = 0.0
    fits: int = 0

    @property
    def calibrated(self) -> bool:
        return self.red_centroid is not None and self.blue_centroid is not None

    @property
    def sample_count(self) -> int:
        return len(self._points)

    def add(self, feature: ChromaFeature | None) -> bool:
        """Add a bumper-band feature from a confirmed moving robot."""
        if feature is None or feature.n_pixels < 12:
            return False
        if feature.red_votes + feature.blue_votes < 8:
            return False
        if len(self._points) >= self.max_samples:
            self._points.pop(0)
        self._points.append(feature.as_point())
        return True

    def fit(self) -> bool:
        if len(self._points) < self.min_samples:
            return self.calibrated
        pts = np.stack(self._points)
        centroids, labels = _two_means(pts)
        share = np.bincount(labels, minlength=2) / float(len(labels))
        sep = float(np.linalg.norm(centroids[0] - centroids[1]))
        if share.min() < self.min_cluster_share or sep < self.min_separation:
            return self.calibrated
        # Red is the centroid with the larger a*+b* (warm), blue the smaller.
        score = centroids[:, 0] + centroids[:, 1]
        red_idx = int(np.argmax(score))
        self.red_centroid = centroids[red_idx].copy()
        self.blue_centroid = centroids[1 - red_idx].copy()
        self.separation = sep
        self.fits += 1
        return True

    def red_probability(self, feature: ChromaFeature | None) -> float | None:
        """P(red) from nearest calibrated centroid; None when not calibrated."""
        if not self.calibrated or feature is None or feature.n_pixels < 12:
            return None
        p = feature.as_point()
        d_red = float(np.linalg.norm(p - self.red_centroid))
        d_blue = float(np.linalg.norm(p - self.blue_centroid))
        tau = max(self.separation / 4.0, 1e-3)
        return float(1.0 / (1.0 + np.exp(-(d_blue - d_red) / tau)))

    def classify(self, roi_or_feature: np.ndarray | ChromaFeature | None) -> tuple[str, float, str]:
        """→ (label, confidence, method). Falls back to HSV thresholds."""
        feature = roi_or_feature
        if isinstance(roi_or_feature, np.ndarray):
            feature = chroma_features(roi_or_feature)
        if feature is None:
            return UNKNOWN, 0.0, "none"
        p_red = self.red_probability(feature)
        if p_red is not None:
            margin = abs(p_red - 0.5)
            if margin >= 0.12:
                label = RED if p_red > 0.5 else BLUE
                return label, float(np.clip(0.5 + margin, 0.5, 0.99)), "calibrated"
        label, conf = hsv_alliance(feature)
        return label, conf, "hsv"

    def as_dict(self) -> dict[str, Any]:
        return {
            "calibrated": self.calibrated,
            "samples": self.sample_count,
            "fits": self.fits,
            "separation": round(self.separation, 2),
            "red_ab": [round(float(v), 2) for v in self.red_centroid] if self.calibrated else None,
            "blue_ab": [round(float(v), 2) for v in self.blue_centroid] if self.calibrated else None,
        }


class AllianceVoter:
    """Sliding-window weighted majority with hysteresis per track."""

    def __init__(self, window: int = 15, flip_ratio: float = 0.65, min_votes: int = 3) -> None:
        self.window = int(max(window, 1))
        self.flip_ratio = float(np.clip(flip_ratio, 0.5, 1.0))
        self.min_votes = int(max(min_votes, 1))
        self._votes: dict[int, deque[tuple[str, float]]] = {}
        self._prior: dict[int, tuple[str, float]] = {}
        self._current: dict[int, str] = {}

    def reset(self) -> None:
        self._votes.clear()
        self._prior.clear()
        self._current.clear()

    def set_prior(self, track_id: int, label: str, weight: float) -> None:
        """Starting-side prior: acts like a sticky pseudo-vote."""
        if label in {RED, BLUE} and weight > 0:
            self._prior[int(track_id)] = (label, float(weight))

    def vote(self, track_id: int, label: str, confidence: float) -> None:
        if label not in {RED, BLUE}:
            return
        q = self._votes.setdefault(int(track_id), deque(maxlen=self.window))
        q.append((label, float(np.clip(confidence, 0.05, 1.0))))

    def tallies(self, track_id: int) -> tuple[float, float, int]:
        red_w = blue_w = 0.0
        q = self._votes.get(int(track_id))
        n = 0
        if q:
            for label, w in q:
                n += 1
                if label == RED:
                    red_w += w
                else:
                    blue_w += w
        prior = self._prior.get(int(track_id))
        if prior:
            if prior[0] == RED:
                red_w += prior[1]
            else:
                blue_w += prior[1]
        return red_w, blue_w, n

    def label(self, track_id: int) -> tuple[str, float]:
        """→ (label, confidence) with hysteresis against flips."""
        red_w, blue_w, n = self.tallies(track_id)
        total = red_w + blue_w
        if total <= 0:
            return UNKNOWN, 0.0
        leader = RED if red_w >= blue_w else BLUE
        share = max(red_w, blue_w) / total
        current = self._current.get(int(track_id))
        if current is None:
            self._current[int(track_id)] = leader
            return leader, float(share)
        if leader != current:
            if n >= self.min_votes and share >= self.flip_ratio:
                self._current[int(track_id)] = leader
                return leader, float(share)
            return current, float(1.0 - share)
        return current, float(share)

    def red_probability(self, track_id: int) -> float | None:
        red_w, blue_w, _n = self.tallies(track_id)
        total = red_w + blue_w
        if total <= 0:
            return None
        return float(red_w / total)


def side_prior(
    x_in: float,
    t: float,
    *,
    field_length: float,
    alliance_depth: float,
    early_s: float = 30.0,
) -> tuple[str, float]:
    """Starting-zone prior: robots begin in their alliance zone.

    Only meaningful early in the video; strength grows with distance into the
    alliance zone. Blue is x≈0, red is x≈field_length by convention.
    """
    if t > early_s or not np.isfinite(x_in):
        return UNKNOWN, 0.0
    mid = field_length / 2.0
    depth = max(alliance_depth, 1.0)
    if x_in < depth:
        return BLUE, float(np.clip((depth - x_in) / depth, 0.0, 1.0)) * 0.8
    if x_in > field_length - depth:
        return RED, float(np.clip((x_in - (field_length - depth)) / depth, 0.0, 1.0)) * 0.8
    # Neutral zone: weak lean toward the nearer half.
    lean = abs(x_in - mid) / max(mid, 1.0)
    return (RED if x_in > mid else BLUE), float(np.clip(lean, 0.0, 1.0)) * 0.25


def assign_alliance_slots(
    p_red: Sequence[float],
    *,
    n_red: int = 3,
    n_blue: int = 3,
    prior_red: Sequence[float] | None = None,
    prior_weight: float = 0.3,
) -> list[str]:
    """Force exactly n_red / n_blue labels via minimum-cost assignment.

    ``p_red[i]`` is the color-evidence probability for track i; ``prior_red``
    is an optional start-side probability. Tracks beyond the slot count get
    their argmax label. Returns one label per track.
    """
    from ramscout.identity import linear_assignment

    n = len(p_red)
    if n == 0:
        return []
    p = np.asarray(p_red, dtype=np.float64)
    if prior_red is not None:
        pr = np.asarray(prior_red, dtype=np.float64)
        w = float(np.clip(prior_weight, 0.0, 1.0))
        p = (1.0 - w) * p + w * pr
    slots = [RED] * int(max(n_red, 0)) + [BLUE] * int(max(n_blue, 0))
    if not slots:
        return [RED if v >= 0.5 else BLUE for v in p]
    cost = np.zeros((n, len(slots)), dtype=np.float64)
    for j, slot in enumerate(slots):
        cost[:, j] = (1.0 - p) if slot == RED else p
    pairs = linear_assignment(cost)
    labels = [RED if v >= 0.5 else BLUE for v in p]
    for i, j in pairs:
        if 0 <= i < n and 0 <= j < len(slots):
            labels[i] = slots[j]
    return labels


def legacy_bumper_alliance(roi_bgr: np.ndarray) -> str:
    """Whole-box HSV heuristic kept for callers that want the old behaviour."""
    label, _conf = hsv_alliance(chroma_features(roi_bgr))
    return label
