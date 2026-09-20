"""Field-aware gating for robot proposals.

The broadcast → field homography is already known when tracking starts, so we
can ask physical questions about every candidate box *before* it reaches the
multi-object tracker:

- Does the ground-contact point land inside the field polygon (with margin)?
- Is the projected footprint plausible for an FRC robot (~28–36 in with bumpers)?
- Is the box sitting on the field perimeter band (wall / driver-station glass)?
- Does it overlap the scorebug band or hug the crop edges?
- Has anything actually moved inside the box lately (long-window motion energy,
  dense optical flow), or is it a static structure lit by alliance LEDs?

Everything here is cheap OpenCV/numpy so it runs in every mode, including
``potato``. Cloud / YOLO boxes get the geometric checks but are exempt from
static rejection (a dead robot is still a robot).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.trackers.types import Detection

FootFn = Callable[[Sequence[float]], tuple[float, float]]

LOCAL_SOURCES = {"motion", "color", "optical_flow", "browser_potato", "mot"}
STRONG_SOURCES = {"yolo", "openai", "gemini"}

# FRC robots with bumpers are roughly 28–36 in on a side; allow perspective and
# blob-merge slop on top of that.
DEFAULT_MIN_FOOTPRINT_IN = 10.0
DEFAULT_MAX_FOOTPRINT_IN = 96.0
DEFAULT_MAX_HEIGHT_IN = 150.0


@dataclass
class GateStats:
    """Counts of accepted / rejected proposals by reason."""

    accepted: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    flagged: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def total_rejected(self) -> int:
        return int(sum(self.rejected.values()))

    def summary(self) -> str:
        flagged = ""
        if self.flagged:
            fl = ", ".join(f"{k}={v}" for k, v in sorted(self.flagged.items(), key=lambda kv: -kv[1]))
            flagged = f"; flagged static ({fl})"
        if not self.rejected:
            return f"Field gate: {self.accepted} proposals accepted, none rejected{flagged}."
        parts = ", ".join(f"{k}={v}" for k, v in sorted(self.rejected.items(), key=lambda kv: -kv[1]))
        return f"Field gate: {self.accepted} accepted, {self.total_rejected()} rejected ({parts}){flagged}."


@dataclass
class GateVerdict:
    keep: bool
    reason: str | None = None
    penalty: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)


def point_in_field(
    x: float,
    y: float,
    *,
    margin_in: float = 0.0,
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> bool:
    """True when a field-inch point lies inside the field rectangle grown by margin."""
    return (-margin_in <= x <= field_length + margin_in) and (-margin_in <= y <= field_width + margin_in)


def distance_to_perimeter(
    x: float,
    y: float,
    *,
    field_length: float = FIELD_LENGTH,
    field_width: float = FIELD_WIDTH,
) -> float:
    """Signed distance to the nearest field edge (positive inside, negative outside)."""
    dx = min(x, field_length - x)
    dy = min(y, field_width - y)
    return float(min(dx, dy))


class FieldGate:
    """Physically-grounded accept/reject filter for per-frame proposals."""

    def __init__(
        self,
        homography: np.ndarray,
        *,
        crop_y0: float,
        crop_w: int,
        crop_h: int,
        frame_h: int,
        field_length: float = FIELD_LENGTH,
        field_width: float = FIELD_WIDTH,
        outside_margin_in: float = 18.0,
        perimeter_band_in: float = 16.0,
        min_footprint_in: float = DEFAULT_MIN_FOOTPRINT_IN,
        max_footprint_in: float = DEFAULT_MAX_FOOTPRINT_IN,
        max_height_in: float = DEFAULT_MAX_HEIGHT_IN,
        scorebug_bands: Iterable[Sequence[float]] | None = None,
        foot_fn: FootFn | None = None,
        dt_s: float = 0.1,
        static_window_s: float = 6.0,
        edge_px: int = 3,
        flow: bool = True,
    ) -> None:
        self.H = np.asarray(homography, dtype=np.float64)
        self.crop_y0 = float(crop_y0)
        self.crop_w = int(max(crop_w, 1))
        self.crop_h = int(max(crop_h, 1))
        self.frame_h = int(max(frame_h, 1))
        self.field_length = float(field_length)
        self.field_width = float(field_width)
        self.outside_margin_in = float(outside_margin_in)
        self.perimeter_band_in = float(perimeter_band_in)
        self.min_footprint_in = float(min_footprint_in)
        self.max_footprint_in = float(max_footprint_in)
        self.max_height_in = float(max_height_in)
        self.scorebug_bands = [(float(a), float(b)) for a, b in (scorebug_bands or [])]
        self.foot_fn = foot_fn
        self.dt_s = float(max(dt_s, 1e-3))
        self.static_window_s = float(max(static_window_s, self.dt_s))
        self.edge_px = int(edge_px)
        self.use_flow = bool(flow)
        self.stats = GateStats()

        # Long-window motion model at low resolution.
        self._scale = max(1.0, self.crop_w / 160.0)
        self._small_w = max(8, int(round(self.crop_w / self._scale)))
        self._small_h = max(4, int(round(self.crop_h / self._scale)))
        self._prev_small: np.ndarray | None = None
        self._diff: np.ndarray | None = None
        self._diff_thresh: float = 6.0 / 255.0
        self._energy: np.ndarray | None = None
        self._occupancy: np.ndarray | None = None
        self._flow: np.ndarray | None = None
        self._frames = 0
        alpha = 1.0 - float(np.exp(-self.dt_s / self.static_window_s))
        self._alpha = float(np.clip(alpha, 0.02, 0.5))
        self.active_energy = 0.0

    # ------------------------------------------------------------------ geometry
    def foot(self, bbox: Sequence[float]) -> tuple[float, float]:
        """Ground-contact point in full-frame pixels."""
        if self.foot_fn is not None:
            try:
                fx, fy = self.foot_fn(list(bbox))
                return float(fx), float(fy)
            except Exception:  # noqa: BLE001
                pass
        x1, _y1, x2, y2 = [float(v) for v in bbox]
        return (x1 + x2) * 0.5, y2 + self.crop_y0

    def project(self, fx: float, fy: float) -> tuple[float, float]:
        """Full-frame pixel → field inches through the homography."""
        v = self.H @ np.array([fx, fy, 1.0], dtype=np.float64)
        if abs(v[2]) < 1e-9:
            return float("nan"), float("nan")
        return float(v[0] / v[2]), float(v[1] / v[2])

    def field_xy(self, bbox: Sequence[float]) -> tuple[float, float]:
        fx, fy = self.foot(bbox)
        return self.project(fx, fy)

    def footprint_in(self, bbox: Sequence[float]) -> tuple[float, float]:
        """Projected (width, height) of the box in inches at its foot row."""
        x1, y1, x2, y2 = [float(v) for v in bbox]
        _fx, fy = self.foot(bbox)
        lx, ly = self.project(x1, fy)
        rx, ry = self.project(x2, fy)
        width_in = float(np.hypot(rx - lx, ry - ly))
        bw = max(x2 - x1, 1.0)
        bh = max(y2 - y1, 1.0)
        in_per_px = width_in / bw if np.isfinite(width_in) else float("nan")
        height_in = bh * in_per_px
        return width_in, height_in

    def scorebug_overlap(self, bbox: Sequence[float]) -> float:
        """Fraction of the box height that lies inside a scorebug band."""
        if not self.scorebug_bands:
            return 0.0
        _x1, y1, _x2, y2 = [float(v) for v in bbox]
        top = (y1 + self.crop_y0) / self.frame_h
        bot = (y2 + self.crop_y0) / self.frame_h
        h = max(bot - top, 1e-6)
        overlap = 0.0
        for a, b in self.scorebug_bands:
            overlap += max(0.0, min(bot, b) - max(top, a))
        return float(np.clip(overlap / h, 0.0, 1.0))

    def edges_touched(self, bbox: Sequence[float]) -> int:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        e = self.edge_px
        n = 0
        n += int(x1 <= e)
        n += int(y1 <= e)
        n += int(x2 >= self.crop_w - e)
        n += int(y2 >= self.crop_h - e)
        return n

    # -------------------------------------------------------------- motion model
    def observe(self, cropped: np.ndarray) -> None:
        """Update long-window motion energy, proposal occupancy, and dense flow."""
        import cv2

        if cropped is None or cropped.size == 0:
            return
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY) if cropped.ndim == 3 else cropped
        small = cv2.resize(gray, (self._small_w, self._small_h), interpolation=cv2.INTER_AREA)
        small = small.astype(np.float32) / 255.0
        if self._energy is None or self._energy.shape != small.shape:
            self._energy = np.zeros_like(small)
            self._occupancy = np.zeros_like(small)
            self._prev_small = small
            self._flow = None
            self._frames = 1
            return

        diff = np.abs(small - self._prev_small)
        self._diff = diff
        self._diff_thresh = max(6.0 / 255.0, 3.0 * float(np.median(diff)))
        self._energy = (1.0 - self._alpha) * self._energy + self._alpha * diff
        if self.use_flow:
            try:
                prev8 = (self._prev_small * 255).astype(np.uint8)
                cur8 = (small * 255).astype(np.uint8)
                self._flow = cv2.calcOpticalFlowFarneback(
                    prev8, cur8, None, 0.5, 2, 9, 2, 5, 1.1, 0
                )
            except Exception:  # noqa: BLE001
                self._flow = None
        self._prev_small = small
        self._frames += 1
        flat = self._energy.reshape(-1)
        if flat.size:
            # Six robots cover well under 1 % of a low-res field crop, so the
            # "is anything moving" statistic looks at the very top tail only.
            k = max(8, int(flat.size * 0.005))
            k = min(k, flat.size)
            top = np.partition(flat, flat.size - k)[flat.size - k :]
            self.active_energy = float(np.mean(top))
            # Most pixels are static carpet: the median is the sensor / codec
            # noise floor, immune to a flickering wall dominating the top 5 %.
            self.noise_floor = float(np.median(flat))

    noise_floor: float = 0.0

    @property
    def frame_active(self) -> bool:
        """Something in the crop has been moving recently (vs. the noise floor)."""
        if self._frames < 3:
            return False
        return self.active_energy > max(4.0 * self.noise_floor, 0.4 / 255.0)

    def _small_slice(self, bbox: Sequence[float]) -> tuple[slice, slice] | None:
        x1, y1, x2, y2 = [float(v) for v in bbox]
        sx1 = int(np.clip(np.floor(x1 / self._scale), 0, self._small_w - 1))
        sx2 = int(np.clip(np.ceil(x2 / self._scale), sx1 + 1, self._small_w))
        sy1 = int(np.clip(np.floor(y1 / self._scale), 0, self._small_h - 1))
        sy2 = int(np.clip(np.ceil(y2 / self._scale), sy1 + 1, self._small_h))
        if sx2 <= sx1 or sy2 <= sy1:
            return None
        return slice(sy1, sy2), slice(sx1, sx2)

    def box_energy(self, bbox: Sequence[float]) -> float:
        if self._energy is None:
            return 0.0
        sl = self._small_slice(bbox)
        if sl is None:
            return 0.0
        return float(np.mean(self._energy[sl]))

    def box_occupancy(self, bbox: Sequence[float]) -> float:
        if self._occupancy is None:
            return 0.0
        sl = self._small_slice(bbox)
        if sl is None:
            return 0.0
        return float(np.mean(self._occupancy[sl]))

    def box_flow(self, bbox: Sequence[float]) -> tuple[float, float, float]:
        """(median dx, median dy, moving_fraction) in full-res pixels per step.

        A pixel counts as moving only when it both changed between frames and
        carries flow: Farneback alone hallucinates motion on flat, noisy
        texture (carpet, glass), which is exactly where walls live.
        """
        if self._diff is None:
            return 0.0, 0.0, float("nan")
        sl = self._small_slice(bbox)
        if sl is None:
            return 0.0, 0.0, float("nan")
        changed = self._diff[sl] > self._diff_thresh
        if changed.size == 0:
            return 0.0, 0.0, float("nan")
        if self._flow is None:
            return 0.0, 0.0, float(np.mean(changed))
        patch = self._flow[sl]
        mag = np.hypot(patch[..., 0], patch[..., 1])
        moving_mask = changed & (mag > 0.35)
        moving = float(np.mean(moving_mask))
        if np.count_nonzero(moving_mask) >= 3:
            dx = float(np.median(patch[..., 0][moving_mask])) * self._scale
            dy = float(np.median(patch[..., 1][moving_mask])) * self._scale
        else:
            dx = dy = 0.0
        return dx, dy, moving

    def _mark_occupancy(self, boxes: Iterable[Sequence[float]]) -> None:
        if self._occupancy is None:
            return
        mask = np.zeros_like(self._occupancy)
        for bbox in boxes:
            sl = self._small_slice(bbox)
            if sl is not None:
                mask[sl] = 1.0
        self._occupancy = (1.0 - self._alpha) * self._occupancy + self._alpha * mask

    # ----------------------------------------------------------------- verdicts
    def evaluate(self, det: Detection) -> GateVerdict:
        bbox = [float(v) for v in det.bbox]
        info: dict[str, Any] = {}
        penalty = 0.0
        source = det.source or ""
        strong = source in STRONG_SOURCES

        fx, fy = self.foot(bbox)
        x_in, y_in = self.project(fx, fy)
        info["foot"] = (fx, fy)
        info["field_xy"] = (x_in, y_in)
        if not (np.isfinite(x_in) and np.isfinite(y_in)):
            return GateVerdict(False, "projection", 0.0, info)

        # Boxes living almost entirely inside the scorebug band are overlay
        # graphics; partial overlap (far robots under the bug) is only penalized.
        bug = self.scorebug_overlap(bbox)
        info["scorebug_overlap"] = bug
        if bug > 0.85:
            return GateVerdict(False, "scorebug", 0.0, info)
        if bug > 0.1:
            penalty += 0.25 * bug

        if not point_in_field(
            x_in,
            y_in,
            margin_in=self.outside_margin_in,
            field_length=self.field_length,
            field_width=self.field_width,
        ):
            return GateVerdict(False, "outside_field", 0.0, info)

        width_in, height_in = self.footprint_in(bbox)
        info["footprint_in"] = (width_in, height_in)
        if np.isfinite(width_in):
            if width_in > self.max_footprint_in:
                return GateVerdict(False, "footprint_huge", 0.0, info)
            if width_in < self.min_footprint_in:
                return GateVerdict(False, "footprint_tiny", 0.0, info)
        if np.isfinite(height_in) and height_in > self.max_height_in:
            return GateVerdict(False, "footprint_tall", 0.0, info)

        edge_dist = distance_to_perimeter(
            x_in, y_in, field_length=self.field_length, field_width=self.field_width
        )
        perimeter = edge_dist < self.perimeter_band_in
        info["perimeter"] = perimeter
        bw = max(bbox[2] - bbox[0], 1.0)
        bh = max(bbox[3] - bbox[1], 1.0)
        aspect = bw / bh

        edges = self.edges_touched(bbox)
        info["edges"] = edges
        if edges >= 2 or (edges >= 1 and bw > 0.35 * self.crop_w):
            return GateVerdict(False, "edge", 0.0, info)
        penalty += 0.08 * edges

        energy = self.box_energy(bbox)
        dx, dy, moving = self.box_flow(bbox)
        info["energy"] = energy
        info["flow"] = (dx, dy)
        info["moving_fraction"] = moving
        info["occupancy"] = self.box_occupancy(bbox)

        if perimeter:
            penalty += 0.15
            # Wide, low blobs hugging the boundary are wall / glass, not robots.
            if aspect > 2.4 or bw > 0.30 * self.crop_w:
                return GateVerdict(False, "perimeter_wall", penalty, info)

        # Static / flow analysis. These do not hard-reject (a robot parked at
        # the loading station is still a robot): they flag the proposal so MOT
        # refuses to *spawn* a track from it and never confirms a track that
        # has not physically moved. Walls therefore never become tracks.
        static = False
        if self.frame_active and not strong:
            floor = self.noise_floor
            static_energy = energy < max(2.5 * floor, 0.75 / 255.0)
            flow_static = (not np.isfinite(moving)) or moving < 0.15
            if static_energy and flow_static:
                static = True
                self.stats.flagged["static"] = self.stats.flagged.get("static", 0) + 1
            elif source == "motion" and np.isfinite(moving) and moving < 0.08:
                # A motion blob whose pixels are not actually flowing is a
                # lighting flicker / shadow on a static structure.
                static = True
                self.stats.flagged["flow_inconsistent"] = self.stats.flagged.get("flow_inconsistent", 0) + 1
            if static:
                penalty += 0.2
            if perimeter and static:
                penalty += 0.1
            if info["occupancy"] > 0.92 and static:
                penalty += 0.1
            if np.isfinite(moving) and moving > 0.5:
                # Motion / flow / (color) agreement: a solidly flowing box.
                penalty -= 0.05
        info["static"] = static

        return GateVerdict(True, None, penalty, info)

    def filter(self, detections: list[Detection], *, mark: bool = True) -> list[Detection]:
        """Apply verdicts; annotate survivors with field-space metadata.

        ``mark`` feeds survivors into the occupancy model (skip for flow
        continuations so the same box is not counted twice per frame).
        """
        kept: list[Detection] = []
        for det in detections:
            verdict = self.evaluate(det)
            if not verdict.keep:
                self.stats.reject(verdict.reason or "unknown")
                continue
            self.stats.accepted += 1
            det.confidence = float(np.clip(det.confidence - verdict.penalty, 0.05, 1.0))
            meta = dict(det.meta or {})
            meta["field_xy"] = verdict.info.get("field_xy")
            meta["foot"] = verdict.info.get("foot")
            meta["perimeter"] = bool(verdict.info.get("perimeter", False))
            meta["flow"] = verdict.info.get("flow")
            meta["moving_fraction"] = verdict.info.get("moving_fraction")
            meta["energy"] = verdict.info.get("energy")
            meta["static"] = bool(verdict.info.get("static", False))
            det.meta = meta
            kept.append(det)
        if mark:
            self._mark_occupancy([d.bbox for d in kept])
        return kept

    def projector(self) -> Callable[[Sequence[float]], tuple[float, float]]:
        """Callable for MOT: bbox (crop px) → field inches."""
        return self.field_xy


def scorebug_bands_for_year(year: int | None = None) -> list[tuple[float, float]]:
    """Frame-normalized scorebug bands from the year config (safe fallback)."""
    try:
        from ramscout.gameconfig import load_game

        overlay = load_game(year).get("overlay") or {}
        bands = []
        for key in ("top_band", "bottom_band"):
            band = overlay.get(key)
            if band and len(band) == 2:
                bands.append((float(band[0]), float(band[1])))
        return bands
    except Exception:  # noqa: BLE001
        return [(0.0, 0.12), (0.84, 1.0)]
