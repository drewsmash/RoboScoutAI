"""Field-line / carpet-boundary homography refinement.

The depth-derived trapezoid in :mod:`ramscout.bev` only encodes camera
*pitch*. Real broadcast cameras are also offset, rolled and zoomed, so the
field rarely fills the default trapezoid. This module looks at the picture
itself (the way sports-field calibration does with pitch lines):

1. **Carpet mask** — FRC fields are grey carpet inside a polycarbonate
   perimeter; the temporal median of a few frames removes robots and people.
   Low-saturation mid-grey pixels are carpet (or bleacher concrete, which is
   why we keep the *largest connected component that touches the pane's
   central band*).
2. **Quadrilateral** — the mask's convex hull is reduced to its four extreme
   corners (top-left/top-right/bottom-right/bottom-left by ``x±y``); the
   result must be a convex, wide, perspective-plausible quad that covers a
   sane fraction of the pane.
3. **Alliance orientation** — saturated red vs blue mass in the left vs right
   thirds (driver-station walls, alliance-zone tape) tells which wall is
   blue; the homography is mirrored so ``x = 0`` is always the blue wall.
4. **Tape lines** — red / blue tape running across the field gives a *cross
   check*: on a good homography the projected line stays roughly at a
   constant field ``x``.

Everything degrades gracefully: when no quad passes, callers keep the depth
trapezoid and only the orientation cue is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass
class FieldQuad:
    corners: list[list[float]]  # TL, TR, BR, BL in *pane* pixels
    area_frac: float
    confidence: float
    detail: str
    mask_frac: float = 0.0
    hull_points: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "corners": [[round(float(x), 2), round(float(y), 2)] for x, y in self.corners],
            "area_frac": round(float(self.area_frac), 4),
            "confidence": round(float(self.confidence), 3),
            "detail": self.detail,
        }


@dataclass
class OrientationCue:
    blue_left: bool
    confidence: float
    red_left: float
    red_right: float
    blue_left_mass: float
    blue_right_mass: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "blue_left": bool(self.blue_left),
            "confidence": round(float(self.confidence), 3),
            "red_left": round(float(self.red_left), 5),
            "red_right": round(float(self.red_right), 5),
            "blue_left_mass": round(float(self.blue_left_mass), 5),
            "blue_right_mass": round(float(self.blue_right_mass), 5),
        }


def texture_map(bgr: np.ndarray, ksize: int = 7) -> np.ndarray:
    """Local gradient energy (0–255 float). Carpet is smooth; crowds are not."""
    import cv2

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    return cv2.blur(mag, (ksize, ksize))


def _seed_pixels(bgr: np.ndarray, seed_mask: np.ndarray | None, texture: np.ndarray, sat: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels used to learn the carpet colour."""
    h, w = bgr.shape[:2]
    if seed_mask is not None and np.count_nonzero(seed_mask) > 200:
        seed = seed_mask > 0
    else:
        seed = np.zeros((h, w), dtype=bool)
        seed[int(h * 0.40) : int(h * 0.90), int(w * 0.25) : int(w * 0.75)] = True
    # Only smooth, unsaturated pixels inside the seed area are carpet: this
    # drops field elements (scale, switches, cubes, tape) from the model.
    smooth = texture < np.percentile(texture[seed], 60)
    cand = seed & smooth & (sat < 70)
    if np.count_nonzero(cand) < 200:
        cand = seed & (sat < 90)
    return cand


def carpet_color(bgr: np.ndarray, seed_mask: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    """Reference carpet colour (Lab) and a distance threshold.

    The colour is the median of smooth, unsaturated pixels in the pane's
    central band — or in ``seed_mask`` (e.g. where robots have driven). The
    threshold adapts to the carpet's own spread (broadcast noise, shading).
    """
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tex = texture_map(bgr)
    cand = _seed_pixels(bgr, seed_mask, tex, hsv[..., 1])
    seed = lab[cand]
    ref = np.median(seed, axis=0)
    dist = np.linalg.norm(seed - ref, axis=1)
    mad = float(np.median(np.abs(dist - np.median(dist)))) + 1e-6
    thr = float(np.clip(np.median(dist) + 3.0 * mad, 12.0, 34.0))
    return ref, thr


def carpet_mask(bgr: np.ndarray, seed_mask: np.ndarray | None = None) -> np.ndarray:
    """Binary mask of pixels that look like the pane's carpet (uint8 0/255).

    Three cues, all adaptive to the pane: (1) Lab distance to the learned
    carpet colour, (2) low saturation (carpet is grey; tape, bumpers and
    alliance-station colours are not), (3) low local texture (crowds,
    banners and bleachers are busy; carpet is flat). Robots, tape lines and
    field elements become holes that the hull step fills.
    """
    import cv2

    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    tex = texture_map(bgr)
    cand = _seed_pixels(bgr, seed_mask, tex, hsv[..., 1])
    seed = lab[cand]
    ref = np.median(seed, axis=0)
    dist_seed = np.linalg.norm(seed - ref, axis=1)
    mad = float(np.median(np.abs(dist_seed - np.median(dist_seed)))) + 1e-6
    thr = float(np.clip(np.median(dist_seed) + 2.0 * mad, 12.0, 30.0))
    tex_thr = float(max(np.percentile(tex[cand], 75) * 1.6, 12.0))

    dist = np.linalg.norm(lab - ref.reshape(1, 1, 3), axis=2)
    mask = (dist < thr) & (hsv[..., 1] < 95) & (tex < tex_thr)
    out = mask.astype(np.uint8) * 255
    h, w = bgr.shape[:2]
    # Bridge tape lines / thin field elements, then drop speckle.
    k = max(5, (min(h, w) // 40) | 1)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # The field perimeter (rails, driver-station tables) is a persistent
    # edge: cutting the mask along strong edges keeps grey bleacher steps
    # from merging with the carpet through a few connecting pixels.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    out[edges > 0] = 0
    return out


def temporal_median(frames: Sequence[np.ndarray]) -> np.ndarray:
    if len(frames) == 1:
        return frames[0]
    h, w = frames[0].shape[:2]
    same = [f for f in frames if f.shape[:2] == (h, w)]
    return np.median(np.stack(same).astype(np.float32), axis=0).astype(np.uint8)


def _largest_component_touching_center(mask: np.ndarray, seed: np.ndarray | None = None) -> np.ndarray | None:
    """Union of the large carpet components that reach the pane's central band.

    Field elements (scale, switches, hubs) plus the edge cut can split the
    carpet into two or three pieces, so every sizeable piece that touches the
    band — or overlaps the robot-motion ``seed`` — is kept.
    """
    import cv2

    h, w = mask.shape[:2]
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    band = np.zeros((h, w), dtype=bool)
    band[int(h * 0.35) : int(h * 0.9), int(w * 0.25) : int(w * 0.75)] = True
    if seed is not None and np.count_nonzero(seed) > 50:
        band |= seed > 0
    areas = {i: int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n)}
    largest = max(areas.values()) if areas else 0
    chosen = [
        i
        for i, area in areas.items()
        if area >= 0.02 * h * w and area >= 0.08 * largest and np.any(band[labels == i])
    ]
    if not chosen:
        return None
    comp = np.isin(labels, chosen).astype(np.uint8) * 255
    # Fill holes (field elements, tape, robots) so the boundary is the outer one.
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(comp)
    cv2.drawContours(filled, contours, -1, 255, thickness=-1)
    return filled


def _robust_line(xs: np.ndarray, ys: np.ndarray, *, iters: int = 4) -> tuple[float, float, float] | None:
    """Fit ``y = a·x + b`` with iterative MAD trimming. Returns (a, b, inlier_ratio)."""
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    if len(xs) < 8:
        return None
    keep = np.ones(len(xs), dtype=bool)
    a = b = 0.0
    for _ in range(iters):
        if keep.sum() < 6:
            return None
        A = np.stack([xs[keep], np.ones(int(keep.sum()))], axis=1)
        sol, *_rest = np.linalg.lstsq(A, ys[keep], rcond=None)
        a, b = float(sol[0]), float(sol[1])
        res = np.abs(ys - (a * xs + b))
        mad = float(np.median(res[keep])) + 1e-6
        new_keep = res <= max(3.0 * mad, 1.5)
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    return a, b, float(keep.mean())


def _intersect(l1: tuple[float, float], l2: tuple[float, float], *, swap1: bool, swap2: bool) -> list[float] | None:
    """Intersect two lines given as (a, b). ``swap`` marks lines in x = a·y + b form."""

    def to_general(a: float, b: float, swap: bool) -> tuple[float, float, float]:
        # y = a x + b  →  a x − y + b = 0 ; x = a y + b → x − a y − b = 0
        return (a, -1.0, b) if not swap else (1.0, -a, -b)

    A1, B1, C1 = to_general(l1[0], l1[1], swap1)
    A2, B2, C2 = to_general(l2[0], l2[1], swap2)
    det = A1 * B2 - A2 * B1
    if abs(det) < 1e-9:
        return None
    x = (B1 * C2 - B2 * C1) / det
    y = (C1 * A2 - C2 * A1) / det
    return [float(x), float(y)]


def quad_from_boundary(comp: np.ndarray) -> tuple[list[list[float]], float] | None:
    """Four straight edges fitted robustly to a filled carpet component.

    Row-wise left/right extents (middle 15–85 % of the component's rows) and
    column-wise top/bottom extents (middle 15–85 % of its columns) are each
    fitted with a trimmed least-squares line; leaks into bleachers, referees
    on the wall and the scorebug are outliers that get trimmed. Returns
    (TL, TR, BR, BL) and the mean inlier ratio.
    """
    ys, xs = np.nonzero(comp)
    if len(xs) < 200:
        return None
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    rows = np.arange(y_min + int(0.15 * (y_max - y_min)), y_min + int(0.85 * (y_max - y_min)) + 1)
    cols = np.arange(x_min + int(0.15 * (x_max - x_min)), x_min + int(0.85 * (x_max - x_min)) + 1)
    left_x, right_x, row_ok = [], [], []
    for r in rows:
        cx = np.nonzero(comp[r])[0]
        if len(cx) < 3:
            continue
        row_ok.append(r)
        left_x.append(cx.min())
        right_x.append(cx.max())
    top_y, bot_y, col_ok = [], [], []
    for c in cols:
        cy = np.nonzero(comp[:, c])[0]
        if len(cy) < 3:
            continue
        col_ok.append(c)
        top_y.append(cy.min())
        bot_y.append(cy.max())
    row_ok = np.asarray(row_ok, dtype=np.float64)
    col_ok = np.asarray(col_ok, dtype=np.float64)
    left = _robust_line(row_ok, np.asarray(left_x))  # x = a y + b
    right = _robust_line(row_ok, np.asarray(right_x))
    top = _robust_line(col_ok, np.asarray(top_y))  # y = a x + b
    bottom = _robust_line(col_ok, np.asarray(bot_y))
    if None in (left, right, top, bottom):
        return None
    tl = _intersect((top[0], top[1]), (left[0], left[1]), swap1=False, swap2=True)
    tr = _intersect((top[0], top[1]), (right[0], right[1]), swap1=False, swap2=True)
    br = _intersect((bottom[0], bottom[1]), (right[0], right[1]), swap1=False, swap2=True)
    bl = _intersect((bottom[0], bottom[1]), (left[0], left[1]), swap1=False, swap2=True)
    if None in (tl, tr, br, bl):
        return None
    inlier = float(np.mean([left[2], right[2], top[2], bottom[2]]))
    return [tl, tr, br, bl], inlier


def _extreme_corners(points: np.ndarray) -> list[list[float]]:
    pts = points.reshape(-1, 2).astype(np.float64)
    s = pts[:, 0] + pts[:, 1]
    d = pts[:, 0] - pts[:, 1]
    tl = pts[int(np.argmin(s))]
    br = pts[int(np.argmax(s))]
    tr = pts[int(np.argmax(d))]
    bl = pts[int(np.argmin(d))]
    return [tl.tolist(), tr.tolist(), br.tolist(), bl.tolist()]


def _quad_area(q: Sequence[Sequence[float]]) -> float:
    x = np.array([p[0] for p in q])
    y = np.array([p[1] for p in q])
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_convex(q: Sequence[Sequence[float]]) -> bool:
    signs = []
    for i in range(4):
        a, b, c = np.array(q[i]), np.array(q[(i + 1) % 4]), np.array(q[(i + 2) % 4])
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        signs.append(np.sign(cross))
    return all(s == signs[0] for s in signs) and signs[0] != 0


def detect_field_quad(
    frames: Sequence[np.ndarray] | np.ndarray,
    *,
    motion_mask: np.ndarray | None = None,
) -> FieldQuad | None:
    """Estimate the carpet's four outer corners in pane pixels.

    ``motion_mask`` (same size as the frames, non-zero where robots have been
    seen moving) seeds the carpet colour model and anchors the component
    choice — robots only drive on the field.
    """
    import cv2

    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    if not frames:
        return None
    bg = temporal_median(list(frames))
    h, w = bg.shape[:2]
    if h < 40 or w < 60:
        return None
    scale = min(1.0, 640.0 / w)
    small = cv2.resize(bg, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else bg
    sh, sw = small.shape[:2]
    seed = None
    if motion_mask is not None and motion_mask.size:
        seed = cv2.resize(motion_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST)
        seed = cv2.erode(seed, np.ones((5, 5), np.uint8))
    mask = carpet_mask(small, seed)
    if seed is not None and np.count_nonzero(seed) > 100:
        # Robots drive on carpet: the field component must contain the motion.
        mask = cv2.bitwise_or(mask, cv2.dilate(seed, np.ones((9, 9), np.uint8)))
    mask_frac = float(np.mean(mask > 0))
    comp = _largest_component_touching_center(mask, seed)
    if comp is None:
        return None
    fitted = quad_from_boundary(comp)
    if fitted is None:
        return None
    quad, inlier = fitted
    # The mask lost ~3 px at every boundary to the edge cut / opening; grow
    # the quad back outward about its centroid.
    cx = float(np.mean([p[0] for p in quad]))
    cy = float(np.mean([p[1] for p in quad]))
    grow = 3.0
    quad = [
        [cx + (x - cx) * (1.0 + 2 * grow / max(abs(x - cx) * 2, 1.0)), cy + (y - cy) * (1.0 + 2 * grow / max(abs(y - cy) * 2, 1.0))]
        for x, y in quad
    ]
    quad = [[float(np.clip(x, -0.2 * sw, 1.2 * sw)), float(np.clip(y, -0.2 * sh, 1.2 * sh))] for x, y in quad]
    area = _quad_area(quad)
    area_frac = area / float(sh * sw)
    top_w = abs(quad[1][0] - quad[0][0])
    bot_w = abs(quad[2][0] - quad[3][0])
    height = abs((quad[2][1] + quad[3][1]) * 0.5 - (quad[0][1] + quad[1][1]) * 0.5)
    detail_bits = [f"carpet {mask_frac:.0%} of pane", f"quad {area_frac:.0%} of pane", f"edge inliers {inlier:.0%}"]
    if not _is_convex(quad):
        return None
    if area_frac < 0.12 or area_frac > 1.1:
        return None
    if height < 0.12 * sh or max(top_w, bot_w) < 0.35 * sw:
        return None
    # A broadcast camera looks *down* at the field: the near (bottom) edge is
    # wider than the far edge. Allow a near-orthographic overhead (ratio ~1).
    persp = bot_w / max(top_w, 1.0)
    if persp < 0.8:
        return None
    # How much of the fitted quad is actually carpet component (robots and
    # field elements are filled, so a clean field approaches 1).
    poly = np.zeros_like(comp)
    cv2.fillPoly(poly, [np.array(quad, dtype=np.int32)], 255)
    fill = float(np.count_nonzero(comp & poly)) / max(float(np.count_nonzero(poly)), 1.0)
    conf = 0.30
    conf += 0.25 * float(np.clip((fill - 0.55) / 0.35, 0.0, 1.0))
    conf += 0.20 * float(np.clip((inlier - 0.5) / 0.4, 0.0, 1.0))
    conf += 0.10 * float(np.clip((area_frac - 0.12) / 0.3, 0.0, 1.0))
    conf += 0.15 * float(np.clip((persp - 0.9) / 0.6, 0.0, 1.0))
    detail_bits.append(f"near/far width {persp:.2f}")
    detail_bits.append(f"fill {fill:.0%}")
    corners = [[x / scale, y / scale] for x, y in quad]
    return FieldQuad(
        corners=corners,
        area_frac=area_frac,
        confidence=float(np.clip(conf, 0.0, 0.95)),
        detail="Field carpet boundary: " + ", ".join(detail_bits) + ".",
        mask_frac=mask_frac,
        hull_points=int(np.count_nonzero(comp)),
    )


def alliance_orientation(frames: Sequence[np.ndarray] | np.ndarray, quad: FieldQuad | None = None) -> OrientationCue:
    """Which side of the pane is the blue alliance wall?"""
    import cv2

    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    bg = temporal_median(list(frames))
    h, w = bg.shape[:2]
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    red = (((hch <= 10) | (hch >= 165)) & (s > 120) & (v > 70)).astype(np.uint8)
    blue = ((hch >= 100) & (hch <= 128) & (s > 120) & (v > 60)).astype(np.uint8)
    k = max(3, (min(h, w) // 80) | 1)
    kernel = np.ones((k, k), np.uint8)
    if quad is not None:
        # Two cues that both point at the alliance's own side:
        #  * *thin* saturated structures on the carpet = alliance tape / zone
        #    outlines (thick blobs are game-piece plates whose colours are
        #    randomised per match in some seasons, so they are removed)
        #  * *thick* saturated blobs just outside the carpet's left / right
        #    edges = driver-station wall panels
        poly = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly, [np.array(quad.corners, dtype=np.int32)], 1)
        inside = poly > 0
        ys = [c[1] for c in quad.corners]
        xs = [c[0] for c in quad.corners]
        x_min, x_max = min(xs), max(xs)
        y_top = int(np.clip(min(ys) - 0.35 * (max(ys) - min(ys)), 0, h - 1))
        y_bot = int(np.clip(max(ys), y_top + 1, h))
        wall_w = max(int(0.12 * (x_max - x_min)), 4)
        walls = np.zeros((h, w), dtype=bool)
        lx0 = int(np.clip(x_min - wall_w, 0, w - 1))
        lx1 = int(np.clip(x_min + wall_w * 0.5, lx0 + 1, w))
        rx1 = int(np.clip(x_max + wall_w, 1, w))
        rx0 = int(np.clip(x_max - wall_w * 0.5, 0, rx1 - 1))
        walls[y_top:y_bot, lx0:lx1] = True
        walls[y_top:y_bot, rx0:rx1] = True
        walls &= ~inside
        mid = 0.5 * (x_min + x_max)
        left_half = np.zeros((h, w), dtype=bool)
        left_half[:, : int(mid)] = True

        def masses(m: np.ndarray) -> tuple[float, float]:
            thick = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel).astype(bool)
            thin = (m > 0) & ~thick
            tape_l = float(np.count_nonzero(thin & inside & left_half))
            tape_r = float(np.count_nonzero(thin & inside & ~left_half))
            wall_l = float(np.count_nonzero(thick & walls & left_half))
            wall_r = float(np.count_nonzero(thick & walls & ~left_half))
            norm = float(max(np.count_nonzero(inside), 1))
            return (tape_l + wall_l) / norm, (tape_r + wall_r) / norm

        rl, rr = masses(red)
        bl, br = masses(blue)
    else:
        # No quad: fall back to the lower band's outer thirds with the
        # speck-removing opening (crowd shirts).
        red_f = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel).astype(np.float32)
        blue_f = cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel).astype(np.float32)
        y0 = int(h * 0.25)
        third = max(w // 3, 1)
        rl, rr = float(red_f[y0:, :third].mean()), float(red_f[y0:, -third:].mean())
        bl, br = float(blue_f[y0:, :third].mean()), float(blue_f[y0:, -third:].mean())
    left_blue_evidence = (bl - br) + (rr - rl)
    total = rl + rr + bl + br + 1e-6
    conf = float(np.clip(abs(left_blue_evidence) / total, 0.0, 1.0))
    blue_left = left_blue_evidence >= 0
    if total < 1e-4:
        conf = 0.0
        blue_left = True
    return OrientationCue(blue_left, conf, rl, rr, bl, br)


def orient_corners(corners: Sequence[Sequence[float]], blue_left: bool) -> list[list[float]]:
    """Reorder TL,TR,BR,BL so the first corner is the blue wall / far side.

    Field coordinates map corner 0 → (0,0) (blue wall, scoring-table side)
    and corner 1 → (FIELD_LENGTH, 0). When blue is on the *right* of the
    picture we mirror horizontally: TR→0, TL→1, BL→2, BR→3.
    """
    tl, tr, br, bl = [list(map(float, c)) for c in corners]
    if blue_left:
        return [tl, tr, br, bl]
    return [tr, tl, bl, br]


def tape_line_consistency(frames: Sequence[np.ndarray] | np.ndarray, homography: np.ndarray, *, crop_x0: float = 0.0, crop_y0: float = 0.0) -> dict[str, float]:
    """How straight (constant field-x) do red/blue tape pixels project?

    Lower spread = the homography agrees with the field markings.
    """
    import cv2

    if isinstance(frames, np.ndarray) and frames.ndim == 3:
        frames = [frames]
    bg = temporal_median(list(frames))
    hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
    hch, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    out: dict[str, float] = {}
    for name, mask in (
        ("red", ((hch <= 10) | (hch >= 165)) & (s > 120) & (v > 70)),
        ("blue", (hch >= 100) & (hch <= 128) & (s > 120) & (v > 60)),
    ):
        ys, xs = np.nonzero(mask)
        if len(xs) < 40:
            continue
        pts = np.stack([xs + crop_x0, ys + crop_y0], axis=1).astype(np.float32).reshape(-1, 1, 2)
        mapped = cv2.perspectiveTransform(pts, homography.astype(np.float64)).reshape(-1, 2)
        finite = np.isfinite(mapped).all(axis=1)
        if finite.sum() < 40:
            continue
        out[f"{name}_x_spread_in"] = float(np.std(mapped[finite, 0]))
        out[f"{name}_x_mean_in"] = float(np.mean(mapped[finite, 0]))
    return out
