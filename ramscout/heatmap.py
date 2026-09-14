"""Field occupancy heatmaps from robot path samples."""

from __future__ import annotations

from typing import Any

import numpy as np

from ramscout.field import FIELD_LENGTH, FIELD_WIDTH


def build_heatmap(
    samples: list[dict[str, Any]],
    *,
    team: str | None = None,
    bins_x: int = 54,
    bins_y: int = 26,
    period: str | None = None,
) -> dict[str, Any]:
    """Return a normalized occupancy grid in field inches."""
    xs: list[float] = []
    ys: list[float] = []
    for sample in samples or []:
        if team and str(sample.get("team") or "") != str(team):
            continue
        t = float(sample.get("t") or 0)
        if period == "auto" and t > 15:
            continue
        if period == "teleop" and (t < 15 or t > 135):
            continue
        if period == "endgame" and t < 135:
            continue
        x = sample.get("x")
        y = sample.get("y")
        if x is None or y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))

    grid = np.zeros((bins_y, bins_x), dtype=np.float32)
    if not xs:
        return {
            "bins_x": bins_x,
            "bins_y": bins_y,
            "field_length_in": FIELD_LENGTH,
            "field_width_in": FIELD_WIDTH,
            "max": 0.0,
            "grid": grid.tolist(),
            "samples": 0,
            "team": team,
            "period": period,
        }

    x_edges = np.linspace(0, FIELD_LENGTH, bins_x + 1)
    y_edges = np.linspace(0, FIELD_WIDTH, bins_y + 1)
    hist, _, _ = np.histogram2d(ys, xs, bins=[y_edges, x_edges])
    peak = float(hist.max()) or 1.0
    norm = (hist / peak).astype(np.float32)
    return {
        "bins_x": bins_x,
        "bins_y": bins_y,
        "field_length_in": FIELD_LENGTH,
        "field_width_in": FIELD_WIDTH,
        "max": peak,
        "grid": [[round(float(v), 4) for v in row] for row in norm.tolist()],
        "samples": len(xs),
        "team": team,
        "period": period,
    }
