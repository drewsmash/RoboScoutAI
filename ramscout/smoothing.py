"""Rauch–Tung–Striebel smoothing of robot paths in field coordinates.

The online tracker is causal: every position carries the latest measurement
noise (blob jitter, foot-point wobble, perspective). Once the match is over
we can run a constant-velocity Kalman filter *forward* and the RTS
*backward* pass so each sample is conditioned on the whole track. Paths get
cleaner without lagging, and speeds / zone transitions stop flickering.

Raw values are preserved as ``x_raw`` / ``y_raw``; ``px`` / ``py`` (pixel
feet) are untouched so recalibration still works.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Process / measurement noise in inches. Robots accelerate hard, so give the
# velocity a wide prior; measurement noise ≈ half a bumper.
DEFAULT_Q_ACCEL_IN_S2 = 260.0
DEFAULT_R_IN = 9.0
MAX_GAP_S = 2.5  # beyond this the track is treated as two independent runs


def rts_smooth(
    ts: np.ndarray,
    zs: np.ndarray,
    *,
    q_accel: float = DEFAULT_Q_ACCEL_IN_S2,
    r_meas: float = DEFAULT_R_IN,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a 2-D trajectory. Returns (positions N×2, velocities N×2)."""
    n = len(ts)
    if n == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))
    if n == 1:
        return zs.astype(np.float64).copy(), np.zeros((1, 2))
    x = np.zeros((n, 4))  # x, y, vx, vy
    P = np.zeros((n, 4, 4))
    x_pred = np.zeros((n, 4))
    P_pred = np.zeros((n, 4, 4))
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    R = np.eye(2) * (r_meas**2)
    x[0] = [zs[0, 0], zs[0, 1], 0.0, 0.0]
    P[0] = np.diag([r_meas**2, r_meas**2, 150.0**2, 150.0**2])
    x_pred[0] = x[0]
    P_pred[0] = P[0]
    Fs = np.zeros((n, 4, 4))
    Fs[0] = np.eye(4)
    for k in range(1, n):
        dt = float(max(ts[k] - ts[k - 1], 1e-3))
        F = np.eye(4)
        F[0, 2] = dt
        F[1, 3] = dt
        Fs[k] = F
        q = q_accel**2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ (np.eye(2) * q) @ G.T
        xp = F @ x[k - 1]
        Pp = F @ P[k - 1] @ F.T + Q
        x_pred[k] = xp
        P_pred[k] = Pp
        S = H @ Pp @ H.T + R
        K = Pp @ H.T @ np.linalg.inv(S)
        y = zs[k] - H @ xp
        x[k] = xp + K @ y
        P[k] = (np.eye(4) - K @ H) @ Pp
    # Backward pass.
    xs = x.copy()
    Ps = P.copy()
    for k in range(n - 2, -1, -1):
        F = Fs[k + 1]
        try:
            C = P[k] @ F.T @ np.linalg.inv(P_pred[k + 1])
        except np.linalg.LinAlgError:
            continue
        xs[k] = x[k] + C @ (xs[k + 1] - x_pred[k + 1])
        Ps[k] = P[k] + C @ (Ps[k + 1] - P_pred[k + 1]) @ C.T
    return xs[:, :2], xs[:, 2:]


def smooth_samples(
    samples: list[dict[str, Any]],
    *,
    dt_hint: float = 0.1,
    q_accel: float = DEFAULT_Q_ACCEL_IN_S2,
    r_meas: float = DEFAULT_R_IN,
    max_gap_s: float = MAX_GAP_S,
) -> list[dict[str, Any]]:
    """Return new sample dicts with RTS-smoothed ``x``/``y`` (+ ``speed_in_s``)."""
    if not samples:
        return samples
    by_track: dict[int, list[int]] = {}
    for i, s in enumerate(samples):
        by_track.setdefault(int(s["track_id"]), []).append(i)
    out = [dict(s) for s in samples]
    for idxs in by_track.values():
        idxs.sort(key=lambda i: float(samples[i]["t"]))
        # Split at long gaps (occlusions / layout switches).
        runs: list[list[int]] = [[idxs[0]]]
        for i in idxs[1:]:
            if float(samples[i]["t"]) - float(samples[runs[-1][-1]]["t"]) > max_gap_s:
                runs.append([i])
            else:
                runs[-1].append(i)
        for run in runs:
            # Skip invalid / unavailable field samples — do not interpolate them in.
            usable = [
                i
                for i in run
                if samples[i].get("field_valid") is not False
                and samples[i].get("x") is not None
                and samples[i].get("y") is not None
                and np.isfinite(float(samples[i]["x"]))
                and np.isfinite(float(samples[i]["y"]))
            ]
            if len(usable) < 2:
                continue
            # Further split when consecutive usable samples jump across a gap
            # that previously contained invalid projections.
            subruns: list[list[int]] = [[usable[0]]]
            for i in usable[1:]:
                prev = subruns[-1][-1]
                if float(samples[i]["t"]) - float(samples[prev]["t"]) > max_gap_s:
                    subruns.append([i])
                else:
                    subruns[-1].append(i)
            for sub in subruns:
                if len(sub) < 2:
                    continue
                ts = np.array([float(samples[i]["t"]) for i in sub])
                zs = np.array([[float(samples[i]["x"]), float(samples[i]["y"])] for i in sub])
                pos, vel = rts_smooth(ts, zs, q_accel=q_accel, r_meas=r_meas)
                for j, i in enumerate(sub):
                    row = out[i]
                    row.setdefault("x_raw", float(samples[i]["x"]))
                    row.setdefault("y_raw", float(samples[i]["y"]))
                    row["x"] = float(pos[j, 0])
                    row["y"] = float(pos[j, 1])
                    row["speed_in_s"] = round(float(np.hypot(vel[j, 0], vel[j, 1])), 1)
                    row["smoothed"] = True
    return out
