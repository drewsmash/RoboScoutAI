#!/usr/bin/env python3
"""Run tracking on a local VOD and emit path-quality + field-view artifacts."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ramscout.detect import track_video
from ramscout.events import Pose, build_cards, detect_events
from ramscout.field import FIELD_LENGTH, FIELD_WIDTH
from ramscout.geometry import is_field_sample_drawable
from ramscout.multicam import analyze_video


def analyze_samples(samples: list[dict]) -> dict:
    total = len(samples)
    valid = [s for s in samples if is_field_sample_drawable(s)]
    invalid = total - len(valid)
    edge = 0
    corner = 0
    for s in valid:
        x, y = float(s["x"]), float(s["y"])
        on_edge = x <= 5 or y <= 5 or x >= FIELD_LENGTH - 5 or y >= FIELD_WIDTH - 5
        if on_edge:
            edge += 1
        if (x <= 5 or x >= FIELD_LENGTH - 5) and (y <= 5 or y >= FIELD_WIDTH - 5):
            corner += 1
    jumps = 0
    by = defaultdict(list)
    for s in valid:
        by[int(s["track_id"])].append(s)
    for rows in by.values():
        rows = sorted(rows, key=lambda r: float(r["t"]))
        for a, b in zip(rows, rows[1:]):
            dt = float(b["t"]) - float(a["t"])
            dist = float(np.hypot(float(b["x"]) - float(a["x"]), float(b["y"]) - float(a["y"])))
            if dt > 1.25 and dist > 150:
                jumps += 1
            elif dt > 0 and dist / dt > 300:
                jumps += 1
    teams = sorted({str(s.get("team") or f"T{s['track_id']}") for s in valid})
    cards_pose = []
    for s in valid:
        cards_pose.append(
            Pose(
                t=float(s["t"]),
                x=float(s["x"]),
                y=float(s["y"]),
                team=str(s.get("team") or f"T{s['track_id']}"),
                alliance=s.get("alliance") if s.get("alliance") in {"red", "blue"} else "unknown",
                track_id=int(s["track_id"]),
            )
        )
    cards = build_cards(cards_pose, detect_events(cards_pose))
    return {
        "n_samples": total,
        "n_valid": len(valid),
        "n_invalid": invalid,
        "edge_valid": edge,
        "corner_valid": corner,
        "teleport_jumps": jumps,
        "n_tracks": len(by),
        "teams": teams,
        "cards": [
            {
                "team": c.team,
                "alliance": c.alliance,
                "path_length_in": c.path_length_in,
                "avg_speed_in_s": c.avg_speed_in_s,
            }
            for c in cards
        ],
    }


def render_field(samples: list[dict], out_path: Path, title: str = "") -> None:
    w, h = 1040, 508
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (24, 32, 28)
    # Alliance bands
    sx = w / FIELD_LENGTH
    sy = h / FIELD_WIDTH
    cv2.rectangle(img, (0, 0), (int(158.6 * sx), h), (60, 40, 20), -1)
    cv2.rectangle(img, (int((FIELD_LENGTH - 158.6) * sx), 0), (w, h), (20, 30, 70), -1)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (80, 90, 85), 2)

    by = defaultdict(list)
    for s in samples:
        if not is_field_sample_drawable(s):
            continue
        by[int(s["track_id"])].append(s)

    for tid, rows in by.items():
        rows = sorted(rows, key=lambda r: float(r["t"]))
        alliance = rows[0].get("alliance")
        color = (80, 80, 230) if alliance == "red" else (230, 180, 80) if alliance == "blue" else (180, 180, 180)
        pts = []
        last_t = None
        last_xy = None
        for s in rows:
            x, y = float(s["x"]), float(s["y"])
            t = float(s["t"])
            if last_t is not None:
                dt = t - last_t
                dist = float(np.hypot(x - last_xy[0], y - last_xy[1])) if last_xy else 0
                if dt > 1.25 or (dt > 0 and dist / dt > 300):
                    if len(pts) >= 2:
                        cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
                    pts = []
            pts.append([int(x * sx), int(y * sy)])
            last_t = t
            last_xy = (x, y)
        if len(pts) >= 2:
            cv2.polylines(img, [np.array(pts, np.int32)], False, color, 2, cv2.LINE_AA)
        if rows:
            s = rows[-1]
            cx, cy = int(float(s["x"]) * sx), int(float(s["y"]) * sy)
            label = str(s.get("team") or tid)
            cv2.rectangle(img, (cx - 22, cy - 22), (cx + 22, cy + 22), color, -1)
            cv2.putText(img, label, (cx - 18, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 1, cv2.LINE_AA)

    if title:
        cv2.putText(img, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv2.LINE_AA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main() -> int:
    video = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/frc/validation/q70_clip90.mp4")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/data/artifacts/tracking-field-paths")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[3] if len(sys.argv) > 3 else "color"

    print(f"Analyzing layout: {video}")
    layout = analyze_video(str(video))
    ov = layout.overview_pane()
    crop_top = float(ov.crop_top) if ov else 0.02
    crop_bottom = float(ov.crop_bottom) if ov else 0.58
    print(f"overview crop {crop_top:.3f}-{crop_bottom:.3f} mode={layout.mode} conf={layout.confidence:.2f}")

    def progress(msg: str, pct: float) -> None:
        if int(pct) % 10 == 0:
            print(f"  [{pct:5.1f}%] {msg}", flush=True)

    print(f"Tracking tracker_mode={mode}…")
    result = track_video(
        video,
        crop_top=crop_top,
        crop_bottom=crop_bottom,
        tracker_mode=mode,
        use_bev=True,
        on_progress=progress,
        layout=layout,
        benchmark=False,
        frame_stride=3,
    )
    samples = result.get("samples") or []
    stats = analyze_samples(samples)
    stats["crop"] = [crop_top, crop_bottom]
    stats["warnings"] = list(result.get("warnings") or [])[:12]
    stats["source_hits"] = result.get("source_hits")
    stats["src_points"] = result.get("src_points")
    print(json.dumps({k: v for k, v in stats.items() if k != "warnings"}, indent=2))
    for w in stats["warnings"]:
        print("WARN:", w)

    stem = video.stem
    render_field(samples, out_dir / f"{stem}_field_after.png", title=f"AFTER · {stem} · {mode}")
    (out_dir / f"{stem}_stats.json").write_text(json.dumps(stats, indent=2))
    # Persist a slim sample dump for UI replay.
    slim = []
    for s in samples:
        slim.append(
            {
                "t": s.get("t"),
                "track_id": s.get("track_id"),
                "x": s.get("x"),
                "y": s.get("y"),
                "alliance": s.get("alliance"),
                "team": s.get("team"),
                "field_valid": s.get("field_valid", True),
                "gap": s.get("gap"),
            }
        )
    (out_dir / f"{stem}_samples.json").write_text(json.dumps(slim))
    print("Wrote", out_dir / f"{stem}_field_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
