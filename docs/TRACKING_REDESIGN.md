# Tracking redesign — starting findings (audit)

Code audit of claimed tracking bugs in RoboScoutAI. Each item is
**CONFIRMED** (with `file:line`), **HYPOTHESIS**, or **FALSE**. Scope:
`ramscout/detect.py`, `identity.py`, `pipeline.py`, `trackers/alliance.py`,
`trackers/mot.py`, `trackers/selection.py`, `depth.py`, `bev.py`, `web/app.js`.

Related: `docs/TRACKING.md` (current pipeline description).

---

## Claim verdicts

### 1. Unknown alliance labels inferred from current field half — **CONFIRMED**

When the voter still returns `"unknown"`, the tracking loop assigns alliance
from the **current** field-mapped `x` vs mid-field — not starting zone, not
early-match gated:

```579:581:ramscout/detect.py
            if alliance == "unknown":
                alliance = "blue" if float(mx) < FIELD_LENGTH * 0.5 else "red"
                alliance_conf = 0.3
```

Same half-field fallback appears in team assignment and majority vote:

```84:87:ramscout/identity.py
        alliance = sample.get("alliance") if sample.get("alliance") in {"red", "blue"} else None
        if alliance is None:
            alliance = "blue" if float(sample["x"]) < FIELD_LENGTH / 2 else "red"
```

```236:240:ramscout/identity.py
def majority_alliance(samples: list[dict[str, Any]]) -> str:
    votes = Counter(s.get("alliance") for s in samples if s.get("alliance") in {"red", "blue"})
    if not votes:
        xs = [float(s["x"]) for s in samples]
        return "blue" if (sum(xs) / max(len(xs), 1)) < FIELD_LENGTH / 2 else "red"
```

`side_prior` (alliance zone, early-only) exists in `trackers/alliance.py:300–323`
but is only used as a voter prior / balance cost — not as the unknown fallback.

**Impact:** Midfield robots and late-match crossings get hard red/blue labels
with conf 0.3; those labels then feed `balance_alliances` and team assignment.

---

### 2. Alliance changes create new output track IDs — **CONFIRMED**

On voter flip red↔blue, the MOT id is aliased to a new output id (`50000+`):

```572:578:ramscout/detect.py
            prev_label = s.emitted_label.get(tid)
            if prev_label in {"red", "blue"} and alliance in {"red", "blue"} and alliance != prev_label:
                split_count += 1
                s.tid_alias[tid] = 50000 + split_count
            if alliance in {"red", "blue"}:
                s.emitted_label[tid] = alliance
            out_tid = s.tid_alias.get(tid, tid)
```

**Impact:** One physical robot can appear as two scout tracks / team cards when
bumper color flips (or half-field fallback flips). Downstream stitching may or
may not rejoin them (`stitch_occlusions` / `assemble_lanes`).

---

### 3. Multiple stages rebalance into 3 red / 3 blue — **CONFIRMED**

Forced 3+3 (or match roster size) runs at least three times:

| Stage | Location |
| --- | --- |
| End of `track_video` | `detect.py:634` — `balance_alliances(samples)` |
| After stitch / lanes / keep_top | `pipeline.py:617–619` — `assemble_lanes` then `balance_alliances` |
| On recalibrate / potato merge | `pipeline.py:862` — `balance_alliances` again |

Core solver: `identity.balance_alliances` → `alliance.assign_alliance_slots`
(`identity.py:158–233`, `alliance.py:326–361`) — Hungarian assignment onto
exactly `n_red` / `n_blue` slots.

**Impact:** Color evidence is overwritten to satisfy roster geometry; repeated
passes can fight OCR / side-view cues and hide true 4v2 or mis-detect counts.

---

### 4. Tracking loop overwrites initial depth with `estimate_depth(..., prefer_neural=False)` — **CONFIRMED**

Session build calibrates BEV with `prefer_neural=prefer_depth_neural` (default
True):

```363:370:ramscout/detect.py
                bev_cal, depth_cal = calibrate_bev(
                    frame,
                    ...
                    prefer_neural=prefer_depth_neural,
```

Every `depth_stride` frames (default 12), the live depth map is replaced with
**classical-only** depth:

```515:518:ramscout/detect.py
        if use_bev and s.frames % depth_refresh == 0:
            try:
                s.depth = estimate_depth(cropped, prefer_neural=False).depth
                s._holder["depth"] = s.depth  # type: ignore[attr-defined]
```

`estimate_depth` with `prefer_neural=False` skips ONNX/torch and always uses
`_classical_depth` (`depth.py:190–225`).

**Impact:** Foot refinement (`refine_foot_point`) after the first refresh uses
the classical prior even when neural depth was available at calibration.

---

### 5. Camera pitch / corners from heuristic depth-derived trapezoid — **CONFIRMED** (with field-line override)

Primary BEV path: depth → pitch/tilt → `_tilt_adjusted_norm` trapezoid →
optional carpet-quad replace/blend (`bev.py:53–133`, method default
`"depth_trapezoid"`).

Pitch itself always comes from `_pitch_from_depth` on whatever depth map was
produced (neural or classical). When neural fails or is disabled, classical
row/texture depth drives pitch (`depth.py:214–225`, `bev.py:71–73`,
`198–225`).

Field lines can replace corners when confidence ≥ 0.55 and perspective agrees
(`bev.py:105–108`); otherwise the depth trapezoid (or blend) remains.

**Impact:** Homography often depends on a geometric prior rather than true
carpet corners unless field-line detection wins; classical refresh (claim 4)
does not recompute corners, only feet depth.

---

### 6. Frontend treats unknown alliance as blue — **CONFIRMED**

```766:767:web/app.js
    const alliance = sample.alliance === "red" ? "red" : "blue";
    const color = alliance === "red" ? "#f28b82" : "#8ab4f8";
```

Same pattern for field bots / seeds / cards:

- `web/app.js:869` — `samples[0]?.alliance || "blue"`
- `web/app.js:933` — `state.bots[alliance] || state.bots.blue`
- `web/app.js:454` — non-`"red"` → blue robot icon
- `web/app.js:1058` — seed picker defaults non-red to blue

Backend often never emits `"unknown"` after claim 1, so the UI default mainly
covers missing/empty alliance strings.

---

### 7. Path colors change by match period — **CONFIRMED**

```959:975:web/app.js
function drawPath(ctx, X, Y, samples, t, color, autoEnd, endgameStart) {
  ...
  const period = t < autoEnd ? "#fdd663" : t >= endgameStart ? "#81c995" : color;
  ctx.strokeStyle = period;
```

Whole path stroke uses **playback time** `t` vs auto/endgame thresholds, not
per-sample alliance color for the entire trail. Auto period → yellow;
endgame → green; teleop → alliance color.

**Note:** intentional UX for period, but reads as “path color changes” and
overrides alliance identity during auto/endgame.

---

### 8. Desktop robot detection depends on optional Ultralytics — **CONFIRMED**

```27:33:ramscout/trackers/yolo.py
def ultralytics_available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False
```

YOLO strategy only loads when importable (`yolo.py:105+`, `175`, `187`). Auto
mode only benchmarks YOLO if available (`selection.py:64–67`). Pipeline warns
to `install ultralytics` when no neural/cloud hits (`pipeline.py:627–629`).
Fallback stack is motion / color / flow (and optional Gemini/OpenAI).

---

### 9. Auto strategy selection rewards count/balance without GT — **CONFIRMED**

`select_tracker_mode` benchmarks candidates on the first ~12 s with no
ground-truth poses or team labels (`selection.py:190–315`). Score weights
(`selection.py:44–52`, `93–187`):

| Term | Weight | Meaning |
| --- | --- | --- |
| `robots` | 0.22 | mean visible ≈ 6 |
| `balance` | 0.14 | 3 red / 3 blue per time bin |
| `persist` | 0.16 | track lifetime |
| `in_field` | 0.12 | inside carpet |
| `speed` | 0.12 | physical speed prior |
| `moving` | 0.14 | non-static |
| `stability` | 0.10 | few flips / births |

No GT / TBA zebra / OCR agreement term. Modes that invent balanced 6-robot
noise can beat sparse-but-correct detectors.

---

### 10. Detection runs on full frame vs overview crop only — **FALSE** (overview crop owns tracks)

Main tracking detects on the **overview pane crop**, not the full frame:

```513:535:ramscout/detect.py
        s = active
        cropped = frame[s.y0 : s.y1, s.x0 : s.x1]
        ...
        detections = s.ensemble.detect(cropped, ctx)
```

Layout timeline pauses when overview is off-air (`detect.py` gaps;
`docs/TRACKING.md` §1). Bboxes are remapped to full-frame coords for overlay
(`detect.py:601`).

**Related (not the claim):** Bottom / side panes run a **separate** detect pass
(`multiview.detect_all_panes`, `pipeline.py:632–702`) for correlation and
scoring/climb cues — they do **not** own top-down robot path samples.

---

## Context notes (not numbered claims)

### Current crop UX

- Advanced form: numeric **Crop top** / **Crop bottom** (`web/index.html:112–124`),
  defaults 0.10 / 0.65; **Auto multi-view crop** checkbox (default on).
- Auto layout (`multicam.apply_layout`) overwrites job crop from overview pane
  (`pipeline.py:513–573`).
- Calibration: click four field corners on `#cal-canvas`
  (`web/app.js:204–217`, `1078–1080`); posts `src_points`. User calibration
  disables BEV auto corners (`use_bev=not job.user_calibrated`,
  `pipeline.py:605`).

### Do bottom panes feed robot tracks?

**No for BEV paths.** Overview `track_video` samples are the path source.
Side/alt panes → `pane_detections` + `side_cues` (hub/climb activity, team
hints via `correlate_views` / `attach_teams_to_cues`). They enrich events /
confidence, not the six field trajectories.

### Coordinate transform helpers

| Helper | Role |
| --- | --- |
| `geometry.homography_from_corners` | 4 pixel corners → field inches |
| `geometry.project_points` | pixel feet → field `(x, y)` |
| `geometry.default_source_points` | default trapezoid in crop band |
| `geometry.reproject_samples` | re-map stored `(px, py)` after corner edit |
| `geometry.crop_bounds` | fractional crop → pixel rows |
| `bev.calibrate_bev` / `project_detections_bev` | depth-aware corners + project |
| `depth.refine_foot_point` | bbox → ground contact using depth |
| `FieldGate` (`trackers/field_gate.py`) | accept/reject detections in field |

Convention: field `x = 0` blue wall, `x = FIELD_LENGTH` red (`alliance_orientation`
/ `orient_corners` may mirror).

---

## Redesign starting points

1. **Never invent alliance from live field half** — keep `"unknown"` until
   calibrated bumper vote / side prior (early only) / scout override.
2. **Do not mint new track IDs on alliance flip** — flip label in place; treat
   flips as confidence events.
3. **Single balance pass** (or scout-confirmed) after identity is stable;
   avoid detect → pipeline → recalibrate triple force-3v3.
4. **Keep neural depth in the loop** when calibration used it; refresh with
   `prefer_neural=True` (or freeze calibration depth for feet).
5. Prefer **field-line / user corners** over classical trapezoid; treat depth
   pitch as a prior, not the sole corner source.
6. Frontend: render unknown distinctly; path stroke should not erase alliance
   identity (period as dash/alpha, not recolor).
7. Auto-select: add GT/zebra agreement or hold-out consistency; down-weight
   pure count/balance when no detector evidence.
8. Keep overview-only ownership of paths; side panes stay cue/correlation only
   unless a deliberate multi-view fusion redesign.

---

## Evidence index

| # | Verdict | Primary evidence |
| --- | --- | --- |
| 1 | CONFIRMED | `ramscout/detect.py:579–581`, `identity.py:84–87`, `236–240` |
| 2 | CONFIRMED | `ramscout/detect.py:572–578` |
| 3 | CONFIRMED | `detect.py:634`, `pipeline.py:617–619`, `862`, `identity.py:158+` |
| 4 | CONFIRMED | `detect.py:363–370` vs `515–518`, `depth.py:190–225` |
| 5 | CONFIRMED | `bev.py:53–133`, `198–225`, `depth.py:214–225` |
| 6 | CONFIRMED | `web/app.js:766–767`, `869`, `933`, `454` |
| 7 | CONFIRMED | `web/app.js:959–975` |
| 8 | CONFIRMED | `trackers/yolo.py:27–33`, `selection.py:64–67`, `pipeline.py:627–629` |
| 9 | CONFIRMED | `trackers/selection.py:44–52`, `93–187`, `190–315` |
| 10 | FALSE | `detect.py:513–535` (crop); side panes separate (`pipeline.py:632–702`) |
