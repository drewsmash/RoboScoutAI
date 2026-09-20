# Tracking, 3D and broadcast decomposition (0.6.1)

This document describes what happens between "video file" and "six robot
paths on the field map", which knobs exist, and how well it works on real
broadcast footage.

## 1. Broadcast decomposition (`ramscout/layout.py`, `ramscout/multicam.py`)

FRC VODs are compositions: an overview camera, one or two close-up cameras,
a scorebug, sometimes a fisheye "alt overview", plus pre/post-match graphics.
Tracking only makes sense on the overview pane, and only while it is on air.

1. `sample_frames` pulls ~140 frames across the VOD (every 2 s).
2. `detect_boundaries` finds horizontal / vertical seams from edges that are
   present in most frames (coverage), continuous along the line
   (continuity), and *uncorrelated across the line over time*
   (`temporal_crossing`: pixels either side of a real composition seam come
   from different cameras, pixels either side of a field wall co-vary).
   Letterbox / pillarbox bars are detected first and excluded.
3. `detect_scorebug` finds the static, high-contrast, digit-bearing region.
4. `_classify_panes` scores each pane box: carpet-grey dominance, red/blue
   bumper mass, texture, temporal liveliness, scorebug overlap, position →
   `overview`, `overview_alt`, `blue_side`, `red_side`, `sideline`,
   `graphics`.
5. `analyze_video_layout` groups frames by layout signature (with windowed
   signatures and blip absorption) into a `LayoutTimeline` of segments.

`multicam.analyze_video` converts the primary segment into the `CameraLayout`
the pipeline already used (`method="decomposition"`, per-pane `confidence`,
`scorebug`, `timeline`) and falls back to the legacy heuristics when the
decomposition is not confident.

`detect.track_video` walks the timeline: one `_PaneSession` (homography,
field gate, ensemble, alliance calibrator) per distinct overview crop, panes
that reappear with the same geometry resume their session, and segments
without an overview are recorded as `gaps` (no samples are emitted).

## 2. Depth → BEV (`ramscout/depth.py`, `ramscout/bev.py`, `ramscout/fieldlines.py`)

Backend chain, first that works wins; the status is exposed by
`depth_backends()`:

| backend | what | needs |
| --- | --- | --- |
| `onnx` | Depth Anything V2 Small, ONNX Runtime | `onnxruntime` (in `requirements.txt` and the desktop spec); weights (~100 MB) auto-download once to the models dir |
| `torch` | Depth Anything V2 Small via `transformers` | `pip install -r requirements-depth.txt` (CPU wheels: `--index-url https://download.pytorch.org/whl/cpu`) |
| `classical` | row/texture prior | nothing |

`ROBOSCOUT_NO_DOWNLOAD=1` disables the weight download (frozen builds in a
sandbox, tests).

`calibrate_bev` takes the depth pitch/tilt and builds the trapezoid as
before, then (new) asks `fieldlines.detect_field_quad` for the carpet
boundary of the pane:

- carpet model: Lab colour learned from smooth, unsaturated pixels of the
  pane's central band (or a robot-motion seed), thresholds adapted from the
  carpet's own spread; the mask is `dist < thr & sat < 95 & texture < thr`,
  cut along strong edges so grey bleacher steps do not merge with the carpet
- components touching the central band are unioned and hole-filled
- four straight edges are fitted with iterative MAD trimming to the row-wise
  left/right and column-wise top/bottom extents; corners are the line
  intersections
- confidence combines fill ratio, edge inlier ratio, quad area and the
  near/far width ratio; the quad replaces the depth trapezoid when
  `confidence ≥ 0.55` and its perspective agrees with the depth pitch, and
  nudges it when `≥ 0.4`

`alliance_orientation` decides blue-left vs blue-right from thin saturated
structures on the carpet (alliance tape / zone outlines — thick blobs are
plates whose colours are randomised in some seasons) and thick saturated
blobs just outside the carpet's left/right edges (driver-station panels).
`orient_corners` mirrors the homography so field `x = 0` is always the blue
wall.

## 3. Proposals (`ramscout/trackers/motion.py`, `color.py`, `flow.py`, `yolo.py`, cloud)

- **Field polygon mask**: `detect.field_mask_for_crop` back-projects the
  carpet rectangle (+14 in) through the homography and lifts the far edge by
  a robot height; motion and colour masks are ANDed with it. Crowd, referees
  and driver stations cannot become proposals.
- **Motion**: MOG2 + KNN + frame-diff blobs, size / aspect / fill filters.
- **Bumper colour**: red/blue masks → *thin* strips (`h ≤ 0.6 w`) with
  something standing on them (`body_differs_from_surroundings`) →
  `bumper_to_robot_box` grows the strip into the robot box. The lowest
  colour band wins over decoys above it.
- **YOLO**: `ROBOSCOUT_ROBOT_WEIGHTS_URL` / `ROBOSCOUT_ROBOT_WEIGHTS` point at
  FRC-tuned weights (a model with a `robot` class) → true detector. COCO
  weights are *verify-only*: their boxes raise the confidence of overlapping
  local proposals (`verify_proposals`) and never spawn tracks.
- **Cloud** (Gemini / OpenAI Vision): sparse keyframes when keyed.

## 4. Field gate + MOT (`ramscout/trackers/field_gate.py`, `mot.py`)

The gate projects each box's depth-corrected foot to field inches and
rejects outside-field, perimeter-wall, footprint < 16 in or > 96 in,
scorebug and multi-edge boxes; it flags static blobs (long-window motion
energy + optical flow) and stores `raw_confidence`, `footprint_in`,
`perimeter`, `field_xy` in the detection meta for the tracker.

`MotTracker` is SORT with:

- **ByteTrack two-stage association**: high-score proposals (`≥ 0.40`, using
  the raw confidence so a robot on the perimeter is not demoted) are matched
  to all tracks; low-score leftovers (`0.10–0.40`) may only extend tracks
  seen in the last 3 steps and never spawn.
- **OC-SORT**: on recovery after coasting the velocity is rebuilt from the
  observed displacement over the gap (80 %), and a momentum term penalises
  proposals whose direction from the last observation disagrees with the
  track.
- **BEV Kalman** in inches with a 20 ft/s association limit.
- **Parked-robot rule**: bumper-derived boxes with a robot-sized footprint
  confirm after 12 colour hits without moving; walls fail the footprint
  test.
- **Static pruning by excursion**: tracks whose maximum excursion from birth
  stays tiny are walls; tracks with lots of path but no excursion are
  plates wiggling in place. Pruned spots become **dead zones** (no spawn for
  20 s; 150 s once the same spot has recurred three times).
- Internal pool of 14 tracks (8 emitted) so tentative real robots are not
  evicted by noise.

## 5. Post-processing (`ramscout/identity.py`, `smoothing.py`, `ocr.py`)

1. `stitch_occlusions` chains tracklets (gap ≤ 3 s, distance ≤ 48 in +
   60 in/s × gap, same alliance, overlap trimming).
2. `assemble_lanes` assigns every time-disjoint tracklet to one of six lanes
   (3 red / 3 blue); an implausible jump is accepted but marked `lane_jump`;
   a tracklet overlapping all three lanes of its alliance is a seventh object
   and is dropped.
3. `balance_alliances` forces 3 + 3.
4. `smooth_samples` (RTS smoother, constant-velocity model) writes `x`, `y`,
   `speed_in_s` and keeps `x_raw`, `y_raw`.
5. `BumperOCR` reads team numbers from stable tracks (Tesseract →
   `cv2.text` → OpenAI Vision), normalises confusables (O→0, S→5, B→8),
   votes, and assigns each team once.

## 6. Auto-benchmark mode (`ramscout/trackers/selection.py`)

`tracker_mode="auto"` runs `_track_video_impl` for each candidate mode on
`[3 s, 15 s]` (or the pinned window), reuses the first BEV calibration, and
scores:

| term | weight | meaning |
| --- | --- | --- |
| robots | 0.22 | mean visible robots vs six, share of time with ≥ 5 |
| balance | 0.14 | per-time-bin \|red−3\| + \|blue−3\| |
| persist | 0.16 | mean track lifetime / window |
| in_field | 0.12 | samples well inside the carpet |
| speed | 0.12 | steps under 20 ft/s, p95 penalty |
| moving | 0.14 | tracks that actually moved |
| stability | 0.10 | alliance flips, ID births per second |

Candidates are de-duplicated by effective strategy set (no key → no cloud;
no `ultralytics` → no YOLO). The result (`chosen`, `scores`, `ranking`,
`timings_s`, `window`) is stored as `tracker_selection`.

## 7. Real-footage results (Chezy Champs 2018 Q70, archive.org, 1280×720 @ 60 fps)

Full pipeline, `auto` → chose `color` (score 0.53 vs motion 0.45, yolo-COCO
0.45, hybrid 0.37) on the 12 s benchmark; then 140 s of match at 10 Hz in
~42 s of CPU time.

| metric | raw tracker | after chaining + lanes |
| --- | --- | --- |
| tracks | 293 tracklets, median 1.4 s | 6 lanes, median 139 s |
| mean visible robots | 6.3 (includes duplicates) | 4.1 of 6 |
| ≥ 5 robots visible | — | 34 % of the match |
| in-field samples | — | 92 % |
| perimeter band | — | 16 % |
| alliance flips | 0 | 0 |
| p95 speed | — | 99 in/s (8 ft/s) |

Decomposition: `stacked_top`, seam at 55.6 %, overview 0.84 / alt-overview
0.52, scorebug found, 4 layout segments (pre-match graphics → main → two
post-match single-pane cuts), overview on air 88 % of the VOD. Depth:
`depth_anything_v2_onnx`, pitch ≈ 46°. Homography from the carpet quad
(confidence 0.63), blue wall detected on the right → mirrored.

What still goes wrong on this footage:

- 2018 switch / scale plates are alliance-coloured, robot-sized, inside the
  field and move in place; they still create short false tracklets and
  occasional lane pollution.
- Robots parked against their own alliance wall are the weakest case (the
  perimeter penalty + bumper strip merges with the wall).
- Raw tracklets are short (blob merges when robots touch); the lane
  assembly recovers full-match paths but a wrong link shows up as a straight
  jump (drawn faint in the BEV plot, `lane_jump` in the sample).

The real fix for the first two is a learned FRC robot detector; the code
path is there (`ROBOSCOUT_ROBOT_WEIGHTS_URL`), the weights are not shipped.
