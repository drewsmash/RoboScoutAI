# Tracking redesign — audit & plan

**Starting commit:** `4a67ad940ec653cf544680aad2e256575ce21f3e`  
**Branch:** `cursor/tracking-ui-redesign-755d` (work began on `cursor/tracking-redesign-audit-678c`)  
**Date:** 2026-09-21

## Confirmed failures (code evidence)

| # | Behavior | Verdict | Evidence |
|---|----------|---------|----------|
| 1 | Unknown alliance inferred from field half | **CONFIRMED** → **FIXED** | Was `detect.py` / `identity.py`; unknown stays unknown; soft field-half team fill removed |
| 2 | Alliance flip creates new output track ID | **CONFIRMED** → **FIXED** | Conflict flagged; confirmed alliance retained |
| 3 | Multiple 3R/3B rebalances | **CONFIRMED** | Still present in legacy balance helpers; IdentityBook does not force six tracks |
| 4 | Loop overwrites neural depth with classical | **CONFIRMED** → **FIXED** | Depth refresh preserves backend provenance |
| 5 | Pitch/corners from depth trapezoid | **CONFIRMED** | `bev.py` heuristic remains; field-quad calibration preferred when user-provided |
| 6 | Frontend unknown → blue | **CONFIRMED** → **FIXED** | `web/js/theme/alliance.js` + `app.js` / potato |
| 7 | Path colors change by match period | **CONFIRMED** → **FIXED** | Paths use stable alliance color; period is a separate band |
| 8 | Desktop detect needs optional Ultralytics | **CONFIRMED** → **IN PROGRESS** | `ramscout/onnx_detector.py` + training workflow; no FRC-ready weights yet |
| 9 | Auto-select rewards count/balance without GT | **CONFIRMED** | `trackers/selection.py` unchanged this pass |
| 10 | Detection on full frame | **FALSE** | Detects on overview crop; side panes are cues only |

## Design decisions for this redesign

1. **Top-overview-only** is the default tracking mode. Side panes never create robot identities.
2. **Unknown stays unknown** (neutral gray). Half-field inference is deleted.
3. **Alliance flip flags association error** — it does not mint a new track ID or recolor a confirmed identity.
4. **Team from starting order is labeled `assignment_source: start_pose_unverified`** when used; default path no longer soft-fills unknowns into roster slots from x-position.
5. **Path strokes use stable alliance colors**; period is a separate timeline / band indicator.
6. **Depth refresh preserves the active backend** (no silent classical overwrite).
7. **Coordinate transforms** are centralized in `ramscout/crop.py`.

## Delivery stages

1. Baseline fixtures + this audit — **done**
2. Crop lock + coordinate correctness — **in progress** (module + API + UI lock)
3. Packaged local FRC detector (ONNX) — **workflow + runtime hook**; weights unvalidated
4. Stable identities + consistent rendering — **in progress**
5. Validated calibration + depth comparison — pending
6. Selective repair + review UI — pending (Next uncertain / Correct team stubs)
7. Full app UI redesign (design system, Analysis Studio, sidebar) — **shell landed**
8. Benchmark report — pending

## Known gaps / unvalidated

- No verified FRC-trained YOLO/YOLOX checkpoint is packaged yet. See `docs/DETECTOR_TRAINING.md`.
- TrackEval HOTA/IDF1 on labeled clips requires representative VODs not present in-repo.
- Depth Anything accuracy benefit vs homography-alone is not yet measured on real footage in this branch.
- Do not declare tracking solved from synthetic tests or six plausible-looking dots alone.
