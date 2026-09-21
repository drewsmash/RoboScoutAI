# FRC robot detector — annotation & training workflow

**Status:** Workflow shipped; **no verified FRC-ready ONNX checkpoint is packaged**.

Do not claim the packaged app detects FRC robots with a trained model until
`models/robot.onnx.meta.json` sets `"frc_ready": true` and the SHA matches.

## Goals

- One class: `robot` (alliance / identity estimated separately).
- Train on wide broadcast overview crops (top pane), not pit close-ups alone.
- Include parked, occluded, far-side small, blurred robots.
- Negatives: referees, walls, game pieces, driver stations, carpet tape, lights.

## Split rule

Split by **match / event**, never by adjacent frames from the same clip.

## Tooling

1. Label with any YOLO-compatible tool (Roboflow, CVAT, Label Studio).
2. Train with Ultralytics (dev machine only):

```bash
yolo detect train data=data/frc_robots.yaml model=yolo11n.pt epochs=100 imgsz=640
yolo export model=runs/detect/train/weights/best.pt format=onnx simplify=True
```

3. Optional YOLOX comparison: train separately; only ship if TrackEval on held-out
   events beats the Ultralytics export on HOTA/IDF1 at equal runtime.

4. Copy artifacts into the app models dir:

```
models/robot.onnx
models/robot.onnx.meta.json
```

Example meta:

```json
{
  "version": "0.1.0",
  "class_names": ["robot"],
  "input_size": [640, 640],
  "sha256": "<sha256 of robot.onnx>",
  "license": "AGPL-3.0 (Ultralytics) — replace if using a different stack",
  "source": "internal FRC broadcast labels · event-split eval",
  "frc_ready": true,
  "notes": "Trained on overview crops only; evaluated HOTA on held-out events."
}
```

## Runtime

The packaged app loads ONNX via `ramscout.onnx_detector` + ONNX Runtime
(CoreML / DirectML / CUDA when available, else CPU). Ultralytics is **not**
required at runtime.

Query status: `GET /api/trackers` → `detector`.
