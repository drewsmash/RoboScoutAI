# RoboScoutAI — install & dependency map

## Quick start (source)

```bash
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
python -m ramscout.deps --strict
uvicorn app:app --reload --port 8765
```

Desktop / freeze (no Ultralytics / torch):

```bash
python -m pip install -r requirements-desktop.txt
python -m ramscout.deps --strict
python desktop/main.py
```

## Requirement files

| File | When to use |
| --- | --- |
| `requirements.txt` | Full **source** install: API, OpenCV tracking, ONNX Runtime, Ultralytics, tests |
| `requirements-desktop.txt` | Slim **desktop / PyInstaller** set (includes `onnxruntime`, excludes Ultralytics/torch) |
| `requirements-depth.txt` | Optional torch + transformers Depth Anything backend |
| `requirements-laya.txt` | Optional local Laya scout verification |

## Required at runtime

These must import for the app to run tracking locally:

- `fastapi`, `uvicorn`, `python-multipart`, `pydantic`, `httpx`
- `numpy`, `scipy`, `opencv-python-headless`, `Pillow`
- `yt-dlp`, `openpyxl`
- **`onnxruntime`** — neural depth **and** the packaged robot ONNX detector path

Verify:

```bash
python -m ramscout.deps --strict
# or
curl -s http://127.0.0.1:8765/api/health | python -m json.tool
```

## Optional

| Package | Unlocks |
| --- | --- |
| `ultralytics` | YOLO training/export + Ultralytics track (source only; not in desktop freeze) |
| `torch` + `transformers` | Alternate Depth Anything backend (`requirements-depth.txt`) |
| `laya` | Local System-1 scout verification (`requirements-laya.txt`) |
| `pywebview` | Chrome-less desktop window (`requirements-desktop.txt`) |
| `nodriver` | Extra YouTube download assistance (source) |

## Model weights (not pip)

| Asset | Purpose | Required? |
| --- | --- | --- |
| Depth Anything V2 Small ONNX | Neural depth | Auto-downloaded on first use into the models dir |
| `models/robot.onnx` (+ `.meta.json`) | Local FRC robot detector | **Not claimed FRC-ready until `frc_ready: true`** |

See `docs/DETECTOR_TRAINING.md`.

## Platform notes

- **Windows / Apple Silicon:** `onnxruntime` in the requirement files is the supported EP path (CPU; DirectML/CoreML when the wheel provides them). Report the active provider via `/api/trackers` → `detector.provider` / depth backends.
- Desktop freezes collect `onnxruntime` + `cv2` + `scipy` in `packaging/app.spec`.
- Do not install Ultralytics into the frozen app; train elsewhere and export ONNX.

## CI

Release tests install `requirements-desktop.txt` + `pytest`. Source contributors should run `pip install -r requirements.txt` before local YOLO work.
