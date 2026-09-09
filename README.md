# RamScoutAI

Local web app that auto-scouts an FRC **match video** from a YouTube link.

Paste a recorded FRC match VOD (not a live stream). RamScoutAI:

1. Reads the video title and looks up the match on [The Blue Alliance](https://www.thebluealliance.com/)
2. Downloads the video
3. Tracks robots onto a 2026 REBUILT field map
4. Estimates hub dwells, defense, collection, and climb attempts from motion

Action counts are **heuristics**, not official TBA scores. Confirm anything you put on a pick list.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

Optional: set `TBA_AUTH_KEY` or paste a key in the UI ([TBA account page](https://www.thebluealliance.com/account)).

For better tracking, put a **robot-trained** Ultralytics `.pt` in the project folder or `models/` (for example `robot.pt`). Without it, the app still shows TBA data; COCO/RT-DETR weights will not reliably see FRC robots.

Click **Try a sample match** to explore the UI with simulated REBUILT paths (Team 59 included).

## Tests

```bash
pytest -q
```

## Layout

- [`app.py`](app.py) — FastAPI server
- [`ramscout/`](ramscout/) — ingest, TBA, tracking, geometry, events
- [`web/`](web/) — dark Material-inspired UI
- [`legacy/frc_live_scout.py`](legacy/frc_live_scout.py) — original PyQt desktop experiment

## Limits

- Wide, mostly fixed cameras work best
- Team identity starts from bumper color + starting stations; OCR is best-effort
- Climb detection is “stayed on the Tower in endgame”, not rung level
- Hub candidates are low-speed dwells next to the Hub, not counted fuel
