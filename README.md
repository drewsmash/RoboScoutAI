# RamScoutAI

Local web app that auto-scouts an FRC **match video** from a YouTube link.

Paste a recorded FRC match VOD (not a live stream). RamScoutAI:

1. Reads **teams and scores from the video** — YouTube title/description plus the on-screen scorebug
2. Downloads the VOD and tracks robots onto that year’s field image
3. Draws each robot with a bumper PNG and its team number
4. Estimates hub dwells, defense, collection, and climb attempts from motion

[The Blue Alliance](https://www.thebluealliance.com/) is **optional**. If you paste a TBA key, nicknames and official extras are merged in. Scores from the broadcast still win when they can be read.

Action counts are **heuristics**. Confirm anything you put on a pick list.

## Desktop builds (Windows / macOS)

RamScoutAI can ship as a local desktop app that opens in your browser:

```bash
# from source
python desktop/main.py

# or package on the target OS
pip install -r requirements-desktop.txt
bash packaging/build.sh
```

GitHub Actions (`.github/workflows/release.yml`) builds:

- `RamScoutAI-windows-x64.exe`
- `RamScoutAI-macos-arm64.zip`
- `RamScoutAI-macos-x64.zip`

Publish the first downloadable builds by pushing a version tag (this creates the GitHub Release the in-app updater looks for):

```bash
git tag v0.4.2
git push origin v0.4.2
```

Until that release exists, Check for updates will say the desktop files are not published yet — that is expected. The release uploads:

- `RamScoutAI-windows-x64.exe`
- `RamScoutAI-macos-arm64.zip` (Apple Silicon Macs)

The desktop UI shows the current version and an **Update** chip when a newer GitHub Release exists. Frozen builds can download and relaunch automatically; source installs open the release page (or use `git pull`).

Desktop binaries are intentionally slim (no PyTorch). Demo mode, scorebug OCR, YouTube ingest, and field playback work out of the box. For YOLO robot tracking, run from source with `ultralytics` installed and drop a robot `.pt` in `models/`.

Optional env vars:

- `RAMSCOUT_GITHUB_REPO=owner/repo` — override the update source (default `drewsmash/RamScoutAI`)
- `RAMSCOUT_DATA=/path` — writable data directory for frozen builds


Click **Try a sample match** to explore the UI without a download.

For better tracking, drop a **robot-trained** Ultralytics `.pt` in the project folder or `models/` (for example `robot.pt`). Default COCO / RT-DETR weights will follow people and other objects — the UI warns when that happens. Weight files are gitignored; do not commit them.

A full YouTube run needs network access. A TBA auth key (`TBA_AUTH_KEY` or the form field) is optional: when present, RamScoutAI resolves the match from the title or YouTube video id, then merges nicknames, `score_breakdown`, and Zebra tracks. Event key / match key fields override auto-resolve. Without TBA, teams and scores still come from the VOD overlay.

### YouTube download (bot checks)

Datacenter / cloud IPs often get YouTube’s “sign in to confirm you’re not a bot” wall. RamScoutAI now tries several download paths:

1. **yt-dlp** with modern player clients (`tv`, `mweb`, …) and the **WebPoClient PO-token** plugin (`yt-dlp-getpot-wpc`) when Chrome/Chromium is available  
2. Your **`YTDLP_PROXY`** / `HTTPS_PROXY` residential proxy (recommended for servers)  
3. An optional **auto-proxy ladder** (`YTDLP_AUTO_PROXY=1`, default on) for flaky public HTTP proxies when everything else is blocked  
4. **Upload a local VOD** in the UI (or pass a local file path) — skips YouTube entirely

Also supported: `YTDLP_COOKIES=/path/to/cookies.txt` or `YTDLP_BROWSER=chrome` on a signed-in machine. Title metadata still loads via oEmbed when the file download is blocked.

## Next year’s game

Game art and timing live in year files, not in the UI:

1. Copy [`ramscout/games/2026.json`](ramscout/games/2026.json) to `ramscout/games/2027.json`
2. Add a top-down field PNG at [`web/fields/2027.png`](web/fields/2026.png) (match the real field aspect ratio, about 2:1)
3. Update `length_in`, `width_in`, auto/endgame timing, overlay crop bands, and `landmarks` so Hub/Tower dots sit on the new art
4. Review [`ramscout/events.py`](ramscout/events.py) and [`ramscout/field.py`](ramscout/field.py) for new scoring locations (Hub/Tower names will change)

The YouTube title year selects the config automatically (`2027 … Qualification Match 4` → `2027.json`). If that year is missing, the newest shipped layout is used.

Robot icons are shared: [`web/robots/blue.png`](web/robots/blue.png) and [`web/robots/red.png`](web/robots/red.png). Team numbers are stamped on the bumper plates at draw time.

## Tests

```bash
pytest -q
```

## Layout

- [`app.py`](app.py) — FastAPI server
- [`ramscout/`](ramscout/) — ingest, overlay OCR, TBA merge, tracking, events
- [`ramscout/games/`](ramscout/games/) — per-year field config
- [`web/fields/`](web/fields/) — field PNGs
- [`web/robots/`](web/robots/) — robot icon PNGs
- [`legacy/frc_live_scout.py`](legacy/frc_live_scout.py) — original PyQt desktop experiment

## Limits

- Wide, mostly fixed cameras work best
- Scorebug OCR needs a readable graphic; title/description is the backup
- Climb detection is “stayed on the Tower in endgame”, not rung level
- Hub candidates are low-speed dwells next to the Hub, not counted fuel
