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

Desktop binaries are intentionally slim (no PyTorch / `ultralytics`). Demo mode, scorebug OCR, YouTube ingest, field playback, and **multi-strategy OpenCV tracking** still work — robot paths are approximate without YOLO/cloud. Tracking modes:

| Mode | What it uses |
| --- | --- |
| `auto` | YOLO (if installed) + bumper color + motion + optical flow + OpenAI/Gemini when keyed |
| `local` | Offline only (YOLO optional + OpenCV) |
| `motion` / `color` | No neural net |
| `openai` / `gemini` / `cloud` | Cloud vision keyframes + local fill between frames |

API keys (optional, also in the UI Advanced panel):

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`

For stronger local neural detection from source:

```bash
pip install -r requirements.txt   # includes ultralytics
# optional: drop a robot-trained .pt in models/
```

### App icons & code signing

Desktop builds use [`web/icons/app.ico`](web/icons/app.ico) / [`web/icons/app.icns`](web/icons/app.icns). The web UI ships a favicon, apple-touch icon, and PWA manifest.

Unsigned Windows/macOS downloads often trigger SmartScreen or Gatekeeper. See **[`docs/SIGNING.md`](docs/SIGNING.md)** for:

- How to run unsigned builds safely
- GitHub Actions secrets for Authenticode + Apple Developer ID + notarization
- Local `signtool` / `codesign` commands

The release workflow signs automatically when those secrets are present and ships unsigned (with a warning) when they are not.

### Pick list & scout book

After analyzing matches, use **Pick list** in the app to:

- Accumulate robots into a local scout book
- Rank draft picks (1st / 2nd / 3rd round suggestions)
- Compare alliance combinations
- Keep team notes and a watchlist (stored in the browser)

Optional env vars:

- `RAMSCOUT_GITHUB_REPO=owner/repo` — override the update source (default `drewsmash/RamScoutAI`)
- `RAMSCOUT_GITHUB_TOKEN` / `GITHUB_TOKEN` — required for in-app updates when the GitHub repo is **private**
- `RAMSCOUT_DATA=/path` — writable data directory for frozen builds

Because this repository is private, open the release while signed into GitHub to download:

https://github.com/drewsmash/RamScoutAI/releases/tag/v0.4.2


Click **Try a sample match** to explore the UI without a download.

For better tracking, drop a **robot-trained** Ultralytics `.pt` in the project folder or `models/` (for example `robot.pt`). When Ultralytics is installed and no local weights are present, RamScoutAI can auto-download YOLO nano. When Ultralytics is missing (desktop builds), OpenCV motion / background-subtraction tracking still fills the path overlay so scout cards are not empty. Default COCO weights may follow people/vehicles — the UI warns when that happens.

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
