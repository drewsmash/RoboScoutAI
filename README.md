# RoboScoutAI

Local web app that auto-scouts an FRC **match video** from a YouTube link.

Paste a recorded FRC match VOD (not a live stream). RoboScoutAI:

1. Reads **teams and scores from the video** — YouTube title/description plus the on-screen scorebug
2. Downloads the VOD and tracks robots onto that year’s field image
3. Draws each robot with a bumper PNG and its team number
4. Estimates hub dwells, defense, collection, and climb attempts from motion

[The Blue Alliance](https://www.thebluealliance.com/) is **optional**. If you paste a TBA key, nicknames and official extras are merged in. Scores from the broadcast still win when they can be read.

Action counts are **heuristics**. Confirm anything you put on a pick list.

## Desktop builds (Windows / macOS)

RoboScoutAI can ship as a local desktop app that opens in its own chrome-less window (no URL bar):

```bash
# from source
pip install -r requirements-desktop.txt
python desktop/main.py

# or package on the target OS
bash packaging/build.sh
```

Use `--browser` for a normal browser tab, or `--no-browser` for server-only.

GitHub Actions (`.github/workflows/release.yml`) builds:

- `RoboScoutAI-windows-x64.exe`
- `RoboScoutAI-macos-arm64.zip`
- `RoboScoutAI-macos-x64.zip`

### In-app updater (git remote — not GitHub Releases)

RoboScoutAI checks for updates by talking to a **git remote** (`git fetch` / shallow clone). It does **not** call `releases/latest`.

| Mode | Behavior |
| --- | --- |
| Source (`.git` present) | `git fetch` → compare `HEAD` to `origin/<branch>` → pull / reset → restart hint |
| Frozen EXE | Shallow clone/fetch into an update cache → look for tracked binaries under `desktop-downloads/` → replace EXE and relaunch |

Config:

| Variable | Default | Purpose |
| --- | --- | --- |
| `RAMSCOUT_GIT_REMOTE` | `https://github.com/drewsmash/RoboScoutAI.git` | Git remote URL |
| `RAMSCOUT_GIT_BRANCH` | `main` | Branch to track |
| `RAMSCOUT_UPDATE_CACHE` | appdata / `.ramscout-update-cache` | Mirror + package cache |
| `RAMSCOUT_GITHUB_TOKEN` / `GH_TOKEN` | — | Optional HTTPS auth for private remotes |

UI: **Check for updates** / **Update** chip. Messages: *up to date*, *update available from git*, or *git remote unreachable*.

For frozen auto-apply, commit desktop binaries under `desktop-downloads/` (e.g. `RoboScoutAI-windows-x64.exe`) on the tracked branch — or rebuild from source.

Desktop binaries are intentionally slim (no PyTorch / `ultralytics`). Demo mode, scorebug OCR, YouTube ingest, field playback, and **multi-strategy OpenCV tracking** still work — robot paths are approximate without YOLO/cloud. Tracking modes:

| Mode | What it uses |
| --- | --- |
| `auto` | YOLO (if installed) + bumper color + motion + optical flow + OpenAI/Gemini when keyed |
| `local` | Offline only (YOLO optional + OpenCV) |
| `motion` / `color` | No neural net |
| `openai` / `gemini` / `cloud` | Sparse cloud keyframes (OpenAI ~every 2s / 30 frames, capped) + local fill; Gemini preferred when keyed |

**Depth / BEV / multi-view:** overview tracking uses Depth Anything V2 when `transformers` + torch + Pillow are installed (falls back to classical depth), then a bird’s-eye (BEV) trapezoid so the top-down map accounts for camera angle. Stacked broadcasts are sectioned: top wide-angle → movement; bottom-left / bottom-right → blue / red scoring & climb cues. The broadcast panel draws a live tracking overlay.

Optional neural depth (source installs):

```bash
pip install transformers torch   # Pillow is in requirements.txt
# Depth-Anything-V2-Small loads from Hugging Face on first use
```

API keys (optional, also in the UI Advanced panel):

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY` or `GEMINI_API_KEY`
- `AI_GATEWAY_API_KEY` — Vercel AI Gateway; enables [Jev](https://vercel.com/ai-gateway/models/jev) (`typesafe-ai/jev`) to verify hub/climb/defense candidates and help classify multi-view layouts. A 403 usually means a bad/missing key or missing evaluation access — scout heuristics still run. Optional strict routing (often causes 403): `AI_GATEWAY_ONLY=typesafe-ai`, `AI_GATEWAY_ZERO_DATA_RETENTION=1`

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

- `ROBOSCOUT_GITHUB_REPO` / `RAMSCOUT_GITHUB_REPO=owner/repo` — override the update source (default `drewsmash/RoboScoutAI`)
- `ROBOSCOUT_GITHUB_TOKEN` / `RAMSCOUT_GITHUB_TOKEN` / `GITHUB_TOKEN` — required for in-app updates when the GitHub repo is **private**
- `ROBOSCOUT_DATA` / `RAMSCOUT_DATA=/path` — writable data directory for frozen builds

Because this repository is private, open the release while signed into GitHub to download, or set `RAMSCOUT_GITHUB_TOKEN` for in-app updates:

https://github.com/drewsmash/RoboScoutAI/releases/tag/v0.4.3


Click **Try a sample match** to explore the UI without a download.

For better tracking, drop a **robot-trained** Ultralytics `.pt` in the project folder or `models/` (for example `robot.pt`). When Ultralytics is installed and no local weights are present, RoboScoutAI can auto-download YOLO nano. When Ultralytics is missing (desktop builds), OpenCV motion / background-subtraction tracking still fills the path overlay so scout cards are not empty. Default COCO weights may follow people/vehicles — the UI warns when that happens.

A full YouTube run needs network access. A TBA auth key (`TBA_AUTH_KEY` or the form field) is optional: when present, RoboScoutAI resolves the match from the title or YouTube video id, then merges nicknames, `score_breakdown`, and Zebra tracks. Event key / match key fields override auto-resolve. Without TBA, teams and scores still come from the VOD overlay.

### YouTube download (bot checks)

Datacenter / cloud IPs often get YouTube’s “sign in to confirm you’re not a bot” wall. RoboScoutAI tries a resilient ladder:

1. **yt-dlp** with multiple player clients (`android`/`ios`, `tv`, `web_embedded`, …) and a **format ladder** (progressive `18` → muxed 720p → `worst`)  
2. Your **`YTDLP_PROXY`** / `HTTPS_PROXY` residential proxy (recommended for servers)  
3. **CLI yt-dlp fallback** (avoids Python JS-runtime pitfalls when Deno/Node is missing)  
4. An optional **auto-proxy ladder** (`YTDLP_AUTO_PROXY=1`, default on) for flaky public HTTP proxies  
5. **Upload a local VOD** in the UI (or pass a local file path) — skips YouTube entirely

Also supported:

- `YTDLP_COOKIES=/path/to/cookies.txt` — Netscape cookies from a signed-in browser  
- Desktop auto-discovery (no env required): `cookies.txt` next to the EXE, `%APPDATA%\RoboScoutAI\cookies.txt`, or `~/RoboScoutAI/cookies.txt`  
- Export with a browser extension such as **Get cookies.txt LOCALLY**, then drop the file in one of those locations  
- `YTDLP_BROWSER=chrome` (or `edge` / `firefox`) — cookies-from-browser on a signed-in machine  
- `YTDLP_FORMAT=…` — override the format selector  
- `YTDLP_AUTO_PROXY=0` — disable the public-proxy ladder  

**Upload a local MP4/MKV** in the UI whenever YouTube blocks — that path never goes through yt-dlp.

### Tracking modes

Default is **Auto (benchmark all)**: the first ~12 s of the match are tracked with every strategy set that is actually available on the machine (motion, bumper color, potato, hybrid, YOLO if `ultralytics` imports, Gemini / OpenAI if keyed). Each run is scored on physically meaningful criteria — mean visible robots vs. six, 3v3 balance per time bin, track persistence, in-field ratio, speed plausibility (20 ft/s cap), fraction of tracks that actually moved, ID churn — and the winner tracks the whole match. The choice and every score land in the job as `tracker_selection` and are shown in the UI; every other mode is still selectable as a manual override, and the mode list marks what each mode needs (`· needs gemini → OpenCV fallback`). See [`docs/TRACKING.md`](docs/TRACKING.md) for the full pipeline.

**Field-aware tracking (all modes, no model needed).** Every proposal is checked against the field before it can become a track:

- **Field-line homography** (`ramscout/fieldlines.py`): the carpet boundary is segmented from a temporal median of the overview pane (adaptive Lab colour + low saturation + low texture + edge cut), four straight edges are fitted robustly, and the resulting quad replaces the depth trapezoid when it is confident and agrees with the depth-derived pitch. Alliance tape and driver-station panels decide which side is blue, so the BEV is mirrored when blue is on the right.
- **Field gate** (`ramscout/trackers/field_gate.py`): the depth-corrected foot of each box is projected through the BEV homography; boxes outside the field polygon, wall-shaped boxes on the perimeter band, boxes with an implausible footprint (< 16 in or > 96 in), scorebug overlays and multi-edge boxes are rejected. A field polygon mask is also applied to the motion / colour masks themselves, so crowd, referees and driver stations never become proposals.
- **ByteTrack + OC-SORT association** (`ramscout/trackers/mot.py`): high-score proposals are matched first, low-score leftovers may only extend recently-seen tracks and never spawn; recovered tracks are re-updated from the observed displacement (observation-centric), and a momentum term penalises direction reversals. The BEV Kalman refuses associations above ~20 ft/s.
- **Bumper strips → robot boxes**: the colour tracker looks for *thin* saturated strips with something standing on them and grows them into a robot box; parked robots confirm without motion when the footprint is robot-sized, while plates and lit panels that never move are pruned and their spot is blocked from respawning.
- **Robust alliance colour** (`ramscout/trackers/alliance.py`): learned per broadcast in CIE Lab chroma from confirmed moving robots; sliding-window votes with hysteresis; final 3 red + 3 blue assignment.
- **Six lanes** (`ramscout/identity.py`): tracklets are chained across occlusions and every time-disjoint tracklet joins one of six robot lanes (three per alliance) instead of being dropped by a track cap; paths are RTS-smoothed in field coordinates (`ramscout/smoothing.py`); bumper-number OCR (`ramscout/ocr.py`, Tesseract / `cv2.text` / OpenAI Vision when keyed) votes team numbers onto stable tracks.

### Depth backends (Anything → 3D)

`ramscout/depth.py` tries **ONNX Runtime** (Depth Anything V2 Small, ~100 MB, auto-downloaded to the models dir on first use — `%LOCALAPPDATA%\RoboScoutAI\models` on Windows, `~/.local/share/RoboScoutAI/models` elsewhere), then **transformers + torch** (`pip install -r requirements-depth.txt`), then the classical row/texture prior. The active backend and anything missing are reported in `/api/trackers` (`depth`), in the job (`depth_backends`, `bev.depth_source`) and in the UI chips. Frozen desktop builds ship ONNX Runtime, so they get real neural depth without torch.

### Broadcast decomposition (multi-view)

`ramscout/layout.py` samples the whole VOD, finds composition seams from temporally persistent edges (coverage + continuity + a cross-seam correlation test that tells a real seam from a field wall), detects letterbox / pillarbox and the scorebug, classifies each pane (overview, alternate overview, blue / red side, sideline, graphics) and builds a timeline of layout segments. `track_video` follows that timeline: each overview crop gets its own homography, field gate and tracker; when the director cuts away from the overview the job records a gap instead of hallucinating paths, and side panes feed scoring / climb cues. The UI shows the real pane crops with labels and confidence.

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
- Multi-angle VODs: top overview + side cams improve scoring/climb confidence; single cams still work via BEV tilt cues
- Scorebug OCR needs a readable graphic; title/description is the backup
- Climb detection is “stayed on the Tower in endgame”, not rung level
- Hub candidates are low-speed dwells next to the Hub, not counted fuel
- Neural depth needs `onnxruntime` (bundled) or torch; without either, classical depth still adjusts the BEV trapezoid
- Local (no-model) tracking is a class-agnostic moving-object detector: alliance-coloured field elements that move in place (2018 switch / scale plates) can still be picked up briefly, and robots parked against their own alliance wall are the hardest case. Set `ROBOSCOUT_ROBOT_WEIGHTS_URL` (or `ROBOSCOUT_ROBOT_WEIGHTS`) to FRC-tuned YOLO weights for a real robot detector; COCO YOLO is only used to verify local proposals
