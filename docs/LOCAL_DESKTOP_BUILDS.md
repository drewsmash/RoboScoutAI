# RamScoutAI desktop builds

The packaged app opens in a **chrome-less window** (native WebView when available, otherwise Chrome/Edge `--app` mode). There is no browser URL bar. Pass `--browser` to the launcher if you want a normal tab instead.

## Building artifacts

```bash
bash packaging/build.sh
```

### Windows
- Output: `dist/release/RamScoutAI-windows-x64.exe` (or `dist/RamScoutAI.exe`)
- Optional signed copy: `RamScoutAI-windows-x64-signed.exe`
- For the **git updater** to auto-apply frozen builds, commit the EXE under `release-artifacts/` on the tracked branch (see below). `desktop-downloads/` is only for local staging and is gitignored.

The Windows app opens Edge/Chrome in `--app` mode (no URL bar). It does **not** use the
bundled WinForms WebView path, which crashed some installs with
`NullReferenceException` in `Control.set_Text`. Edge or Chrome must be installed
(almost always true on Windows 10/11 via Edge).

### Linux
```bash
# after packaging/build.sh
# dist/release/RamScoutAI-linux-<arch>.tar.gz
```

### macOS (build on a Mac)
```bash
bash packaging/build.sh
# dist/release/RamScoutAI-macos-arm64.zip (or macos-x64)
```

## In-app updater (git)

The updater **never** calls GitHub Releases (`/releases/latest`). It uses git:

1. Resolve remote: `RAMSCOUT_GIT_REMOTE` (default repo URL) and `RAMSCOUT_GIT_BRANCH` (default `main`)
2. **Source installs** (`.git` present): fetch, compare SHAs, pull/reset, ask for restart
3. **Frozen EXE**: shallow clone/fetch into `%APPDATA%/RamScoutAI/update-cache` (or `RAMSCOUT_UPDATE_CACHE`), sparse-checkout known artifact paths under `release-artifacts/` (fallback: `desktop-downloads/`), replace the running binary

| Check result | User-facing message |
| --- | --- |
| Local SHA == remote SHA | Up to date |
| Remote ahead + apply possible | Update available from git → Update now |
| Git missing / network / auth fail | git remote unreachable |
| Remote ahead but no platform binary | Update available, but no desktop binary on the branch |

### Publishing a desktop binary for the updater

```bash
bash packaging/build.sh
mkdir -p release-artifacts
cp dist/release/RamScoutAI-windows-x64.exe release-artifacts/
# optional: echo "$(git rev-parse HEAD)" > release-artifacts/UPDATE.txt
git add -f release-artifacts/RamScoutAI-windows-x64.exe
git commit -m "Publish Windows desktop build for git updater"
git push
```

Private remotes: set `RAMSCOUT_GITHUB_TOKEN`, `GH_TOKEN`, or `RAMSCOUT_GIT_TOKEN` (injected into the fetch URL only — not written into `.git/config`).

Legacy `RAMSCOUT_GITHUB_REPO=owner/repo` is still accepted and rewritten to `https://github.com/owner/repo.git`.

## YouTube cookies (desktop)

If YouTube bot-checks downloads, either **upload the MP4/MKV** or place Netscape `cookies.txt` in one of:

1. `YTDLP_COOKIES` env var
2. Next to `RamScoutAI-windows-x64.exe`
3. `%APPDATA%\RamScoutAI\cookies.txt` (Windows)
4. `~/RamScoutAI/cookies.txt`

Export with a browser extension such as **Get cookies.txt LOCALLY**. Upload failures are classified separately from YouTube bot blocks — uploading a local file never shows the YouTube-blocked banner.

## Unsigned apps
Windows SmartScreen / macOS Gatekeeper may warn. See `docs/SIGNING.md`.

## Optional env flags (tracking / download)

| Variable | Purpose |
| --- | --- |
| `YTDLP_COOKIES` | Path to Netscape `cookies.txt` for YouTube |
| `YTDLP_BROWSER` | `chrome` / `edge` / `firefox` cookies-from-browser |
| `YTDLP_PROXY` / `HTTPS_PROXY` | Residential proxy for yt-dlp |
| `YTDLP_AUTO_PROXY` | `1` (default) enables public-proxy ladder; `0` disables |
| `YTDLP_FORMAT` | Override yt-dlp format selector |
| `RAMSCOUT_GIT_REMOTE` | Git remote URL for updates |
| `RAMSCOUT_GIT_BRANCH` | Branch to track (default `main`) |
| `RAMSCOUT_UPDATE_CACHE` | Override update mirror/cache directory |
| `RAMSCOUT_GITHUB_REPO` | Legacy `owner/repo` → rewritten to a git HTTPS URL |
| `RAMSCOUT_GITHUB_TOKEN` / `GH_TOKEN` / `RAMSCOUT_GIT_TOKEN` | HTTPS auth for private git remotes |
| `RAMSCOUT_OPENAI_INTERVAL_S` | Seconds between OpenAI vision keyframes (default `2`) |
| `RAMSCOUT_OPENAI_FRAME_STRIDE` | Min processed frames between OpenAI calls (default `30`) |
| `RAMSCOUT_OPENAI_MAX_CALLS` | Cap OpenAI calls per match (default `24`) |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Cloud vision keys |
