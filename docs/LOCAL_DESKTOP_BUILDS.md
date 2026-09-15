# RamScoutAI desktop builds

The packaged app opens in a **chrome-less window** (native WebView when available, otherwise Chrome/Edge `--app` mode). There is no browser URL bar. Pass `--browser` to the launcher if you want a normal tab instead.

## Upload to GitHub Release

### Windows (ready)
- `RamScoutAI-windows-x64.exe` — upload as-is

The Windows app opens Edge/Chrome in `--app` mode (no URL bar). It does **not** use the
bundled WinForms WebView path, which crashed some installs with
`NullReferenceException` in `Control.set_Text`. Edge or Chrome must be installed
(almost always true on Windows 10/11 via Edge).

### Linux (ready, split for download size)
Download all `RamScoutAI-linux-x86_64.tar.gz.part*` files, then:
```bash
cat RamScoutAI-linux-x86_64.tar.gz.part* > RamScoutAI-linux-x86_64.tar.gz
```
Upload the reassembled `.tar.gz` to the release (or keep parts if you prefer).

### macOS (not buildable here)
Must be built on a Mac:
```bash
bash packaging/build.sh
```
Upload `dist/release/RamScoutAI-macos-arm64.zip`.

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
| `RAMSCOUT_OPENAI_INTERVAL_S` | Seconds between OpenAI vision keyframes (default `2`) |
| `RAMSCOUT_OPENAI_FRAME_STRIDE` | Min processed frames between OpenAI calls (default `30`) |
| `RAMSCOUT_OPENAI_MAX_CALLS` | Cap OpenAI calls per match (default `24`) |
| `OPENAI_API_KEY` / `GOOGLE_API_KEY` / `GEMINI_API_KEY` | Cloud vision keys |
