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
