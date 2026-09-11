# RamScoutAI desktop builds

## Upload to GitHub Release

### Windows (ready)
- `RamScoutAI-windows-x64.exe` — upload as-is

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
