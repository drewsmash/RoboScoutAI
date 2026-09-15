# Desktop update channel (git)

Frozen RamScoutAI builds look here when **Check for updates** / **Update now** runs.

Commit platform binaries on the branch tracked by `RAMSCOUT_GIT_BRANCH` (default `main`):

- `RamScoutAI-windows-x64.exe`
- `RamScoutAI-windows-x64-signed.exe` (optional)
- `RamScoutAI-linux-x86_64.tar.gz`
- `RamScoutAI-macos-arm64.zip` / `RamScoutAI-macos-x64.zip`

This directory is the git updater source of truth. Local-only copies may still live under `desktop-downloads/` (gitignored).
