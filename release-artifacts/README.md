# Desktop update channel (git)

Frozen RoboScoutAI builds look here when **Check for updates** / **Update now** runs.

Commit platform binaries on the branch tracked by `RAMSCOUT_GIT_BRANCH` / `update-channel.txt`:

- `RoboScoutAI-windows-x64.exe` (preferred) and legacy `RamScoutAI-windows-x64.exe`
- Optional signed copies (`*-windows-x64-signed.exe`)
- `RoboScoutAI-linux-x86_64.tar.gz` / `RoboScoutAI-macos-arm64.zip` (and legacy RamScoutAI names)

This directory is the git updater source of truth. Local-only copies may still live under `desktop-downloads/` (gitignored).

If in-app apply hits **Access Denied** (Program Files), download the Windows exe from
GitHub Releases and run it — newer builds install under `%LOCALAPPDATA%\RoboScoutAI`.
