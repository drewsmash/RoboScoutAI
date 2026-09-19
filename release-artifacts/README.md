# Desktop update channel (git)

Frozen RoboScoutAI builds look here when **Check for updates** / **Update now** runs.

Commit platform binaries on the branch tracked by `RAMSCOUT_GIT_BRANCH` / `update-channel.txt`:

- `RoboScoutAI-windows-x64.exe`
- Optional signed copy (`RoboScoutAI-windows-x64-signed.exe`)
- `RoboScoutAI-linux-x86_64.tar.gz` / `RoboScoutAI-macos-arm64.zip`

If in-app apply hits Access Denied or SmartScreen, download the Windows exe from
GitHub Releases and run it from `%LOCALAPPDATA%\RoboScoutAI` (More info → Run anyway while unsigned).
