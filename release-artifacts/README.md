# Desktop update channel (git)

`RoboScoutAI.exe` (the Windows manager) checks this folder on the branch named in
`update-channel.txt` (overridable per install via `manager.json` / `RoboScoutAI.exe --channel <branch-or-tag>`).
GitHub Releases is the automatic fallback when the git channel is unreachable.

Published here by `.github/workflows/release.yml` (job `publish-channel`) on every `v*` tag:

| File | Purpose |
| --- | --- |
| `manifest.json` | version, `min_manager_version`, sha256 + size for every artifact. The manager refuses anything that does not verify. |
| `RoboScoutAI-app-windows-x64.exe` | the app build; installed side-by-side to `%LOCALAPPDATA%\RoboScoutAI\app\<version>\RoboScoutAI-app.exe` |
| `RoboScoutAI.exe` | the manager itself; only downloaded when `min_manager_version` requires a newer manager |
| `RoboScoutAI-windows-x64.exe` | **legacy name** = copy of the manager. 0.5.x in-app updaters download this into `%LOCALAPPDATA%\RoboScoutAI\RoboScoutAI.exe`, which migrates them onto the manager layout on next launch. |
| `UPDATE.txt` | commit SHA of the release |

New users should not download anything from here: grab `RoboScoutAI-Setup.exe` from
GitHub Releases once; the manager handles every later version (any branch or tag).

Manual publish (normally unnecessary):

```bash
pwsh packaging/build_windows.ps1            # on Windows → dist/release/*
cp dist/release/{manifest.json,RoboScoutAI.exe,RoboScoutAI-app-windows-x64.exe} release-artifacts/
git add -f release-artifacts/manifest.json release-artifacts/RoboScoutAI.exe release-artifacts/RoboScoutAI-app-windows-x64.exe
git commit -m "Publish RoboScoutAI <version> on update channel" && git push
```

See `docs/DESKTOP_APP.md` for the full install / update / rollback flow.
