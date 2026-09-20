# RoboScoutAI desktop app (Windows installer, manager, updater)

RoboScoutAI ships on Windows as a **two-binary launcher/manager design**. Users
download **one** file once — `RoboScoutAI-Setup.exe` — and every later version
(any branch or tag) arrives through the manager.

```
RoboScoutAI-Setup.exe        one-time installer (manager code + bundled payload)
%LOCALAPPDATA%\RoboScoutAI\
├── RoboScoutAI.exe          manager: launcher / updater / rollback / uninstaller (thin, stdlib only)
├── app\
│   ├── 0.6.0\RoboScoutAI-app.exe     the actual app (FastAPI + web UI), never run in-place by users
│   └── 0.6.1\RoboScoutAI-app.exe     next version downloaded side-by-side
├── current.json             which app version is active (atomic pointer + sha256 + previous version)
├── manager.json             update channel (branch/tag), git remote, GitHub repo, flags
├── update-status.json       live progress of the current update (read by the web UI)
├── ipc.json                 per-launch token the app must present to trigger an update
├── update.log / manager.log / app.log
├── update-cache\mirror      blob-less shallow git mirror of the channel branch
└── data\                    user data (jobs, models, cookies) — kept on update and uninstall
```

## Why two binaries

* The manager is the only thing that starts `RoboScoutAI-app.exe`, and every
  app build lives in its own `app\<version>\` folder. Nothing is ever written
  over a running exe, so the old "Access is denied / 0 file(s) moved" failures of
  the in-place updater cannot happen.
* Switching versions is one atomic `current.json` write. Rollback is the same
  write in reverse. The previous version is kept; older ones are pruned (keep 2).
* The manager is tiny and pure standard library, so it rarely needs to change.
  When it must (`min_manager_version` in the manifest), it updates itself first
  via a staged `RoboScoutAI.exe.new` + `manager-update.bat` that waits for the
  manager PID, swaps, and relaunches.

## Install (what a user does)

1. Download `RoboScoutAI-Setup.exe` from GitHub Releases and run it.
   No administrator rights are needed; nothing goes into Program Files.
2. Setup installs `RoboScoutAI.exe` and the bundled `app\<version>\RoboScoutAI-app.exe`
   (verified against the bundled `manifest.json`), writes `manager.json`
   (channel seeded from the manifest / `release-artifacts/update-channel.txt`),
   creates a **Start Menu** shortcut (+ optional **Desktop** shortcut), registers
   the uninstall entry under
   `HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\RoboScoutAI`
   (`DisplayName`, `DisplayVersion`, `Publisher`, `InstallLocation`,
   `DisplayIcon`, `UninstallString`, `QuietUninstallString`, `EstimatedSize`…),
   and launches the app.
3. Later: start RoboScoutAI from the Start Menu. The manager verifies the active
   build (exists + sha256), picks a free port, issues an IPC token and starts
   `RoboScoutAI-app.exe --managed --port N --manager-token T --manager-root R --manager-exe M`
   while showing a small splash until `/api/health` answers.

`RoboScoutAI-Setup.exe --silent --no-desktop-shortcut --no-launch` installs unattended.
Double-clicking a bare `RoboScoutAI.exe` (e.g. from Downloads) also works: on
first launch it copies itself into the install root, writes `manager.json`,
creates the Start Menu shortcut, registers uninstall, and downloads the app from
the channel.

## Update flow

1. **Check** — `RoboScoutAI.exe --check` (the app calls this for
   `GET /api/updates/check`). Channels, in order:
   * **git** (primary): `git init` + `git fetch --depth 1 --filter=blob:none origin +refs/heads/<channel>:refs/channel/<channel>`
     (tags tried second, filter dropped if unsupported) into `update-cache\mirror`,
     then `git cat-file -p refs/channel/<channel>:release-artifacts/manifest.json`.
     The manifest gives version, sha256/size per artifact and `min_manager_version`.
     If the branch has no manifest the manager reports the branch's
     `ramscout/__init__.py` version for diagnostics and falls through.
   * **GitHub Releases** (fallback): newest semver tag (or the exact tag if the
     channel *is* a tag) whose assets include `manifest.json`; downloads use
     `browser_download_url` (or the API asset URL when a token is present).
2. **Download** — `RoboScoutAI.exe --update --token <ipc token>` (the app spawns
   this for `POST /api/updates/download` and returns immediately). The artifact
   streams into `app\<newversion>\.download.part` (`git cat-file blob …` or HTTP),
   progress is written to `update-status.json` (`checking → downloading N% →
   verifying → installing → done | error`). The web UI polls `GET /api/updates/status`
   and shows it on the Update chip.
3. **Verify** — sha256 + size must match the manifest, otherwise the part file
   is deleted and the status is `error`; `current.json` is untouched.
4. **Install** — the file is renamed to `app\<version>\RoboScoutAI-app.exe`,
   `current.json` is written atomically (temp file + `os.replace`) with
   `previous_version` set, versions older than the newest two (and never the
   active or previous) are pruned.
5. **Relaunch** — the UI asks "Restart now?"; `POST /api/updates/relaunch`
   closes the app window and exits with code **75**. The manager sees 75,
   re-verifies `current.json`, and starts the new version on the same port; the
   page reloads itself when `/api/health` is back.

If the manifest's `min_manager_version` is newer than the running manager, step 2
downloads `RoboScoutAI.exe` from the channel instead, stages it as
`RoboScoutAI.exe.new`, schedules the swap `.bat`, and relaunches
`RoboScoutAI.exe --update --launch-after` so the app update continues with the
new manager. Manifests with a `schema` newer than the manager understands are
refused with a clear message.

## Rollback, repair, channels, uninstall

| Command | Effect |
| --- | --- |
| `RoboScoutAI.exe` | launch the active app (default) |
| `RoboScoutAI.exe --check [--json --out file]` | check the channel |
| `RoboScoutAI.exe --update [--force] [--launch-after]` | download + install side-by-side |
| `RoboScoutAI.exe --rollback [VERSION]` | point `current.json` at the previous (or given) installed version; verified by sha256 |
| `RoboScoutAI.exe --repair` | re-download the channel build if the active exe is missing/corrupt, refresh manager copy, shortcuts, registry |
| `RoboScoutAI.exe --channel <branch-or-tag>` | switch channel (`main`, `release/2027`, `v0.7.2`, any future branch) |
| `RoboScoutAI.exe --list` / `--status` / `--version` | installed versions / update progress / versions |
| `RoboScoutAI.exe --uninstall [--purge] [--silent]` | remove registry entry + shortcuts now, then a `.bat` waits for the manager PID and deletes `app\`, the manager and json files. `data\` is kept unless `--purge`. Windows "Installed apps → Uninstall" runs this. |

The launcher itself self-heals: if the active exe is missing or its hash does not
match, it rolls back to the previous version automatically, and if none exists
it repairs from the channel before starting.

Environment overrides (mostly for testing): `ROBOSCOUT_INSTALL_ROOT`,
`ROBOSCOUT_CHANNEL`, `ROBOSCOUT_GIT_REMOTE`, `ROBOSCOUT_GITHUB_REPO`,
`ROBOSCOUT_GITHUB_TOKEN` / `GITHUB_TOKEN` (private repos; injected as an
`http.extraHeader`, never written to git config).

## Manifest (`release-artifacts/manifest.json`, also a Release asset)

```json
{
  "schema": 1,
  "product": "RoboScoutAI",
  "version": "0.6.0",
  "channel": "main",
  "commit": "…",
  "created_at": "2026-09-20T05:00:00+00:00",
  "min_manager_version": "0.6.0",
  "artifacts": {
    "RoboScoutAI-app-windows-x64.exe": {"sha256": "…", "size": 91545776, "kind": "app", "platform": "windows-x64"},
    "RoboScoutAI.exe":                 {"sha256": "…", "size": 9000000,  "kind": "manager", "platform": "windows-x64"},
    "RoboScoutAI-Setup.exe":           {"sha256": "…", "size": 100000000, "kind": "setup", "platform": "windows-x64"}
  }
}
```

`python -m roboscout_manager.manifest build|verify` creates/validates it.

## Release pipeline (`.github/workflows/release.yml`)

Tag `vX.Y.Z` (version must equal `ramscout.__version__`, which must equal
`roboscout_manager.__version__` — a test enforces this):

1. `test` — manager + API tests on Linux.
2. `build-windows` — `packaging/build_windows.ps1`: `app.spec` → `RoboScoutAI-app-windows-x64.exe`,
   `manager.spec` → `RoboScoutAI.exe`, manifest, `setup.spec` (bundles the two + manifest as `payload/`)
   → `RoboScoutAI-Setup.exe`, final manifest including Setup. Optional Authenticode
   signing of all three when `WINDOWS_CERT_PFX_BASE64` / `WINDOWS_CERT_PASSWORD`
   are set. Smoke test: `--version`, silent Setup into a temp root, `--list`, `--uninstall`.
3. `build-macos` — `app.spec` → `RoboScoutAI-macos-arm64.zip` (unchanged flow, optional signing/notarization).
4. `publish` — verifies digests, checks manifest version == tag, creates the GitHub Release.
5. `publish-channel` — merges the release commit into the channel branch from
   `release-artifacts/update-channel.txt` and commits `manifest.json`,
   `RoboScoutAI-app-windows-x64.exe`, `RoboScoutAI.exe`, and the legacy-named
   `RoboScoutAI-windows-x64.exe` (= manager) so 0.5.x in-app updaters migrate onto
   the manager layout.

Local: `pwsh packaging/build_windows.ps1` (Windows); `bash packaging/build.sh` (macOS/Linux app only).

## Code map

| Module | Role |
| --- | --- |
| `roboscout_manager/manifest.py` | manifest model/parse/build, sha256 verify, version compare, `min_manager_version` gate |
| `roboscout_manager/state.py` | install-root layout, atomic JSON, `current.json`, `manager.json`, `update.log` |
| `roboscout_manager/install.py` | side-by-side install, prune, verify, Setup flow, first-run bootstrap, uninstall |
| `roboscout_manager/updater.py` | `GitChannel`, `ReleasesChannel`, `check_for_update`, `perform_update`, manager self-update |
| `roboscout_manager/rollback.py` | rollback candidates + switch |
| `roboscout_manager/launcher.py` | verify → launch child → relaunch on exit 75; splash; self-heal |
| `roboscout_manager/ipc.py` | IPC token + status file |
| `roboscout_manager/shortcuts.py` / `registry.py` | PowerShell `.lnk` script, HKCU uninstall entry |
| `roboscout_manager/cli.py` / `__main__.py` / `setup_entry.py` | command line; `RoboScoutAI.exe` and `RoboScoutAI-Setup.exe` entry points |
| `ramscout/managed.py` | app-side bridge: shells out to the manager, reads status, requests relaunch |
| `desktop/main.py` | `--managed`, `--manager-token`, `--manager-root`, `--manager-exe` |
| `app.py` | `/api/updates/check|download|status|relaunch` delegate to the manager when managed |

`ramscout/updater.py` (in-place git updater) remains for source checkouts and
refuses to run when a manager install is present.

## Limitations

* Builds are unsigned unless the signing secrets are configured; SmartScreen will
  show "More info → Run anyway" on first run of `RoboScoutAI-Setup.exe`.
  Because the manager only replaces files it downloaded and verified, later
  updates never trigger SmartScreen again.
* The git channel needs `git` on PATH (Git for Windows). Without it the manager
  transparently uses GitHub Releases.
* The manager, Setup and app are 64-bit Windows only; macOS keeps the single
  zipped binary without a manager.
* The manager's tkinter splash/dialogs fall back to silent operation when tkinter
  is unavailable (all commands still work with `--json --out FILE`).
