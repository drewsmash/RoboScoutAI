#!/usr/bin/env bash
# Build RoboScoutAI desktop release artifacts on the CURRENT OS.
#
# Windows: use packaging/build_windows.ps1 instead (app + manager + Setup + manifest).
# macOS / Linux: builds the app binary from packaging/app.spec and zips/tars it.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -U pip wheel
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-desktop.txt

rm -rf build dist
python3 -m PyInstaller --noconfirm --clean packaging/app.spec

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
mkdir -p dist/release

if [[ "$OS" == "darwin" ]]; then
  KEY="macos-arm64"
  if [[ "$ARCH" == "x86_64" ]]; then KEY="macos-x64"; fi
  ( cd dist && mv -f RoboScoutAI-app RoboScoutAI && chmod +x RoboScoutAI && zip -9 "release/RoboScoutAI-${KEY}.zip" RoboScoutAI )
  echo "Built: dist/release/RoboScoutAI-${KEY}.zip"
elif [[ "$OS" == linux* ]]; then
  ( cd dist && mv -f RoboScoutAI-app RoboScoutAI )
  tar -C dist -czf "dist/release/RoboScoutAI-linux-${ARCH}.tar.gz" RoboScoutAI
  echo "Built: dist/release/RoboScoutAI-linux-${ARCH}.tar.gz"
else
  echo "On Windows run: pwsh packaging/build_windows.ps1"
  if [[ -f dist/RoboScoutAI-app.exe ]]; then
    cp dist/RoboScoutAI-app.exe "dist/release/RoboScoutAI-app-windows-x64.exe"
    echo "Built: dist/release/RoboScoutAI-app-windows-x64.exe (manager + Setup still needed)"
  fi
fi
