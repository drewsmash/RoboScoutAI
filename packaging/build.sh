#!/usr/bin/env bash
# Build RoboScoutAI desktop release artifacts on the CURRENT OS.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m pip install -U pip wheel
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-desktop.txt

rm -rf build dist
python3 -m PyInstaller --noconfirm --clean packaging/ramscout.spec

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
mkdir -p dist/release

if [[ "$OS" == "darwin" ]]; then
  KEY="macos-arm64"
  if [[ "$ARCH" == "x86_64" ]]; then KEY="macos-x64"; fi
  ( cd dist && chmod +x RoboScoutAI && zip -9 "release/RoboScoutAI-${KEY}.zip" RoboScoutAI )
  echo "Built: dist/release/RoboScoutAI-${KEY}.zip"
elif [[ "$OS" == linux* ]]; then
  tar -C dist -czf "dist/release/RoboScoutAI-linux-${ARCH}.tar.gz" RoboScoutAI
  echo "Built: dist/release/RoboScoutAI-linux-${ARCH}.tar.gz"
else
  if [[ -f dist/RoboScoutAI.exe ]]; then
    cp dist/RoboScoutAI.exe "dist/release/RoboScoutAI-windows-x64.exe"
    echo "Built: dist/release/RoboScoutAI-windows-x64.exe"
  else
    echo "Built dist/RoboScoutAI — rename to RoboScoutAI-windows-x64.exe for release-artifacts/."
  fi
fi
