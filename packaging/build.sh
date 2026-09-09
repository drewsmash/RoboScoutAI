#!/usr/bin/env bash
# Build a local RamScoutAI binary with PyInstaller (run on Windows or macOS for release artifacts).
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
  (
    cd dist
    zip -9 "release/RamScoutAI-${KEY}.zip" RamScoutAI
  )
  echo "Built dist/release/RamScoutAI-${KEY}.zip"
elif [[ "$OS" == linux* ]]; then
  tar -C dist -czf "dist/release/RamScoutAI-linux-${ARCH}.tar.gz" RamScoutAI
  echo "Built dist/release/RamScoutAI-linux-${ARCH}.tar.gz"
else
  echo "Built dist/RamScoutAI (copy/rename for Windows release as RamScoutAI-windows-x64.exe)"
fi
