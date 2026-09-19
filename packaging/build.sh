#!/usr/bin/env bash
# Build RoboScoutAI desktop release artifacts on the CURRENT OS.
# Run this on a Mac to produce RoboScoutAI-macos-arm64.zip / macos-x64.zip.
# Run on Windows (Git Bash / PowerShell) to produce RoboScoutAI-windows-x64.exe.
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
    chmod +x RoboScoutAI
    # Prefer a zip that expands to a runnable binary (same layout as CI releases)
    zip -9 "release/RoboScoutAI-${KEY}.zip" RoboScoutAI
  )
  echo ""
  echo "Built: dist/release/RamScoutAI-${KEY}.zip"
  echo "Commit under release-artifacts/ for the git updater, or distribute directly."
  echo "First-run tip (unsigned): xattr -dr com.apple.quarantine ./RamScoutAI && chmod +x ./RamScoutAI"
elif [[ "$OS" == linux* ]]; then
  tar -C dist -czf "dist/release/RamScoutAI-linux-${ARCH}.tar.gz" RamScoutAI
  echo "Built: dist/release/RamScoutAI-linux-${ARCH}.tar.gz"
  echo "Copy to release-artifacts/ and commit for the git updater."
else
  # Windows / Git Bash / MSYS
  if [[ -f dist/RamScoutAI.exe ]]; then
    cp dist/RamScoutAI.exe "dist/release/RamScoutAI-windows-x64.exe"
    echo "Built: dist/release/RamScoutAI-windows-x64.exe"
    echo "Copy to release-artifacts/ and commit for the git updater."
  else
    echo "Built dist/RamScoutAI — rename to RamScoutAI-windows-x64.exe for release-artifacts/."
  fi
fi
