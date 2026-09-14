#!/usr/bin/env bash
# Sign RamScoutAI desktop artifacts.
#
# Real trust (no SmartScreen / Gatekeeper nag) needs a paid CA / Apple Developer ID.
# This script also supports a LOCAL self-signed Windows Authenticode signature so the
# PE has a signature block (still untrusted by SmartScreen until a real cert is used).
#
# Env (optional, preferred for release):
#   WINDOWS_CERT_PFX / WINDOWS_CERT_PASSWORD  — real Authenticode .pfx
#   or WINDOWS_CERT_PFX_BASE64
#
# Usage:
#   ./packaging/sign.sh path/to/RamScoutAI-windows-x64.exe
#   ./packaging/sign.sh --self-sign path/to/RamScoutAI-windows-x64.exe
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SELF_SIGN=0
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --self-sign) SELF_SIGN=1; shift ;;
    -h|--help)
      sed -n '1,20p' "$0"
      exit 0
      ;;
    *) TARGET="$1"; shift ;;
  esac
done

if [[ -z "$TARGET" ]]; then
  echo "Usage: $0 [--self-sign] <file.exe>" >&2
  exit 2
fi
if [[ ! -f "$TARGET" ]]; then
  echo "Missing file: $TARGET" >&2
  exit 1
fi

need_ossl() {
  if ! command -v osslsigncode >/dev/null 2>&1; then
    echo "osslsigncode not found. Install it (apt install osslsigncode / brew install osslsigncode)." >&2
    exit 1
  fi
}

sign_windows() {
  local exe="$1"
  local pfx="$2"
  local pass="$3"
  local out="${exe}.signed"
  need_ossl
  osslsigncode sign \
    -pkcs12 "$pfx" \
    -pass "$pass" \
    -n "RamScoutAI" \
    -i "https://github.com/drewsmash/RamScoutAI" \
    -t http://timestamp.digicert.com \
    -in "$exe" \
    -out "$out"
  mv "$out" "$exe"
  osslsigncode verify -in "$exe" || true
  echo "Signed: $exe"
}

CERT_DIR="${RAMSCOUT_CERT_DIR:-$ROOT/packaging/certs}"
mkdir -p "$CERT_DIR"

if [[ "$SELF_SIGN" -eq 1 || ( -z "${WINDOWS_CERT_PFX:-}" && -z "${WINDOWS_CERT_PFX_BASE64:-}" ) ]]; then
  echo "No release PFX in env — creating/using a LOCAL self-signed code-signing cert."
  echo "SmartScreen will still warn until you buy a real Authenticode certificate."
  PFX="$CERT_DIR/ramscout-selfsigned.pfx"
  PASS="${WINDOWS_CERT_PASSWORD:-ramscout-dev}"
  if [[ ! -f "$PFX" ]]; then
    openssl req -x509 -newkey rsa:4096 -sha256 -days 825 -nodes \
      -keyout "$CERT_DIR/ramscout-selfsigned.key" \
      -out "$CERT_DIR/ramscout-selfsigned.crt" \
      -subj "/CN=RamScoutAI Local Dev/O=RamScoutAI/C=US" \
      -addext "extendedKeyUsage=codeSigning"
    openssl pkcs12 -export \
      -out "$PFX" \
      -inkey "$CERT_DIR/ramscout-selfsigned.key" \
      -in "$CERT_DIR/ramscout-selfsigned.crt" \
      -passout "pass:$PASS"
    echo "Wrote $PFX (do not commit private keys)."
  fi
  # Keep private material out of git
  if [[ -d "$ROOT/.git" ]]; then
    grep -qxF 'packaging/certs/' "$ROOT/.gitignore" 2>/dev/null || echo 'packaging/certs/' >> "$ROOT/.gitignore"
  fi
  sign_windows "$TARGET" "$PFX" "$PASS"
  exit 0
fi

if [[ -n "${WINDOWS_CERT_PFX_BASE64:-}" ]]; then
  PFX="$CERT_DIR/release.pfx"
  echo "$WINDOWS_CERT_PFX_BASE64" | base64 --decode > "$PFX"
  sign_windows "$TARGET" "$PFX" "${WINDOWS_CERT_PASSWORD:-}"
  rm -f "$PFX"
  exit 0
fi

sign_windows "$TARGET" "$WINDOWS_CERT_PFX" "${WINDOWS_CERT_PASSWORD:-}"
