# Signing RamScoutAI desktop builds

Unsigned Windows `.exe` and macOS apps are often flagged by SmartScreen / Gatekeeper
("unrecognized developer" or false-positive malware warnings). Code signing fixes that.

## Honest limits

| Approach | Stops SmartScreen / Gatekeeper? | Cost |
| --- | --- | --- |
| **Self-signed** (`packaging/sign.sh --self-sign`) | No — still untrusted, but the PE has a signature block | Free |
| **Windows OV/EV Authenticode** (DigiCert, Sectigo, SSL.com) | Yes (EV gets reputation faster) | Paid (~$200–500+/yr) |
| **Apple Developer ID + notarization** | Yes | Apple Developer Program ($99/yr) |

There is no free way to make Windows/macOS fully trust an unknown publisher. Self-sign
is still useful for teams that distribute internally and install the cert once.

## Bypass (unsigned / self-signed)

### Windows

1. Right-click `RamScoutAI-windows-x64.exe` → **Properties** → **Unblock** (if present) → OK  
2. Or click **More info** → **Run anyway** on the SmartScreen prompt

### macOS

```bash
xattr -dr com.apple.quarantine ./RamScoutAI
chmod +x ./RamScoutAI
./RamScoutAI
```

## Local self-sign (Windows, free)

Requires `osslsigncode` and `openssl`:

```bash
./packaging/sign.sh --self-sign dist/release/RamScoutAI-windows-x64.exe
```

This writes `packaging/certs/` (gitignored). Share the `.crt` with teammates who want to
trust the self-signed publisher on their machines.

## Real signing (recommended for public releases)

### GitHub Actions secrets

Add these repository secrets, then push a `v*` tag. The release workflow signs when
secrets exist and skips (with a warning) when they do not.

| Secret | Platform | Purpose |
| --- | --- | --- |
| `WINDOWS_CERT_PFX_BASE64` | Windows | Base64-encoded `.pfx` code-signing certificate |
| `WINDOWS_CERT_PASSWORD` | Windows | Password for the `.pfx` |
| `MACOS_CERTIFICATE_P12_BASE64` | macOS | Base64-encoded Developer ID `.p12` |
| `MACOS_CERTIFICATE_PASSWORD` | macOS | Password for the `.p12` |
| `MACOS_API_KEY_ID` | macOS | App Store Connect API key id (notarization) |
| `MACOS_API_ISSUER_ID` | macOS | App Store Connect issuer id |
| `MACOS_API_KEY_P8_BASE64` | macOS | Base64-encoded `.p8` AuthKey |
| `MACOS_TEAM_ID` | macOS | Apple Team ID |

These names match `.github/workflows/release.yml`.

### Local with a real Windows PFX

```bash
export WINDOWS_CERT_PFX=/path/to/cert.pfx
export WINDOWS_CERT_PASSWORD='…'
./packaging/sign.sh RamScoutAI-windows-x64.exe
```

Or with `signtool` on Windows:

```powershell
signtool sign /fd SHA256 /f cert.pfx /p "PASSWORD" /tr http://timestamp.digicert.com /td SHA256 RamScoutAI-windows-x64.exe
signtool verify /pa RamScoutAI-windows-x64.exe
```

### macOS (codesign + notarytool) — must run on a Mac

```bash
codesign --force --deep --options runtime \
  --sign "Developer ID Application: Your Name (TEAMID)" \
  ./RamScoutAI

ditto -c -k --keepParent RamScoutAI RamScoutAI.zip
xcrun notarytool submit RamScoutAI.zip \
  --key AuthKey.p8 --key-id KEYID --issuer ISSUER \
  --wait
xcrun stapler staple RamScoutAI
```

Ad-hoc sign (no Apple account, still Gatekeeper-blocked for downloaded apps):

```bash
codesign --force --deep -s - ./RamScoutAI
```

## Certificate vendors

- Windows: DigiCert, Sectigo, SSL.com (OV or EV Authenticode)
- macOS: Apple Developer Program → Certificates → Developer ID Application

Do **not** commit private keys or `.pfx` / `.p12` / `.p8` files to git.
