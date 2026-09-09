# Signing RamScoutAI desktop builds

Unsigned Windows `.exe` and macOS apps are often flagged by SmartScreen / Gatekeeper
("unrecognized developer" or false-positive malware warnings). Code signing fixes that.

## Why this happens

- Windows SmartScreen trusts binaries signed with an **Authenticode** certificate from a
  public CA (or an EV cert for immediate reputation).
- macOS Gatekeeper requires **Developer ID Application** signing + **notarization**
  through Apple.

RamScoutAI's GitHub Actions release workflow can sign automatically when secrets are set.
Until then, users can still run the app:

### Windows (unsigned)

1. Right-click `RamScoutAI-windows-x64.exe` → **Properties** → **Unblock** (if present) → OK  
2. Or click **More info** → **Run anyway** on the SmartScreen prompt

### macOS (unsigned)

```bash
xattr -dr com.apple.quarantine ./RamScoutAI
chmod +x ./RamScoutAI
./RamScoutAI
```

## GitHub Actions secrets (recommended)

Add these repository secrets, then push a `v*` tag:

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

The release workflow signs when those secrets exist and skips signing (with a warning) when they do not.

## Local signing

### Windows (signtool)

```powershell
signtool sign /fd SHA256 /f cert.pfx /p "PASSWORD" /tr http://timestamp.digicert.com /td SHA256 RamScoutAI-windows-x64.exe
signtool verify /pa RamScoutAI-windows-x64.exe
```

### macOS (codesign + notarytool)

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

## Certificate vendors

- Windows: DigiCert, Sectigo, SSL.com (OV or EV Authenticode)
- macOS: Apple Developer Program → Certificates → Developer ID Application

Do **not** commit private keys or `.pfx` / `.p12` / `.p8` files to git.
