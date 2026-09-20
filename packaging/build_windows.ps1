<#
.SYNOPSIS
  Build the three RoboScoutAI Windows binaries plus manifest.json.

.DESCRIPTION
  1. RoboScoutAI-app-windows-x64.exe  (packaging/app.spec)     — FastAPI + web UI
  2. RoboScoutAI.exe                  (packaging/manager.spec) — launcher / updater / uninstaller
  3. manifest.json                    (roboscout_manager.manifest) — version, sha256, min_manager_version
  4. RoboScoutAI-Setup.exe            (packaging/setup.spec)   — installer with 1–3 bundled as payload
  5. manifest.json is rebuilt to include the Setup exe digest.

  Signing is optional: set WINDOWS_CERT_PFX_BASE64 + WINDOWS_CERT_PASSWORD (or pass -Sign
  with a signtool on PATH) and every exe is Authenticode-signed before it is bundled/hashed.

.EXAMPLE
  pwsh packaging/build_windows.ps1
  pwsh packaging/build_windows.ps1 -Version 0.6.0 -Channel main -SkipInstall
#>
[CmdletBinding()]
param(
    [string]$Version = "",
    [string]$Channel = "",
    [string]$Commit = "",
    [string]$MinManagerVersion = "",
    [switch]$SkipInstall,
    [switch]$Sign,
    [switch]$NoClean
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

function Get-PyVersion {
    $text = Get-Content (Join-Path $Root "ramscout/__init__.py") -Raw
    if ($text -match '__version__\s*=\s*"([^"]+)"') { return $Matches[1] }
    throw "Could not read __version__ from ramscout/__init__.py"
}

function Get-ManagerVersion {
    $text = Get-Content (Join-Path $Root "roboscout_manager/__init__.py") -Raw
    if ($text -match '__version__\s*=\s*"([^"]+)"') { return $Matches[1] }
    throw "Could not read __version__ from roboscout_manager/__init__.py"
}

function Get-Channel {
    $file = Join-Path $Root "release-artifacts/update-channel.txt"
    if (Test-Path $file) {
        foreach ($line in Get-Content $file) {
            $t = $line.Trim()
            if ($t -and -not $t.StartsWith("#")) { return $t }
        }
    }
    return "main"
}

function Find-SignTool {
    $cmd = Get-Command signtool -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kits = Join-Path ${env:ProgramFiles(x86)} "Windows Kits\10\bin"
    if (Test-Path $kits) {
        $hit = Get-ChildItem -Path $kits -Recurse -Filter signtool.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\" } | Sort-Object FullName -Descending | Select-Object -First 1
        if ($hit) { return $hit.FullName }
    }
    return $null
}

function Sign-Files([string[]]$Files) {
    $pfxB64 = $env:WINDOWS_CERT_PFX_BASE64
    if (-not $pfxB64 -and -not $Sign) {
        Write-Warning "WINDOWS_CERT_PFX_BASE64 not set — shipping unsigned (SmartScreen may warn)."
        return
    }
    $signtool = Find-SignTool
    if (-not $signtool) { throw "signtool.exe not found; install the Windows SDK or unset signing secrets." }
    $pfx = $null
    try {
        $args = @("sign", "/fd", "SHA256", "/tr", "http://timestamp.digicert.com", "/td", "SHA256")
        if ($pfxB64) {
            $pfx = Join-Path ([IO.Path]::GetTempPath()) "roboscout-codesign.pfx"
            [IO.File]::WriteAllBytes($pfx, [Convert]::FromBase64String($pfxB64))
            $args += @("/f", $pfx, "/p", $env:WINDOWS_CERT_PASSWORD)
        } else {
            $args += @("/a")
        }
        foreach ($f in $Files) {
            Write-Host "Signing $f"
            & $signtool @args $f
            if ($LASTEXITCODE -ne 0) { throw "signtool sign failed for $f" }
            & $signtool verify /pa $f
            if ($LASTEXITCODE -ne 0) { throw "signtool verify failed for $f" }
        }
    } finally {
        if ($pfx -and (Test-Path $pfx)) { Remove-Item $pfx -Force -ErrorAction SilentlyContinue }
    }
}

if (-not $Version) { $Version = Get-PyVersion }
if (-not $Channel) { $Channel = Get-Channel }
if (-not $MinManagerVersion) { $MinManagerVersion = Get-ManagerVersion }
if (-not $Commit) {
    try { $Commit = (git rev-parse HEAD 2>$null).Trim() } catch { $Commit = "" }
}

Write-Host "== RoboScoutAI Windows build: version=$Version channel=$Channel min_manager=$MinManagerVersion commit=$Commit"

if (-not $SkipInstall) {
    python -m pip install -U pip wheel
    python -m pip install -r requirements-desktop.txt
}

if (-not $NoClean) {
    Remove-Item -Recurse -Force build, dist -ErrorAction SilentlyContinue
}
$Release = Join-Path $Root "dist/release"
New-Item -ItemType Directory -Force -Path $Release | Out-Null

# 1. App
python -m PyInstaller --noconfirm --clean packaging/app.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller app.spec failed" }
Copy-Item dist/RoboScoutAI-app.exe (Join-Path $Release "RoboScoutAI-app-windows-x64.exe") -Force

# 2. Manager
python -m PyInstaller --noconfirm --clean packaging/manager.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller manager.spec failed" }
Copy-Item dist/RoboScoutAI.exe (Join-Path $Release "RoboScoutAI.exe") -Force

Sign-Files @((Join-Path $Release "RoboScoutAI-app-windows-x64.exe"), (Join-Path $Release "RoboScoutAI.exe"))

# 3. Manifest for the Setup payload (app + manager)
python -m roboscout_manager.manifest build `
    --version $Version --channel $Channel --commit $Commit --min-manager-version $MinManagerVersion `
    --out (Join-Path $Release "manifest.json") `
    (Join-Path $Release "RoboScoutAI-app-windows-x64.exe") (Join-Path $Release "RoboScoutAI.exe")
if ($LASTEXITCODE -ne 0) { throw "manifest build failed" }

# 4. Setup (bundles dist/release payload)
$env:ROBOSCOUT_SETUP_PAYLOAD = $Release
python -m PyInstaller --noconfirm --clean packaging/setup.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller setup.spec failed" }
Copy-Item dist/RoboScoutAI-Setup.exe (Join-Path $Release "RoboScoutAI-Setup.exe") -Force
Sign-Files @((Join-Path $Release "RoboScoutAI-Setup.exe"))

# 5. Final manifest (adds the Setup digest)
python -m roboscout_manager.manifest build `
    --version $Version --channel $Channel --commit $Commit --min-manager-version $MinManagerVersion `
    --out (Join-Path $Release "manifest.json") `
    (Join-Path $Release "RoboScoutAI-app-windows-x64.exe") (Join-Path $Release "RoboScoutAI.exe") (Join-Path $Release "RoboScoutAI-Setup.exe")
if ($LASTEXITCODE -ne 0) { throw "final manifest build failed" }

python -m roboscout_manager.manifest verify (Join-Path $Release "manifest.json") `
    (Join-Path $Release "RoboScoutAI-app-windows-x64.exe") (Join-Path $Release "RoboScoutAI.exe") (Join-Path $Release "RoboScoutAI-Setup.exe")
if ($LASTEXITCODE -ne 0) { throw "manifest verify failed" }

Write-Host "== Built:"
Get-ChildItem $Release | ForEach-Object { Write-Host ("   {0,-40} {1,12:N0} bytes" -f $_.Name, $_.Length) }
