# Build the toocomfy desktop app (Electron) on Windows.
# Produces desktop\dist\toocomfy-x64\ (portable).
#
# Prerequisite: Python 3.10+ and Node 18+ on PATH.
#
# Usage:
#     .\scripts\build_desktop.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot

# 1. Python backend in an isolated venv
$venv = Join-Path $root "venv"
if (-not (Test-Path $venv)) {
    python -m venv $venv
}
& (Join-Path $venv "Scripts\python.exe") -m pip install -q -r (Join-Path $root "requirements.txt")

# 2. Electron app
Push-Location (Join-Path $root "desktop")
npm install
npm run build
Pop-Location

Write-Host ""
Write-Host "Build output: $(Join-Path $root 'desktop\dist')" -ForegroundColor Green
