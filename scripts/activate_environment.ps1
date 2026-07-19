$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ActivateScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (-not (Test-Path -LiteralPath $ActivateScript -PathType Leaf)) {
    throw "FreshSight .venv was not found. Run scripts\setup_environment.ps1 first."
}

. $ActivateScript
$ActivePython = (Get-Command python -ErrorAction Stop).Source
Write-Host "FreshSight environment activated." -ForegroundColor Green
Write-Host "Python executable: $ActivePython"
& python --version
Write-Host "To keep this environment active in your current terminal, dot-source this script:"
Write-Host ". .\scripts\activate_environment.ps1"
