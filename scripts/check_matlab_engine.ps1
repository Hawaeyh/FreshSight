$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $PSScriptRoot "activate_environment.ps1")

Write-Host "Running the real MATLAB Engine check..." -ForegroundColor Cyan
& python (Join-Path $PSScriptRoot "check_matlab_engine.py")
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB Engine check failed with exit code $LASTEXITCODE."
}
