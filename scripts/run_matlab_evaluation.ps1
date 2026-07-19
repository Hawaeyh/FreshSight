$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Write-Host "Running held-out MATLAB rule-based evaluation..." -ForegroundColor Cyan
& python (Join-Path $PSScriptRoot "run_matlab_evaluation.py")
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB rule-based evaluation ended with exit code $LASTEXITCODE. Review the failed-image output."
}
