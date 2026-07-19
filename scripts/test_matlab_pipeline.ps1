param(
    [string]$ImagePath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Write-Host "Running the real FreshSight MATLAB pipeline test..." -ForegroundColor Cyan

$Arguments = @((Join-Path $PSScriptRoot "test_matlab_pipeline.py"))
if ($ImagePath) {
    $Arguments += @("--image", $ImagePath)
}
& python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "MATLAB pipeline test failed with exit code $LASTEXITCODE."
}
