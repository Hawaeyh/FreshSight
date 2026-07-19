$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $ProjectRoot "scripts\setup_environment.ps1"
$InspectionLauncher = Join-Path $ProjectRoot "scripts\run_dataset_inspection.ps1"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host "FreshSight .venv is missing; starting environment setup." -ForegroundColor Yellow
    & $SetupScript
    if ($LASTEXITCODE -ne 0) {
        throw "Environment setup failed. Dataset inspection was not started."
    }
}

& $InspectionLauncher
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
