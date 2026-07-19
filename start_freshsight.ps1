$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$SetupScript = Join-Path $ProjectRoot "scripts\setup_environment.ps1"
$RunScript = Join-Path $ProjectRoot "scripts\run_freshsight.ps1"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    Write-Host "FreshSight .venv is missing; starting environment setup." -ForegroundColor Yellow
    & $SetupScript
    if ($LASTEXITCODE -ne 0) {
        throw "Environment setup failed. FreshSight was not started."
    }
}

& $RunScript
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
