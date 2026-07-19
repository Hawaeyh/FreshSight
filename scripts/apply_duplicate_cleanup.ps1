param(
    [switch]$ConfirmCleanup
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ConfirmCleanup) {
    throw "Cleanup was not applied. Review the preview, then rerun with -ConfirmCleanup after explicit approval."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$ApplyScript = Join-Path $ProjectRoot "evaluation\apply_duplicate_cleanup.py"
$PlanPath = Join-Path $ProjectRoot "evaluation\outputs\duplicate_cleanup_plan.csv"
$SummaryPath = Join-Path $ProjectRoot "evaluation\outputs\duplicate_cleanup_summary.json"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) {
    throw "Cleanup plan is missing. Run scripts\preview_duplicate_cleanup.ps1 first."
}
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Cleanup summary is missing. Run scripts\preview_duplicate_cleanup.ps1 again."
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Push-Location $ProjectRoot
try {
    & $VenvPython $ApplyScript
    if ($LASTEXITCODE -ne 0) {
        throw "Duplicate cleanup failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
