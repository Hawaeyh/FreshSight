$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$PreviewScript = Join-Path $ProjectRoot "evaluation\prepare_duplicate_cleanup.py"
$InspectionReport = Join-Path $ProjectRoot "evaluation\outputs\dataset_inspection.json"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $InspectionReport -PathType Leaf)) {
    throw "Dataset inspection report is missing. Run inspect_dataset.ps1 first."
}
if (-not (Test-Path -LiteralPath $PreviewScript -PathType Leaf)) {
    throw "Duplicate-cleanup preview script is missing: $PreviewScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Push-Location $ProjectRoot
try {
    & $VenvPython $PreviewScript
    if ($LASTEXITCODE -ne 0) {
        throw "Duplicate-cleanup preview failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
