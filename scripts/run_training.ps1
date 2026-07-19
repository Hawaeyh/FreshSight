$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$TrainingScript = Join-Path $ProjectRoot "ai\train_model.py"
$ManifestPath = Join-Path $ProjectRoot "evaluation\outputs\dataset_manifest.csv"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $ManifestPath -PathType Leaf)) {
    throw "Dataset manifest is missing. Run scripts\run_dataset_split.ps1 first."
}
if (-not (Test-Path -LiteralPath $TrainingScript -PathType Leaf)) {
    throw "Training script is missing: $TrainingScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
& $VenvPython -c "import pandas, PIL, torch, torchvision"
if ($LASTEXITCODE -ne 0) {
    throw "Training dependencies are missing. Run scripts\setup_environment.ps1 again."
}

Write-Host "Starting the approved FreshSight MobileNetV2 baseline training." -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & $VenvPython $TrainingScript
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
