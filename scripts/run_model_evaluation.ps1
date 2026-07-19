$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$EvaluationScript = Join-Path $ProjectRoot "evaluation\evaluate_mobilenetv2.py"
$BestCheckpoint = Join-Path $ProjectRoot "ai\checkpoints\mobilenetv2_cleaned_baseline\best_model.pth"

if (-not (Test-Path -LiteralPath $BestCheckpoint -PathType Leaf)) {
    throw "Best cleaned-baseline checkpoint is missing: $BestCheckpoint"
}
if (-not (Test-Path -LiteralPath $EvaluationScript -PathType Leaf)) {
    throw "Evaluation script is missing: $EvaluationScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Write-Host "Running held-out MobileNetV2 evaluation without deployment..." -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & python $EvaluationScript
    if ($LASTEXITCODE -ne 0) {
        throw "MobileNetV2 evaluation failed with exit code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
