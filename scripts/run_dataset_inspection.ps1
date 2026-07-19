$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$InspectionScript = Join-Path $ProjectRoot "evaluation\inspect_dataset.py"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $InspectionScript -PathType Leaf)) {
    throw "Dataset inspection script is missing: $InspectionScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
& $VenvPython -c "import pandas, PIL, sklearn"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset-inspection dependencies are missing. Run scripts\setup_environment.ps1 again."
}

Push-Location $ProjectRoot
try {
    & $VenvPython $InspectionScript
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset inspection reported errors (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
