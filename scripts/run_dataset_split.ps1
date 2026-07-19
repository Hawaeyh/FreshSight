$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$SplitScript = Join-Path $ProjectRoot "evaluation\split_dataset.py"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $SplitScript -PathType Leaf)) {
    throw "Dataset split script is missing: $SplitScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
& $VenvPython -c "import pandas, PIL, torch"
if ($LASTEXITCODE -ne 0) {
    throw "Dataset-splitting dependencies are missing. Run scripts\setup_environment.ps1 again."
}

Push-Location $ProjectRoot
try {
    & $VenvPython $SplitScript
    if ($LASTEXITCODE -ne 0) {
        throw "Dataset splitting reported errors (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
