$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RunScript = Join-Path $ProjectRoot "run_web.py"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $RunScript -PathType Leaf)) {
    throw "FreshSight launcher is missing: $RunScript"
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
& $VenvPython -c "import flask, torch, torchvision; from PIL import Image"
if ($LASTEXITCODE -ne 0) {
    throw "FreshSight dependencies are missing. Run scripts\setup_environment.ps1 again."
}

Write-Host "Starting FreshSight at http://127.0.0.1:5000" -ForegroundColor Cyan
Push-Location $ProjectRoot
try {
    & $VenvPython $RunScript
    if ($LASTEXITCODE -ne 0) {
        throw "FreshSight exited with code $LASTEXITCODE."
    }
}
finally {
    Pop-Location
}
