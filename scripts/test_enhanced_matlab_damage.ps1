$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "activate_environment.ps1")
Write-Host "=== FreshSight Controlled MATLAB Damage Development Test ==="
Write-Host "Only the 24 declared train/validation calibration samples are permitted."
Push-Location $projectRoot
try {
    & (Join-Path $projectRoot ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "test_enhanced_matlab_damage.py")
    if ($LASTEXITCODE -ne 0) { throw "Controlled MATLAB damage test failed." }
}
finally { Pop-Location }
