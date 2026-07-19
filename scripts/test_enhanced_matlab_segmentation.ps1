$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "activate_environment.ps1")
Write-Host "=== FreshSight Enhanced MATLAB Segmentation Development Test ==="
Write-Host "Only train/validation samples declared in config/development_segmentation_samples.json are permitted."
Push-Location $projectRoot
try {
    & (Join-Path $projectRoot ".venv\Scripts\python.exe") (Join-Path $PSScriptRoot "test_enhanced_matlab_segmentation.py")
    if ($LASTEXITCODE -ne 0) { throw "Enhanced MATLAB segmentation test failed." }
}
finally {
    Pop-Location
}
