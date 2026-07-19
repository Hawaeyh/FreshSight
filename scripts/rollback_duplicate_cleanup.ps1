param(
    [switch]$ConfirmRollback
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if (-not $ConfirmRollback) {
    throw "Rollback was not started. Rerun with -ConfirmRollback after confirming the audit log."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$RollbackScript = Join-Path $ProjectRoot "evaluation\rollback_duplicate_cleanup.py"
$AuditPath = Join-Path $ProjectRoot "evaluation\outputs\duplicate_cleanup_audit.csv"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}
if (-not (Test-Path -LiteralPath $AuditPath -PathType Leaf)) {
    throw "Cleanup audit log is missing; there is nothing to roll back."
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")
Push-Location $ProjectRoot
try {
    & $VenvPython $RollbackScript
    if ($LASTEXITCODE -ne 0) {
        throw "Duplicate-cleanup rollback failed (exit code $LASTEXITCODE)."
    }
}
finally {
    Pop-Location
}
