$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$RequirementsPath = Join-Path $ProjectRoot "requirements.txt"

try {
    Write-Host "=== FreshSight Environment Setup ===" -ForegroundColor Cyan
    Write-Host "Project root: $ProjectRoot"

    if (-not (Test-Path -LiteralPath $RequirementsPath -PathType Leaf)) {
        throw "requirements.txt was not found at: $RequirementsPath"
    }

    $BootstrapCommand = $null
    $BootstrapArguments = @()
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0) {
            $BootstrapCommand = "py"
            $BootstrapArguments = @("-3.11")
        }
    }
    if (-not $BootstrapCommand -and (Get-Command python -ErrorAction SilentlyContinue)) {
        $BootstrapCommand = "python"
    }
    if (-not $BootstrapCommand) {
        throw "Python 3.11 was not found. Install Python 3.11 and ensure either 'py' or 'python' is available on PATH."
    }

    $Version = & $BootstrapCommand @BootstrapArguments -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
    if ($LASTEXITCODE -ne 0) {
        throw "Unable to query the Python version."
    }
    & $BootstrapCommand @BootstrapArguments -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)"
    if ($LASTEXITCODE -ne 0) {
        throw "FreshSight requires Python 3.11. Detected Python $Version."
    }
    Write-Host "Compatible Python detected: $Version" -ForegroundColor Green

    if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
        if (Test-Path -LiteralPath $VenvPath) {
            throw ".venv exists but is incomplete. Move or remove it, then run setup again: $VenvPath"
        }
        Write-Host "Creating virtual environment: $VenvPath"
        & $BootstrapCommand @BootstrapArguments -m venv $VenvPath
        if ($LASTEXITCODE -ne 0) {
            throw "Python failed to create .venv."
        }
    }
    else {
        Write-Host "Using existing virtual environment: $VenvPath"
    }

    . (Join-Path $VenvPath "Scripts\Activate.ps1")
    Write-Host "Upgrading pip..."
    & $VenvPython -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
        throw "pip upgrade failed."
    }

    Write-Host "Installing requirements..."
    & $VenvPython -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency installation failed."
    }

    Write-Host "Environment setup completed successfully." -ForegroundColor Green
    Write-Host "Virtual-environment Python: $VenvPython"
    & $VenvPython --version
}
catch {
    Write-Error "FreshSight environment setup failed: $($_.Exception.Message)"
    exit 1
}
