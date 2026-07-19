$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $VenvPython -PathType Leaf)) {
    throw "FreshSight .venv is missing. Run scripts\setup_environment.ps1 first."
}

. (Join-Path $PSScriptRoot "activate_environment.ps1")

$CudaCheck = @'
import sys
import torch

print('=== FreshSight CUDA Check ===')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA build version: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA device count: {torch.cuda.device_count()}')

if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    print('FAILED: CUDA is required, and FreshSight will not fall back to CPU.', file=sys.stderr)
    raise SystemExit(1)

device = torch.device('cuda:0')
properties = torch.cuda.get_device_properties(device)
free_bytes, total_bytes = torch.cuda.mem_get_info(device)
print(f'GPU name: {torch.cuda.get_device_name(device)}')
print(f'Total GPU memory: {properties.total_memory / 1024**3:.2f} GiB ({properties.total_memory} bytes)')
print(f'Currently available GPU memory: {free_bytes / 1024**3:.2f} GiB')

left = torch.rand((512, 512), device=device)
right = torch.rand((512, 512), device=device)
result = left @ right
torch.cuda.synchronize(device)
print(f'Tensor result device: {result.device}')
print('CUDA tensor operation: PASS')
del left, right, result
torch.cuda.empty_cache()
print('=== CUDA Check PASSED ===')
'@

& $VenvPython -c $CudaCheck
if ($LASTEXITCODE -ne 0) {
    throw "FreshSight CUDA check failed. Training must not be started."
}
