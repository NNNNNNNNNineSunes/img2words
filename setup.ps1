$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$env:PIP_DISABLE_PIP_VERSION_CHECK = '1'
$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$requirements = Join-Path $PSScriptRoot 'requirements-lock.txt'
if (-not (Test-Path -LiteralPath $requirements)) {
    $requirements = Join-Path $PSScriptRoot 'requirements.txt'
}
if (-not (Test-Path -LiteralPath $python)) {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.11 -m venv .venv
    } elseif (Get-Command python -ErrorAction SilentlyContinue) {
        & python -c "import sys; assert sys.version_info[:2] == (3, 11), 'Install Python 3.11 first'; assert sys.maxsize > 2**32, 'Use 64-bit Python'"
        if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 (64-bit) is required.' }
        & python -m venv .venv
    } else {
        throw 'Install Python 3.11 (64-bit) from python.org, including Tcl/Tk and pip.'
    }
    if ($LASTEXITCODE -ne 0) { throw 'Virtual environment creation failed.' }
}
& $python -c "import sys, tkinter; assert sys.version_info[:2] == (3, 11); assert sys.maxsize > 2**32"
if ($LASTEXITCODE -ne 0) { throw 'The project environment must use 64-bit Python 3.11 with Tkinter.' }
$stamp = Join-Path $PSScriptRoot '.venv\installed-requirements.sha256'
$hash = (Get-FileHash -LiteralPath $requirements -Algorithm SHA256).Hash
$installed = (Test-Path -LiteralPath $stamp) -and ((Get-Content -LiteralPath $stamp -Raw).Trim() -eq $hash)
if (-not $installed) {
    & $python -m pip install --no-cache-dir -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed. Check the network and run setup.ps1 again.' }
}
& $python -c "import tkinter, PIL, mss, onnxruntime, rapidocr_onnxruntime; from pathlib import Path; import rapidocr_onnxruntime as r; assert len(list((Path(r.__file__).parent / 'models').glob('*.onnx'))) >= 3, 'Bundled OCR models missing'"
if ($LASTEXITCODE -ne 0) { throw 'Dependency or OCR model check failed. Run .venv\Scripts\python.exe -m pip install --force-reinstall -r requirements-lock.txt' }
Set-Content -LiteralPath $stamp -Value $hash -Encoding ASCII
Write-Host 'Screen OCR environment is ready.'
