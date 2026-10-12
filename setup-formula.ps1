[CmdletBinding()]
param(
    [Parameter()][string]$ProjectRoot,
    [Parameter()][string]$LocalModelsPath,
    [Parameter()][double]$Timeout = 300.0
)

$ErrorActionPreference = 'Stop'

function Test-LocalAbsolutePath([string]$Value) {
    if ([string]::IsNullOrWhiteSpace($Value) -or $Value.IndexOf([char]0) -ge 0) { return $false }
    if ($Value.StartsWith('\\') -or $Value.StartsWith('//')) { return $false }
    return $Value -match '^[A-Za-z]:[\\/]'
}

if ([double]::IsNaN($Timeout) -or [double]::IsInfinity($Timeout) -or $Timeout -le 0) {
    throw 'Timeout must be finite and positive.'
}
if (-not $PSBoundParameters.ContainsKey('ProjectRoot')) {
    $ProjectRoot = [System.IO.Path]::GetFullPath($PSScriptRoot)
}
if (-not (Test-LocalAbsolutePath $ProjectRoot)) {
    throw 'ProjectRoot must be a trusted drive-qualified local absolute path (not UNC).'
}
if ($PSBoundParameters.ContainsKey('LocalModelsPath') -and -not (Test-LocalAbsolutePath $LocalModelsPath)) {
    throw 'LocalModelsPath must be a trusted drive-qualified local absolute path (not UNC).'
}

$cfgPath = Join-Path $ProjectRoot '.venv\pyvenv.cfg'
if (-not (Test-Path -LiteralPath $cfgPath -PathType Leaf)) {
    throw "Original .venv/pyvenv.cfg is missing: $cfgPath"
}
$config = @{}
foreach ($line in [System.IO.File]::ReadAllLines($cfgPath)) {
    if ($line -match '^\s*([^=]+?)\s*=\s*(.*?)\s*$') { $config[$matches[1].ToLowerInvariant()] = $matches[2] }
}
$baseValue = $config['executable']
if (-not $baseValue) { $baseValue = $config['home'] }
if (-not $baseValue) { throw 'Original .venv/pyvenv.cfg has no executable/home entry.' }
if ($baseValue -match '\.exe$') { $basePython = $baseValue }
else { $basePython = Join-Path $baseValue 'python.exe' }
if (-not (Test-LocalAbsolutePath $basePython) -or -not (Test-Path -LiteralPath $basePython -PathType Leaf)) {
    throw 'The original base Python executable recorded by .venv/pyvenv.cfg is unavailable or not a local absolute path.'
}

$helper = Join-Path $PSScriptRoot 'formula_setup.py'
$arguments = @('-I', '-S', '-B', '-X', 'utf8', $helper, '--project-root', $ProjectRoot, '--timeout', $Timeout.ToString([Globalization.CultureInfo]::InvariantCulture))
if ($PSBoundParameters.ContainsKey('LocalModelsPath')) {
    $arguments += @('--local-models-path', $LocalModelsPath)
}
$env:PYTHONDONTWRITEBYTECODE = '1'
& $basePython @arguments
$code = $LASTEXITCODE
if ($code -ne 0) { throw "Offline formula setup failed (base Python exit $code)." }
exit 0
