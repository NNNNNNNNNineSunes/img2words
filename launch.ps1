<#
.SYNOPSIS
    Start the Screen OCR GUI from the original, unlabelled base interpreter.

.DESCRIPTION
    .venv\Scripts\pythonw.exe lives inside the workspace and carries a low
    mandatory label, so starting that image drops the process to low integrity.
    This launcher never executes it: it starts <base>\pythonw.exe, and
    bootstrap.py prepends .venv\Lib\site-packages to sys.path instead.

    Nothing here elevates, edits ACLs, injects input or touches the process
    token.  No temp file and no log are written; the application owns logging.

.PARAMETER Console
    Run the GUI with the base console interpreter (python.exe).

.PARAMETER Diagnose
    Run bootstrap.py --diagnose and exit with its exit code.  No GUI.

.PARAMETER SmokeSeconds
    Pass --smoke-seconds N and wait for the exit code; never while the old
    instance is still running.
#>
[CmdletBinding()]
param(
    [switch]$Console,
    [switch]$Diagnose,
    [ValidateRange(0, 86400)]
    [double]$SmokeSeconds = 0
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# Process-scoped: only the interpreters below inherit this, nothing is written
# to the user or machine environment, and the GUI must not drop .pyc files.
$env:PYTHONDONTWRITEBYTECODE = '1'

$ProjectRoot = $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$VenvDir = Join-Path $ProjectRoot '.venv'
$Bootstrap = Join-Path $ProjectRoot 'bootstrap.py'
$Setup = Join-Path $ProjectRoot 'setup.ps1'
$Invariant = [System.Globalization.CultureInfo]::InvariantCulture

try {
    # setup.ps1 creates .venv, so a fresh checkout must not require it yet.
    foreach ($required in @($Bootstrap, $Setup)) {
        if (-not (Test-Path -LiteralPath $required)) {
            throw ('missing required file: ' + $required + ' (run setup.ps1 / start.cmd)')
        }
    }

    Write-Host 'Checking the Screen OCR environment...'
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Setup
    if ($LASTEXITCODE -ne 0) {
        throw ('setup.ps1 failed with exit code ' + $LASTEXITCODE)
    }
    if (-not (Test-Path -LiteralPath $VenvPython)) {
        throw ('setup.ps1 did not create ' + $VenvPython)
    }

    # Short console-only resolver; this low-integrity process never shows a GUI.
    $baseConsole = & $VenvPython -c 'import sys; print(sys._base_executable)'
    if ($LASTEXITCODE -ne 0 -or -not $baseConsole) {
        throw ('cannot read the base interpreter from ' + $VenvPython)
    }
    $baseConsole = ([string]($baseConsole | Select-Object -First 1)).Trim()
    if (-not (Test-Path -LiteralPath $baseConsole)) {
        throw ('the base interpreter does not exist: ' + $baseConsole)
    }
    $baseDir = Split-Path -Parent $baseConsole
    # Canonical paths: the resolver may answer with a short 8.3 path.
    $venvKey = [System.IO.Path]::GetFullPath($VenvDir).TrimEnd('\').ToLowerInvariant()
    $baseKey = [System.IO.Path]::GetFullPath($baseDir).TrimEnd('\').ToLowerInvariant()
    if ($baseKey -eq $venvKey -or ($baseKey + '\').StartsWith($venvKey + '\', [System.StringComparison]::Ordinal)) {
        throw ('refusing to launch the workspace interpreter: ' + $baseConsole) }
    $basePythonw = Join-Path $baseDir ([System.IO.Path]::GetFileNameWithoutExtension($baseConsole) + 'w.exe')

    $smokeArg = $SmokeSeconds.ToString($Invariant)
    if ($Diagnose -or $Console) {
        if ($Diagnose) { $extra = @('--diagnose') }
        elseif ($SmokeSeconds -gt 0) { $extra = @('--smoke-seconds', $smokeArg) }
        else { $extra = @() }
        Write-Host ('running bootstrap.py with ' + $baseConsole)
        & $baseConsole -X utf8 $Bootstrap @extra
        exit $LASTEXITCODE
    }
    # A Windows file name cannot contain a double quote, so wrapping the script
    # path is enough to keep it one argv element across spaces.
    $argumentList = '"' + $Bootstrap + '"'
    if ($SmokeSeconds -gt 0) { $argumentList += ' --smoke-seconds ' + $smokeArg }
    Write-Host ('starting the GUI from ' + $basePythonw)
    $process = Start-Process -FilePath $basePythonw -WorkingDirectory $ProjectRoot `
        -ArgumentList $argumentList -NoNewWindow -PassThru
    Write-Host ('Screen OCR GUI PID ' + $process.Id)
    if ($SmokeSeconds -gt 0) {
        if (-not $process.WaitForExit(60000)) {
            throw ('the smoke run did not exit within 60s; PID ' + $process.Id + ' is still running') }
        $process.WaitForExit()  # makes ExitCode reliable on Windows PowerShell 5.1
        Write-Host ('smoke run exit code: ' + $process.ExitCode)
        exit ([int]$process.ExitCode)
    }
    exit 0
} catch {
    [Console]::Error.WriteLine('launch.ps1: ' + $_.Exception.Message)
    exit 1
}
