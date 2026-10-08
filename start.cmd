@echo off
setlocal
cd /d "%~dp0"
rem launch.ps1 runs setup.ps1 itself and starts the GUI from the original
rem unlabelled base interpreter. Never start the workspace interpreter image
rem (.venv\Scripts) here: that image carries a low-integrity label.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" %*
if errorlevel 1 (
  echo Launch failed. See the error above.
  pause
  exit /b 1
)
