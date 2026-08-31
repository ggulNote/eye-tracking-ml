@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run setup_demo_windows.bat first.
  exit /b 1
)
.venv\Scripts\python.exe -m gaze_pipeline demo --verify-only %*
endlocal
