@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run setup_demo_windows.bat first.
  exit /b 1
)

rem Change 0 and 1 below if Windows assigned different camera numbers.
.venv\Scripts\python.exe -m gaze_pipeline demo --front-camera 0 --side-camera 1 --show-cameras %*
endlocal
