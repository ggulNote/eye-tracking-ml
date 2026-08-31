@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
  echo [ERROR] Python launcher was not found. Install Python 3.12 first.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3.12 -m venv .venv
  if errorlevel 1 exit /b 1
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip "setuptools>=77" wheel
if errorlevel 1 exit /b 1
python -m pip install -r requirements-demo.txt
if errorlevel 1 exit /b 1
python -m pip install -e . --no-deps --no-build-isolation
if errorlevel 1 exit /b 1

echo.
echo Setup complete. Next run verify_demo_windows.bat.
endlocal
