[CmdletBinding()]
param(
    [string]$PythonPath = "",
    [switch]$Dev
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

function Find-Python312 {
    param([string]$RequestedPath)

    $candidates = [System.Collections.Generic.List[string]]::new()
    if (-not [string]::IsNullOrWhiteSpace($RequestedPath)) {
        $candidates.Add($RequestedPath)
    }
    $candidates.Add((Join-Path $projectRoot ".python312\python.exe"))
    $command = Get-Command python3.12 -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        $candidates.Add($command.Source)
    }
    $candidates.Add((Join-Path $env:LocalAppData "Programs\Python\Python312\python.exe"))
    $candidates.Add((Join-Path $env:ProgramFiles "Python312\python.exe"))

    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate -PathType Leaf -ErrorAction SilentlyContinue)) {
            continue
        }
        $version = & $candidate -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
        if ($LASTEXITCODE -eq 0 -and $version.Trim() -eq "3.12") {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    throw @"
Python 3.12 실행 파일을 찾지 못했습니다.
Python 3.12를 설치한 뒤 아래처럼 절대경로를 지정하세요.
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -PythonPath "C:\path\to\python.exe" -Dev
"@
}

$python = Find-Python312 -RequestedPath $PythonPath
$venv = Join-Path $projectRoot ".venv"
$venvPython = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    Write-Host "[SETUP] Python 3.12 virtual environment: $venv"
    & $python -m venv $venv
    if ($LASTEXITCODE -ne 0) {
        throw "가상환경 생성에 실패했습니다."
    }
}

& $venvPython -m pip install --no-cache-dir --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "pip 업그레이드에 실패했습니다."
}
& $venvPython -m pip install --no-cache-dir -r (Join-Path $projectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "runtime 의존성 설치에 실패했습니다."
}
& $venvPython -m pip install --no-cache-dir -e $projectRoot
if ($LASTEXITCODE -ne 0) {
    throw "프로젝트 editable 설치에 실패했습니다."
}
if ($Dev) {
    & $venvPython -m pip install --no-cache-dir -r (Join-Path $projectRoot "requirements-dev.txt")
    if ($LASTEXITCODE -ne 0) {
        throw "개발·테스트 의존성 설치에 실패했습니다."
    }
}

& $venvPython (Join-Path $PSScriptRoot "check_environment.py")
if ($LASTEXITCODE -ne 0) {
    throw "환경 검사에 실패했습니다."
}
Write-Host "[OK] Windows setup complete: $venvPython"
