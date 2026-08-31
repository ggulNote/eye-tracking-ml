[CmdletBinding()]
param(
    [string]$ProcessDataRoot = "",
    [string]$ReferenceRoot = "",
    [string]$OutputRoot = "",
    [switch]$SkipPrepare
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$desktopRoot = Split-Path -Parent $projectRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "Python 가상환경이 없습니다. 먼저 scripts\setup_windows.ps1을 실행하세요."
}
if ([string]::IsNullOrWhiteSpace($ProcessDataRoot)) {
    $ProcessDataRoot = Join-Path $desktopRoot "process_data"
}
if ([string]::IsNullOrWhiteSpace($ReferenceRoot)) {
    $ReferenceRoot = Join-Path $desktopRoot "Participants"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot ".demo\process_data_audit"
}

$resolvedDataRoot = (Resolve-Path -LiteralPath $ProcessDataRoot).Path
$resolvedReferenceRoot = (Resolve-Path -LiteralPath $ReferenceRoot).Path
$auditOutput = Join-Path $OutputRoot "report"
$manifest = Join-Path $OutputRoot "manifest.csv"
$rejections = Join-Path $OutputRoot "rejections.csv"
$qualityExclusions = Join-Path $auditOutput "quality_exclusions.csv"
$trainingOutput = Join-Path $OutputRoot "outputs"
$mlflowDb = Join-Path $OutputRoot "mlflow.db"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$oldPythonPath = $env:PYTHONPATH
$oldDataRoot = $env:PROCESS_DATA_ROOT
$oldManifest = $env:PROCESS_DATA_MANIFEST
$oldOutputRoot = $env:GAZE_OUTPUT_ROOT
$oldMlflow = $env:MLFLOW_TRACKING_URI
try {
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    $env:PROCESS_DATA_ROOT = $resolvedDataRoot
    $env:PROCESS_DATA_MANIFEST = $manifest
    $env:GAZE_OUTPUT_ROOT = $trainingOutput
    $env:MLFLOW_TRACKING_URI = "sqlite:///$($mlflowDb.Replace('\', '/'))"

    Write-Host "[AUDIT] Checking all Front/Side ROI images and inputs.csv rows"
    & $venvPython -m scripts.audit_process_data `
        --process-data-root $resolvedDataRoot `
        --reference-root $resolvedReferenceRoot `
        --output-dir $auditOutput `
        --sessions head_down neutral `
        --samples-per-session 2 `
        --summary-only
    if ($LASTEXITCODE -ne 0) {
        throw "process_data audit에 실패했습니다."
    }

    Write-Host "[MANIFEST] Applying valid=1, complete-pair, and quality-exclusion rules"
    & $venvPython -m scripts.create_process_data_manifest `
        --process-data-root $resolvedDataRoot `
        --output-manifest $manifest `
        --rejection-csv $rejections `
        --quality-exclusions $qualityExclusions `
        --sessions head_down neutral `
        --force
    if ($LASTEXITCODE -ne 0) {
        throw "process_data manifest 생성에 실패했습니다."
    }

    $profiles = @(
        "configs\profiles\blazegaze.yaml",
        "configs\profiles\front_precomputed_eye_roi.yaml",
        "configs\profiles\side_profile_90.yaml",
        "configs\profiles\side_precomputed_roi.yaml",
        "configs\profiles\process_data.yaml",
        "configs\models\front_webeyetrack.yaml",
        "configs\models\side_mobilenet_v4.yaml"
    )
    $profileArguments = @()
    foreach ($profile in $profiles) {
        $profileArguments += @("--profile", (Join-Path $projectRoot $profile))
    }

    Write-Host "[CONFIG] Validating process_data direct-ROI pipeline"
    & $venvPython -m gaze_pipeline validate-config `
        --config (Join-Path $projectRoot "configs\config.yaml") `
        @profileArguments
    if ($LASTEXITCODE -ne 0) {
        throw "process_data config 검증에 실패했습니다."
    }

    if (-not $SkipPrepare) {
        Write-Host "[PREPARE] Verifying paths, pairs, and subject-wise split"
        & $venvPython -m gaze_pipeline prepare `
            --config (Join-Path $projectRoot "configs\config.yaml") `
            @profileArguments
        if ($LASTEXITCODE -ne 0) {
            throw "process_data prepare에 실패했습니다."
        }
    }

    Write-Host "[OUTPUT] Audit: $(Join-Path $auditOutput 'process_data_audit.json')"
    Write-Host "[OUTPUT] Missingness: $(Join-Path $auditOutput 'missingness.csv')"
    Write-Host "[OUTPUT] Manifest: $manifest"
    Write-Host "[RULE] valid=0, incomplete pairs, missing sessions, and quality-excluded sessions are not trained."
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PROCESS_DATA_ROOT = $oldDataRoot
    $env:PROCESS_DATA_MANIFEST = $oldManifest
    $env:GAZE_OUTPUT_ROOT = $oldOutputRoot
    $env:MLFLOW_TRACKING_URI = $oldMlflow
}
