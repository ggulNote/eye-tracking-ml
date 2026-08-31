[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [string]$SideRoiRoot = "",
    [string]$DatasetRoot = "",
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
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = Join-Path $desktopRoot "participants"
}
if ([string]::IsNullOrWhiteSpace($SideRoiRoot)) {
    $SideRoiRoot = Join-Path $desktopRoot "new"
}
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = $desktopRoot
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot ".demo\precomputed_side_roi"
}

$resolvedSourceRoot = (Resolve-Path -LiteralPath $SourceRoot).Path
$resolvedSideRoiRoot = (Resolve-Path -LiteralPath $SideRoiRoot).Path
$resolvedDatasetRoot = (Resolve-Path -LiteralPath $DatasetRoot).Path
$manifest = Join-Path $OutputRoot "manifest.csv"
$previewOutput = Join-Path $OutputRoot "preview"
$trainingOutput = Join-Path $OutputRoot "outputs"
$mlflowDb = Join-Path $OutputRoot "mlflow.db"
New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$oldPythonPath = $env:PYTHONPATH
$oldDataRoot = $env:GAZE_DATA_ROOT
$oldManifest = $env:MEASURED_DATA_MANIFEST
$oldOutputRoot = $env:GAZE_OUTPUT_ROOT
$oldMlflow = $env:MLFLOW_TRACKING_URI
try {
    $env:PYTHONPATH = Join-Path $projectRoot "src"
    $env:GAZE_DATA_ROOT = $resolvedDatasetRoot
    $env:MEASURED_DATA_MANIFEST = $manifest
    $env:GAZE_OUTPUT_ROOT = $trainingOutput
    $env:MLFLOW_TRACKING_URI = "sqlite:///$($mlflowDb.Replace('\', '/'))"

    Write-Host "[MANIFEST] Matching external Side ROI images to measured pairs"
    & $venvPython (Join-Path $PSScriptRoot "create_measured_data_manifest.py") `
        --source-root $resolvedSourceRoot `
        --dataset-root $resolvedDatasetRoot `
        --side-roi-root $resolvedSideRoiRoot `
        --output-manifest $manifest `
        --sessions head_down neutral `
        --require-front-pose `
        --force
    if ($LASTEXITCODE -ne 0) {
        throw "precomputed Side ROI manifest 생성에 실패했습니다."
    }

    Write-Host "[PREVIEW] Auditing source sizes and rendering exact 128x256 inputs"
    & $venvPython (Join-Path $PSScriptRoot "preview_precomputed_side_roi.py") `
        --side-roi-root $resolvedSideRoiRoot `
        --output-dir $previewOutput `
        --sessions head_down neutral `
        --samples-per-session 1
    if ($LASTEXITCODE -ne 0) {
        throw "precomputed Side ROI preview 생성에 실패했습니다."
    }

    $profiles = @(
        "configs\profiles\blazegaze.yaml",
        "configs\profiles\side_profile_90.yaml",
        "configs\profiles\side_precomputed_roi.yaml",
        "configs\profiles\measured_head_down_neutral.yaml",
        "configs\models\front_webeyetrack.yaml",
        "configs\models\side_mobilenet_v4.yaml"
    )
    $profileArguments = @()
    foreach ($profile in $profiles) {
        $profileArguments += @("--profile", (Join-Path $projectRoot $profile))
    }

    Write-Host "[CONFIG] Validating precomputed Side ROI pipeline"
    & $venvPython -m gaze_pipeline validate-config `
        --config (Join-Path $projectRoot "configs\config.yaml") `
        @profileArguments
    if ($LASTEXITCODE -ne 0) {
        throw "precomputed Side ROI config 검증에 실패했습니다."
    }

    if (-not $SkipPrepare) {
        Write-Host "[PREPARE] Verifying image paths, pairs, and subject split"
        & $venvPython -m gaze_pipeline prepare `
            --config (Join-Path $projectRoot "configs\config.yaml") `
            @profileArguments
        if ($LASTEXITCODE -ne 0) {
            throw "precomputed Side ROI 데이터 준비에 실패했습니다."
        }
    }

    Write-Host "[OUTPUT] Manifest: $manifest"
    Write-Host "[OUTPUT] ROI audit: $(Join-Path $previewOutput 'precomputed_side_roi_audit.json')"
    Write-Host "[RULE] Missing subject/session side_eye_roi folders were excluded."
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:GAZE_DATA_ROOT = $oldDataRoot
    $env:MEASURED_DATA_MANIFEST = $oldManifest
    $env:GAZE_OUTPUT_ROOT = $oldOutputRoot
    $env:MLFLOW_TRACKING_URI = $oldMlflow
}
