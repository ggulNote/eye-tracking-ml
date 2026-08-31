[CmdletBinding()]
param(
    [string]$DataRoot = "",
    [string]$OutputRoot = "",
    [string]$ReviewOverrides = "",
    [switch]$AcceptHaar,
    [switch]$RequireComplete
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw @"
Python 3.12 가상환경이 없습니다. 먼저 실행하세요.
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1 -PythonPath "C:\path\to\python.exe" -Dev
"@
}

if ([string]::IsNullOrWhiteSpace($DataRoot)) {
    $DataRoot = Join-Path (Split-Path -Parent $projectRoot) "participants"
}
$resolvedDataRoot = (Resolve-Path -LiteralPath $DataRoot).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $projectRoot ".demo\measured_head_down_neutral"
}
$roiOutput = Join-Path $OutputRoot "side_roi_detection"
$annotationCsv = Join-Path $OutputRoot "side_annotations.csv"
$modelRoot = Join-Path $projectRoot "models"
$modelAsset = Join-Path $modelRoot "face_landmarker_v2_with_blendshapes.task"
$yunetModel = Join-Path $modelRoot "face_detection_yunet_2023mar.onnx"
$auditOutput = Join-Path $projectRoot "outputs\preprocessing_missingness_after_side_roi"

$oldPythonPath = $env:PYTHONPATH
try {
    $sourceRoot = Join-Path $projectRoot "src"
    $env:PYTHONPATH = if ([string]::IsNullOrWhiteSpace($oldPythonPath)) {
        $sourceRoot
    } else {
        "$sourceRoot;$oldPythonPath"
    }

    if (-not (Test-Path -LiteralPath $modelAsset -PathType Leaf)) {
        Write-Host "[ASSET] Downloading and verifying MediaPipe/WebEyeTrack assets"
        & $venvPython (Join-Path $PSScriptRoot "fetch_webeyetrack_assets.py") `
            --output-dir $modelRoot
        if ($LASTEXITCODE -ne 0) {
            throw "모델 asset 준비에 실패했습니다."
        }
    }

    $arguments = @(
        (Join-Path $PSScriptRoot "generate_side_eye_annotations.py"),
        "--data-root", $resolvedDataRoot,
        "--output-dir", $roiOutput,
        "--output-csv", $annotationCsv,
        "--model-asset", $modelAsset,
        "--yunet-model", $yunetModel,
        "--sessions", "head_down", "neutral",
        "--detector", "hybrid",
        "--force"
    )
    if (-not [string]::IsNullOrWhiteSpace($ReviewOverrides)) {
        $arguments += @("--review-overrides", (Resolve-Path -LiteralPath $ReviewOverrides).Path)
    }
    if ($AcceptHaar) {
        $arguments += "--accept-haar"
    }
    if ($RequireComplete) {
        $arguments += "--require-complete"
    }

    Write-Host "[SIDE ROI] Detecting and validating eye bboxes"
    & $venvPython @arguments
    $generatorExitCode = $LASTEXITCODE
    if ($generatorExitCode -notin @(0, 3)) {
        throw "Side bbox 생성에 실패했습니다. exit=$generatorExitCode"
    }

    Write-Host "[AUDIT] Rechecking missingness with generated Side annotations"
    & $venvPython (Join-Path $PSScriptRoot "audit_preprocessing_missingness.py") `
        --data-root $resolvedDataRoot `
        --output-dir $auditOutput `
        --sessions head_down neutral `
        --side-annotations $annotationCsv
    if ($LASTEXITCODE -ne 0) {
        throw "후속 결측치 검사에 실패했습니다."
    }

    Write-Host "[OUTPUT] Annotation CSV: $annotationCsv"
    Write-Host "[OUTPUT] Review queue: $(Join-Path $roiOutput 'side_annotations_review_queue.csv')"
    Write-Host "[OUTPUT] Audit summary: $(Join-Path $auditOutput 'summary.json')"
    if ($generatorExitCode -eq 3) {
        throw "자동 승인되지 않은 bbox가 남아 있습니다. review queue를 검수하세요."
    }
} finally {
    $env:PYTHONPATH = $oldPythonPath
}
