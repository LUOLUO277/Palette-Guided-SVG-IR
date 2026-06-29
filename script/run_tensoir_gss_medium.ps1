param(
    [string]$Scene = "armadillo",
    [int]$Iterations = 500,
    [string]$RunName = "gss_medium_i500",
    [string]$PythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot\setup_native_windows.ps1" -SkipCondaCuda -SkipPipDeps -SkipBuild | Out-Null

$sourcePath = Join-Path $repoRoot "dataset\TensoIR\$Scene"
$modelPath = Join-Path $repoRoot "output\TensoIR\$Scene\$RunName"

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Scene not found: $sourcePath"
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

& $PythonExe .\train.py --eval `
    --skip_final_eval `
    -s $sourcePath `
    -m $modelPath `
    --iterations $Iterations `
    --lambda_normal_render_depth 0.0 `
    --lambda_normal_smooth 0.02 `
    --lambda_mask_entropy 0.1 `
    --save_training_vis `
    --save_training_vis_iteration $Iterations `
    --densify_grad_normal_threshold 1e-8 `
    --lambda_depth_var 1e-2 `
    --save_interval $Iterations `
    --checkpoint_interval $Iterations `
    --test_interval 1000

if ($LASTEXITCODE -ne 0) {
    throw "Stage-1 medium run failed."
}

$stopwatch.Stop()
$elapsed = $stopwatch.Elapsed
$summary = @(
    "scene=$Scene"
    "iterations=$Iterations"
    "model_path=$modelPath"
    "elapsed=$($elapsed.ToString())"
    "elapsed_seconds=$([math]::Round($elapsed.TotalSeconds, 2))"
)
$summaryPath = Join-Path $modelPath "stage1_medium_timing.txt"
$summary | Set-Content -LiteralPath $summaryPath -Encoding ascii

Write-Host "Stage-1 medium run complete in $($elapsed.ToString())"
Write-Host "Timing summary written to $summaryPath"
