param(
    [string]$Scene = "armadillo",
    [int]$StartIteration = 20,
    [int]$EndIteration = 22,
    [int]$SampleNum = 8,
    [int]$EnvResolution = 16,
    [string]$PythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

. "$PSScriptRoot\setup_native_windows.ps1" -SkipCondaCuda -SkipPipDeps -SkipBuild | Out-Null

$sourcePath = Join-Path $repoRoot "dataset\TensoIR\$Scene"
$modelPath = Join-Path $repoRoot "output\TensoIR\$Scene\render_relight_native_smoke"
$checkpointPath = Join-Path $repoRoot "output\TensoIR\$Scene\gss_smoke_i20\chkpnt$StartIteration.pth"

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Scene not found: $sourcePath"
}
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Stage-1 checkpoint not found: $checkpointPath"
}
if ($EndIteration -lt $StartIteration) {
    throw "EndIteration must be >= StartIteration"
}

& $PythonExe .\train.py --eval `
    --skip_final_eval `
    -s $sourcePath `
    -m $modelPath `
    -c $checkpointPath `
    --iterations $EndIteration `
    --checkpoint_interval 1 `
    --save_interval $EndIteration `
    --test_interval 1000 `
    --save_training_vis `
    --save_training_vis_iteration $EndIteration `
    --position_lr_init 0.0 `
    --position_lr_final 0.0 `
    --normal_lr 0.001 `
    --sh_lr 0.00025 `
    --opacity_lr 0.005 `
    --scaling_lr 0.0 `
    --rotation_lr 0.0 `
    --lambda_base_color_smooth 0.1 `
    --lambda_roughness_smooth 0.05 `
    --lambda_light_smooth 0.0 `
    --lambda_light 0.0 `
    --lambda_env_smooth 0.02 `
    --env_resolution $EnvResolution `
    -t render_relight `
    --sample_num $SampleNum

if ($LASTEXITCODE -ne 0) {
    throw "render_relight native smoke run failed."
}
