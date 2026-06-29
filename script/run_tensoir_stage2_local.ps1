param(
    [string]$Scene = "armadillo",
    [int]$StartIteration = 30000,
    [int]$EndIteration = 50000,
    [int]$Resolution = 4,
    [int]$SampleNum = 64,
    [int]$EnvResolution = 32,
    [string]$Stage1RunName = "gss_local_r4_i30000",
    [string]$RunName = "render_relight_local_r4_i50000",
    [string]$AliasDrive = "X:",
    [string]$PythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot

function Get-RepoRunRoot([string]$ActualRoot, [string]$DriveName) {
    if ($ActualRoot -notmatch '\s') {
        return $ActualRoot
    }

    $drive = $DriveName.TrimEnd(':') + ':'
    $existing = (cmd /c "subst $drive" 2>$null)
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($existing)) {
        cmd /c "subst $drive `"$ActualRoot`"" | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create subst drive $drive for $ActualRoot"
        }
    }

    return "$drive\"
}

$repoRunRoot = Get-RepoRunRoot -ActualRoot $repoRoot -DriveName $AliasDrive
Set-Location $repoRunRoot

. "$PSScriptRoot\setup_native_windows.ps1" -SkipCondaCuda -SkipPipDeps -SkipBuild | Out-Null

$sourcePath = Join-Path $repoRunRoot "dataset\TensoIR\$Scene"
$modelPath = Join-Path $repoRunRoot "output\TensoIR\$Scene\$RunName"
$checkpointPath = Join-Path $repoRunRoot "output\TensoIR\$Scene\$Stage1RunName\chkpnt$StartIteration.pth"

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Scene not found: $sourcePath"
}
if (-not (Test-Path -LiteralPath $checkpointPath)) {
    throw "Stage-1 checkpoint not found: $checkpointPath"
}
if ($EndIteration -lt $StartIteration) {
    throw "EndIteration must be >= StartIteration"
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

& $PythonExe .\train.py --eval `
    --skip_final_eval `
    -s $sourcePath `
    -m $modelPath `
    -c $checkpointPath `
    -r $Resolution `
    --iterations $EndIteration `
    --checkpoint_interval $EndIteration `
    --save_interval $EndIteration `
    --test_interval 1000 `
    --save_training_vis `
    --save_training_vis_iteration 1000 `
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
    throw "Stage-2 local run failed."
}

$stopwatch.Stop()
$elapsed = $stopwatch.Elapsed
$summary = @(
    "scene=$Scene"
    "start_iteration=$StartIteration"
    "end_iteration=$EndIteration"
    "resolution=$Resolution"
    "sample_num=$SampleNum"
    "env_resolution=$EnvResolution"
    "stage1_run=$Stage1RunName"
    "model_path=$modelPath"
    "elapsed=$($elapsed.ToString())"
    "elapsed_seconds=$([math]::Round($elapsed.TotalSeconds, 2))"
)
$summaryPath = Join-Path $modelPath "stage2_local_timing.txt"
$summary | Set-Content -LiteralPath $summaryPath -Encoding ascii

Write-Host "Stage-2 local run complete in $($elapsed.ToString())"
Write-Host "Timing summary written to $summaryPath"
