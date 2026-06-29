param(
    [string]$Scene = "armadillo",
    [int]$Iterations = 30000,
    [int]$Resolution = 4,
    [string]$RunName = "gss_local_r4_i30000",
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

if (-not (Test-Path -LiteralPath $sourcePath)) {
    throw "Scene not found: $sourcePath"
}

$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

& $PythonExe .\train.py --eval `
    --skip_final_eval `
    -s $sourcePath `
    -m $modelPath `
    -r $Resolution `
    --iterations $Iterations `
    --lambda_normal_render_depth 0.0 `
    --lambda_normal_smooth 0.02 `
    --lambda_mask_entropy 0.1 `
    --save_training_vis `
    --save_training_vis_iteration 1000 `
    --densify_grad_normal_threshold 1e-8 `
    --lambda_depth_var 1e-2 `
    --save_interval $Iterations `
    --checkpoint_interval $Iterations `
    --test_interval 1000

if ($LASTEXITCODE -ne 0) {
    throw "Stage-1 local run failed."
}

$stopwatch.Stop()
$elapsed = $stopwatch.Elapsed
$summary = @(
    "scene=$Scene"
    "iterations=$Iterations"
    "resolution=$Resolution"
    "model_path=$modelPath"
    "elapsed=$($elapsed.ToString())"
    "elapsed_seconds=$([math]::Round($elapsed.TotalSeconds, 2))"
)
$summaryPath = Join-Path $modelPath "stage1_local_timing.txt"
$summary | Set-Content -LiteralPath $summaryPath -Encoding ascii

Write-Host "Stage-1 local run complete in $($elapsed.ToString())"
Write-Host "Timing summary written to $summaryPath"
