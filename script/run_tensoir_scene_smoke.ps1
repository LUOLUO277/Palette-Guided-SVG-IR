param(
  [ValidateSet("armadillo", "ficus", "hotdog", "lego")]
  [string]$Scene = "armadillo",
  [int]$Iterations = 200
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
if (-not (Test-Path $pythonExe)) {
  if ($env:CONDA_PREFIX -and (Test-Path (Join-Path $env:CONDA_PREFIX "python.exe"))) {
    $pythonExe = Join-Path $env:CONDA_PREFIX "python.exe"
  } else {
    $pythonExe = "python"
  }
}

$vendorRoot = Join-Path $root "vendor\site-packages"
New-Item -ItemType Directory -Force -Path $vendorRoot | Out-Null

function Expand-EggIfNeeded {
  param(
    [string]$EggPath,
    [string]$ProbePath
  )

  $probe = Join-Path $vendorRoot $ProbePath
  if (-not (Test-Path $probe)) {
    tar -xf $EggPath -C $vendorRoot
  }
}

Expand-EggIfNeeded "svgss_rasterization\dist\rgss_rasterization-0.0.0-py3.7-win-amd64.egg" "rgss_rasterization\_C.cp37-win_amd64.pyd"
Expand-EggIfNeeded "svgss_rasterization\dist\svgss_rasterization-0.0.0-py3.7-win-amd64.egg" "svgss_rasterization\_C.cp37-win_amd64.pyd"
Expand-EggIfNeeded "svgss_rasterization\dist\diff_gaussian_rasterization-0.0.0-py3.7-win-amd64.egg" "diff_gaussian_rasterization\_C.cp37-win_amd64.pyd"

$env:PYTHONPATH = "$vendorRoot;$root"
$env:PATH = "$vendorRoot\rgss_rasterization;$vendorRoot\svgss_rasterization;$vendorRoot\diff_gaussian_rasterization;$env:PATH"
$env:MPLCONFIGDIR = Join-Path $root ".mplconfig"
$env:SVGIR_USE_PYTORCH_FALLBACK = "1"
$env:SVGIR_SMOKE_NUM_PTS = "10000"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

$sceneRoot = "dataset/TensoIR/$Scene"
if (-not (Test-Path (Join-Path $root "$sceneRoot\transforms_train.json"))) {
  throw "Scene not found at $sceneRoot. Run script/download_tensoir_scene.ps1 first."
}

$modelRoot = "output/TensoIR/$Scene/gss_smoke_i$Iterations"

& $pythonExe train.py `
  --eval `
  -s $sceneRoot `
  -m $modelRoot `
  --resolution 8 `
  --iterations $Iterations `
  --checkpoint_interval $Iterations `
  --save_interval $Iterations `
  --test_interval 100 `
  --save_training_vis `
  --save_training_vis_iteration 100 `
  --lambda_normal_render_depth 0.0 `
  --lambda_normal_smooth 0.02 `
  --lambda_mask_entropy 0.1 `
  --densify_grad_normal_threshold 1e-8 `
  --lambda_depth_var 1e-2 `
  --densify_until_iter 0 `
  --quiet
if ($LASTEXITCODE -ne 0) {
  throw "train.py failed with exit code $LASTEXITCODE"
}

& $pythonExe eval_nvs.py `
  --eval `
  -m $modelRoot `
  -c "$modelRoot/chkpnt$Iterations.pth"
if ($LASTEXITCODE -ne 0) {
  throw "eval_nvs.py failed with exit code $LASTEXITCODE"
}
