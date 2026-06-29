$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$pythonCandidates = @("C:\Users\c2483\anaconda3\envs\svgir37p\python.exe")
if ($env:CONDA_PREFIX) {
  $pythonCandidates += Join-Path $env:CONDA_PREFIX "python.exe"
}
$pythonCandidates += "python"

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
  if ($candidate -eq "python" -or (Test-Path $candidate)) {
    $pythonExe = $candidate
    break
  }
}

if (-not $pythonExe) {
  throw "Could not locate a Python interpreter for the smoke test."
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
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

& $pythonExe script/generate_tiny_blender_dataset.py

& $pythonExe train.py `
  --eval `
  -s dataset/tiny_blender/cube `
  -m output/tiny_blender/cube/gss_smoke `
  --iterations 200 `
  --checkpoint_interval 200 `
  --save_interval 200 `
  --test_interval 100 `
  --save_training_vis `
  --save_training_vis_iteration 100 `
  --lambda_mask_entropy 0.05 `
  --lambda_normal_smooth 0.01 `
  --densify_from_iter 50 `
  --densify_until_iter 0 `
  --densification_interval 50 `
  --max_points 20000 `
  --quiet

& $pythonExe eval_nvs.py `
  --eval `
  -m output/tiny_blender/cube/gss_smoke `
  -c output/tiny_blender/cube/gss_smoke/chkpnt200.pth
