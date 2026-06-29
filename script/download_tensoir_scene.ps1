param(
  [ValidateSet("armadillo", "ficus", "hotdog", "lego")]
  [string]$Scene = "armadillo"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$sceneMeta = @{
  "armadillo" = @{
    Url = "https://zenodo.org/records/7880113/files/armadillo.zip?download=1"
    Md5 = "d1b0cc19d73fbb2af6aea7bcb1e44781"
  }
  "ficus" = @{
    Url = "https://zenodo.org/records/7880113/files/ficus.zip?download=1"
    Md5 = "d1cdb8d726b03bf736d3c85ccd08e9ce"
  }
  "hotdog" = @{
    Url = "https://zenodo.org/records/7880113/files/hotdog.zip?download=1"
    Md5 = "0d0cc5833abff879e045a958eaaa1da0"
  }
  "lego" = @{
    Url = "https://zenodo.org/records/7880113/files/lego.zip?download=1"
    Md5 = "181da25d84bdcec54f6b3c9e4006266c"
  }
}

$datasetRoot = Join-Path $root "dataset\TensoIR"
$sceneRoot = Join-Path $datasetRoot $Scene
$zipPath = Join-Path $datasetRoot "$Scene.zip"

if ((Test-Path (Join-Path $sceneRoot "transforms_train.json")) -and (Test-Path (Join-Path $sceneRoot "transforms_test.json"))) {
  Write-Output "Scene already prepared at $sceneRoot"
  exit 0
}

New-Item -ItemType Directory -Force -Path $datasetRoot | Out-Null

if (-not (Test-Path $zipPath)) {
  Write-Output "Downloading $Scene from Zenodo..."
} else {
  $existingSize = (Get-Item -LiteralPath $zipPath).Length
  Write-Output "Resuming existing download for $Scene ($existingSize bytes already present)..."
}

& curl.exe -L -C - $sceneMeta[$Scene].Url -o $zipPath
if ($LASTEXITCODE -ne 0) {
  throw "curl download failed with exit code $LASTEXITCODE. Partial file kept at $zipPath for resume."
}

$actualMd5 = (Get-FileHash -Algorithm MD5 -LiteralPath $zipPath).Hash.ToLowerInvariant()
if ($actualMd5 -ne $sceneMeta[$Scene].Md5) {
  throw "MD5 mismatch for $zipPath. Expected $($sceneMeta[$Scene].Md5), got $actualMd5. The file is likely incomplete or corrupted."
}

if (Test-Path $sceneRoot) {
  $existingChildren = Get-ChildItem -LiteralPath $sceneRoot -Force -ErrorAction SilentlyContinue
  if ($existingChildren) {
    Remove-Item -LiteralPath $sceneRoot -Recurse -Force
  }
}

New-Item -ItemType Directory -Force -Path $sceneRoot | Out-Null
Write-Output "Extracting $zipPath ..."
tar -xf $zipPath -C $datasetRoot

# Some official archives unpack directly into datasetRoot rather than a scene-named folder.
if ((-not (Test-Path (Join-Path $sceneRoot "transforms_train.json"))) -and (Test-Path (Join-Path $datasetRoot "transforms_train.json"))) {
  Write-Output "Normalizing flat extraction layout into $sceneRoot ..."
  $sceneItems = Get-ChildItem -LiteralPath $datasetRoot -Force | Where-Object {
    $_.Name -notin @($Scene, "$Scene.zip")
  }
  foreach ($item in $sceneItems) {
    Move-Item -LiteralPath $item.FullName -Destination $sceneRoot
  }
}

if (-not (Test-Path (Join-Path $sceneRoot "transforms_train.json"))) {
  $nestedCandidate = Get-ChildItem -LiteralPath $sceneRoot -Directory -ErrorAction SilentlyContinue | Where-Object {
    Test-Path (Join-Path $_.FullName "transforms_train.json")
  } | Select-Object -First 1
  if ($nestedCandidate) {
    Write-Output "Flattening nested extraction layout from $($nestedCandidate.FullName) ..."
    Get-ChildItem -LiteralPath $nestedCandidate.FullName -Force | Move-Item -Destination $sceneRoot
    Remove-Item -LiteralPath $nestedCandidate.FullName -Recurse -Force
  }
}

if (-not (Test-Path (Join-Path $sceneRoot "transforms_train.json"))) {
  throw "Could not find transforms_train.json after extraction in either flat or nested layout."
}

Write-Output "Prepared scene at $sceneRoot"
