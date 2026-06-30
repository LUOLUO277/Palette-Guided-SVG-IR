Set-Location "D:\Grade Three Down\Experiment\SVG-IR"
$py = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
$src = "D:\Grade Three Down\Experiment\SVG-IR\dataset\TensoIR\armadillo"
$baseModel = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_nopalette_shorttrain_r4_i50000"
$baseCkpt = Join-Path $baseModel 'chkpnt50000.pth'
$palModel = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_palette_shorttrain_r4_i50000"
$palCkpt = Join-Path $palModel 'chkpnt50000.pth'
$summary = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\compare_50000_summary.txt"

function Wait-Checkpoint([string]$path, [string]$tag) {
  while (-not (Test-Path -LiteralPath $path)) {
    Start-Sleep -Seconds 60
  }
  Write-Output "[$tag] checkpoint ready: $path"
}

Wait-Checkpoint $baseCkpt 'baseline'
Wait-Checkpoint $palCkpt 'palette'

& $py .\eval_nvs.py --source_path $src --model_path $baseModel --checkpoint $baseCkpt -t render_relight --skip_train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py .\eval_nvs.py --source_path $src --model_path $palModel --checkpoint $palCkpt -t render_relight --skip_train
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $py .\eval_albedo.py --pred_dir (Join-Path $baseModel 'test\ours_50000\base_color') --gt_dir $src --mask_dir $src --out_json (Join-Path $baseModel 'test\ours_50000\metrics_albedo.json') --out_txt (Join-Path $baseModel 'test\ours_50000\metrics_albedo.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py .\eval_albedo.py --pred_dir (Join-Path $palModel 'test\ours_50000\base_color') --gt_dir $src --mask_dir $src --out_json (Join-Path $palModel 'test\ours_50000\metrics_albedo.json') --out_txt (Join-Path $palModel 'test\ours_50000\metrics_albedo.txt')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$baseMetric = Get-Content -LiteralPath (Join-Path $baseModel 'metric_test.txt') -Raw
$palMetric = Get-Content -LiteralPath (Join-Path $palModel 'metric_test.txt') -Raw
$baseAlb = Get-Content -LiteralPath (Join-Path $baseModel 'test\ours_50000\metrics_albedo.txt') -Raw
$palAlb = Get-Content -LiteralPath (Join-Path $palModel 'test\ours_50000\metrics_albedo.txt') -Raw
@(
  'Baseline metric_test.txt'
  $baseMetric
  ''
  'Palette metric_test.txt'
  $palMetric
  ''
  'Baseline metrics_albedo.txt'
  $baseAlb
  ''
  'Palette metrics_albedo.txt'
  $palAlb
) | Set-Content -LiteralPath $summary -Encoding UTF8
Write-Output "[done] summary written: $summary"
