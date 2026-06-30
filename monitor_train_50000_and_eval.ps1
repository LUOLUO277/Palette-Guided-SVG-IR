$ErrorActionPreference = "Stop"

$repo = "D:\Grade Three Down\Experiment\SVG-IR"
$pythonExe = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
$sourcePath = Join-Path $repo "dataset\TensoIR\armadillo"

$baselineDir = Join-Path $repo "output\TensoIR\armadillo\render_relight_nopalette_shorttrain_r4_i50000"
$paletteDir = Join-Path $repo "output\TensoIR\armadillo\render_relight_palette_shorttrain_r4_i50000"
$baselineCkpt = Join-Path $baselineDir "chkpnt50000.pth"
$paletteCkpt = Join-Path $paletteDir "chkpnt50000.pth"

$monitorLog = Join-Path $repo "monitor_train_50000_and_eval.log"
$reportPath = Join-Path $repo "compare_report_50000.txt"

function Write-Log([string]$Message) {
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    "$ts $Message" | Tee-Object -FilePath $monitorLog -Append | Out-Null
}

function Read-AlbedoPsnr([string]$JsonPath) {
    if (-not (Test-Path -LiteralPath $JsonPath)) {
        return $null
    }
    $json = Get-Content -LiteralPath $JsonPath -Raw | ConvertFrom-Json
    return $json.summary.averages.albedo_psnr
}

Set-Location $repo
Write-Log "Monitor started."
Write-Log "Waiting for $baselineCkpt"
Write-Log "Waiting for $paletteCkpt"

while ((-not (Test-Path -LiteralPath $baselineCkpt)) -or (-not (Test-Path -LiteralPath $paletteCkpt))) {
    $baseReady = Test-Path -LiteralPath $baselineCkpt
    $palReady = Test-Path -LiteralPath $paletteCkpt
    Write-Log "Checkpoint status: baseline=$baseReady palette=$palReady"
    Start-Sleep -Seconds 120
}

Write-Log "Both checkpoints detected. Starting eval_nvs exports."

& $pythonExe .\eval_nvs.py --eval -s $sourcePath -m $baselineDir -c $baselineCkpt -t render_relight --skip_train
if ($LASTEXITCODE -ne 0) {
    throw "Baseline eval_nvs failed."
}

& $pythonExe .\eval_nvs.py --eval -s $sourcePath -m $paletteDir -c $paletteCkpt -t render_relight --skip_train
if ($LASTEXITCODE -ne 0) {
    throw "Palette eval_nvs failed."
}

Write-Log "eval_nvs finished. Starting albedo evaluation."

$baselineAlbedoJson = Join-Path $baselineDir "test\ours_50000\metrics_albedo.json"
$baselineAlbedoTxt = Join-Path $baselineDir "test\ours_50000\metrics_albedo.txt"
$paletteAlbedoJson = Join-Path $paletteDir "test\ours_50000\metrics_albedo.json"
$paletteAlbedoTxt = Join-Path $paletteDir "test\ours_50000\metrics_albedo.txt"

& $pythonExe .\eval_albedo.py `
    --pred_dir (Join-Path $baselineDir "test\ours_50000\base_color") `
    --gt_dir $sourcePath `
    --mask_dir $sourcePath `
    --out_json $baselineAlbedoJson `
    --out_txt $baselineAlbedoTxt `
    --skip_lpips
if ($LASTEXITCODE -ne 0) {
    throw "Baseline eval_albedo failed."
}

& $pythonExe .\eval_albedo.py `
    --pred_dir (Join-Path $paletteDir "test\ours_50000\base_color") `
    --gt_dir $sourcePath `
    --mask_dir $sourcePath `
    --out_json $paletteAlbedoJson `
    --out_txt $paletteAlbedoTxt `
    --skip_lpips
if ($LASTEXITCODE -ne 0) {
    throw "Palette eval_albedo failed."
}

Write-Log "Albedo evaluation finished. Writing summary report."

$baselineAlbedoPsnr = Read-AlbedoPsnr -JsonPath $baselineAlbedoJson
$paletteAlbedoPsnr = Read-AlbedoPsnr -JsonPath $paletteAlbedoJson
$baselineTestLoss = if (Test-Path -LiteralPath (Join-Path $baselineDir "test_loss.txt")) { Get-Content -LiteralPath (Join-Path $baselineDir "test_loss.txt") -Raw } else { "missing" }
$paletteTestLoss = if (Test-Path -LiteralPath (Join-Path $paletteDir "test_loss.txt")) { Get-Content -LiteralPath (Join-Path $paletteDir "test_loss.txt") -Raw } else { "missing" }

$report = @(
    "Compare Report 50000"
    "generated_at=$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
    ""
    "baseline_dir=$baselineDir"
    "baseline_ckpt=$baselineCkpt"
    "baseline_test_loss=$baselineTestLoss"
    "baseline_albedo_psnr=$baselineAlbedoPsnr"
    "baseline_metric_test=$(Join-Path $baselineDir 'metric_test.txt')"
    "baseline_vis_render=$(Join-Path $baselineDir 'test\ours_50000\renders\00000.png')"
    "baseline_vis_base_color=$(Join-Path $baselineDir 'test\ours_50000\base_color\00000.png')"
    "baseline_vis_roughness=$(Join-Path $baselineDir 'test\ours_50000\roughness\00000.png')"
    ""
    "palette_dir=$paletteDir"
    "palette_ckpt=$paletteCkpt"
    "palette_test_loss=$paletteTestLoss"
    "palette_albedo_psnr=$paletteAlbedoPsnr"
    "palette_metric_test=$(Join-Path $paletteDir 'metric_test.txt')"
    "palette_vis_render=$(Join-Path $paletteDir 'test\ours_50000\renders\00000.png')"
    "palette_vis_base_color=$(Join-Path $paletteDir 'test\ours_50000\base_color\00000.png')"
    "palette_vis_roughness=$(Join-Path $paletteDir 'test\ours_50000\roughness\00000.png')"
    ""
    "baseline_metrics_albedo=$baselineAlbedoTxt"
    "palette_metrics_albedo=$paletteAlbedoTxt"
)

$report | Set-Content -LiteralPath $reportPath -Encoding ascii
Write-Log "Summary report written to $reportPath"
Write-Log "Monitor finished."
