Set-Location "D:\Grade Three Down\Experiment\SVG-IR"
$py = "C:\Users\c2483\anaconda3\envs\svgir37p\python.exe"
$src = "D:\Grade Three Down\Experiment\SVG-IR\dataset\TensoIR\armadillo"

$baseModel = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_nopalette_shorttrain_r4_i50000"
$baseCkpt = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_nopalette_shorttrain_r4_i31000\chkpnt31000.pth"
& $py .\train.py --eval --skip_final_eval -s $src -m $baseModel -c $baseCkpt -r 4 --iterations 50000 --checkpoint_interval 1000 --save_interval 1000 --test_interval 1000 --save_training_vis --save_training_vis_iteration 1000 --position_lr_init 0.0 --position_lr_final 0.0 --normal_lr 0.001 --sh_lr 0.00025 --opacity_lr 0.005 --scaling_lr 0.0 --rotation_lr 0.0 --lambda_base_color_smooth 0.1 --lambda_roughness_smooth 0.05 --lambda_light_smooth 0.0 --lambda_light 0.0 --lambda_env_smooth 0.02 --env_resolution 32 -t render_relight --sample_num 64 --densify_until_iter 0
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$palModel = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_palette_shorttrain_r4_i50000"
$palCkpt = "D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_palette_shorttrain_r4_i31000\chkpnt31000.pth"
& $py .\train.py --eval --skip_final_eval -s $src -m $palModel -c $palCkpt -r 4 --iterations 50000 --checkpoint_interval 1000 --save_interval 1000 --test_interval 1000 --save_training_vis --save_training_vis_iteration 1000 --position_lr_init 0.0 --position_lr_final 0.0 --normal_lr 0.001 --sh_lr 0.00025 --opacity_lr 0.005 --scaling_lr 0.0 --rotation_lr 0.0 --lambda_base_color_smooth 0.1 --lambda_roughness_smooth 0.05 --lambda_light_smooth 0.0 --lambda_light 0.0 --lambda_env_smooth 0.02 --env_resolution 32 -t render_relight --sample_num 64 --densify_until_iter 0 --use_palette_material --palette_debug
exit $LASTEXITCODE
