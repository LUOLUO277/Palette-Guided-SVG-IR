from native_path_setup import ensure_local_native_module_paths

ensure_local_native_module_paths()

import argparse
import json
import os
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from lpipsPyTorch import lpips
from utils.loss_utils import ssim
from utils.image_utils import mse, psnr


def parse_args():
    parser = argparse.ArgumentParser(description="Offline albedo / roughness evaluation")
    parser.add_argument("--pred_dir", type=str, required=True, help="Predicted albedo directory")
    parser.add_argument("--gt_dir", type=str, required=True, help="GT albedo directory or TensoIR scene root")
    parser.add_argument("--mask_dir", type=str, default=None, help="GT mask directory or TensoIR scene root")
    parser.add_argument("--alpha_dir", type=str, default=None, help="Optional predicted alpha directory")
    parser.add_argument("--out_json", type=str, default=None, help="Output json path")
    parser.add_argument("--out_txt", type=str, default=None, help="Output txt path")
    parser.add_argument("--alpha_thresh", type=float, default=0.5)
    parser.add_argument("--pred_roughness_dir", type=str, default=None)
    parser.add_argument("--gt_roughness_dir", type=str, default=None)
    parser.add_argument("--pred_rgb_dir", type=str, default=None)
    parser.add_argument("--gt_rgb_dir", type=str, default=None)
    parser.add_argument("--gt_name", type=str, default="albedo.png")
    parser.add_argument("--mask_name", type=str, default="rgba.png")
    parser.add_argument("--gt_roughness_name", type=str, default="roughness.png")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_lpips", action="store_true", default=False)
    parser.add_argument(
        "--target_resolution",
        type=str,
        choices=["pred", "gt"],
        default="pred",
        help="Resize GT to predicted resolution ('pred') or resize predictions to GT resolution ('gt').",
    )
    return parser.parse_args()


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".exr"}


def list_image_files(folder):
    if folder is None or not os.path.isdir(folder):
        return []
    files = []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and Path(name).suffix.lower() in IMG_EXTS:
            files.append(path)
    return files


def load_image_any(path):
    suffix = Path(path).suffix.lower()
    if suffix == ".exr":
        import pyexr
        arr = pyexr.open(path).get()
    else:
        arr = imageio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 2:
        arr = arr[..., None]
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(info.max)
    else:
        arr = arr.astype(np.float32)
    return arr


def split_rgb_alpha(arr):
    if arr.shape[-1] >= 4:
        return arr[..., :3], arr[..., 3:4]
    if arr.shape[-1] == 1:
        return np.repeat(arr, 3, axis=-1), None
    return arr[..., :3], None


def clamp01(arr):
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def resize_image(arr, hw, is_mask=False):
    h, w = hw
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_AREA
    if arr.ndim == 3 and arr.shape[-1] == 1:
        out = cv2.resize(arr[..., 0], (w, h), interpolation=interp)[..., None]
    else:
        out = cv2.resize(arr, (w, h), interpolation=interp)
        if out.ndim == 2:
            out = out[..., None]
    return out.astype(np.float32)


def is_numeric_stem(stem):
    try:
        int(stem)
        return True
    except ValueError:
        return False


def resolve_sample_path(root_dir, stem, filename):
    if root_dir is None:
        return None
    direct = os.path.join(root_dir, f"{stem}{Path(filename).suffix}")
    if os.path.exists(direct):
        return direct
    named = os.path.join(root_dir, filename)
    if os.path.exists(named):
        return named
    subdir = os.path.join(root_dir, stem, filename)
    if os.path.exists(subdir):
        return subdir
    if is_numeric_stem(stem):
        idx = int(stem)
        tensoir = os.path.join(root_dir, f"test_{idx:03d}", filename)
        if os.path.exists(tensoir):
            return tensoir
    return None


def make_mask(gt_mask, pred_alpha, gt_rgb, alpha_thresh):
    notes = []
    final_mask = None
    if gt_mask is not None:
        final_mask = gt_mask > 0.5
        notes.append("gt_mask")
        if pred_alpha is not None:
            final_mask = np.logical_and(final_mask, pred_alpha > alpha_thresh)
            notes.append("and_pred_alpha")
    elif pred_alpha is not None:
        final_mask = pred_alpha > alpha_thresh
        notes.append("pred_alpha_only")
    else:
        final_mask = np.any(gt_rgb > 1.0 / 255.0, axis=-1, keepdims=True)
        notes.append("fallback_gt_nonzero")
    return final_mask.astype(np.float32), "+".join(notes)


def masked_tensor(arr):
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def compute_masked_psnr(pred, gt, mask):
    pred_t = masked_tensor(pred * mask)
    gt_t = masked_tensor(gt * mask)
    return float(psnr(pred_t, gt_t).mean().item())


def compute_masked_ssim(pred, gt, mask):
    pred_t = masked_tensor(pred * mask)
    gt_t = masked_tensor(gt * mask)
    return float(ssim(pred_t, gt_t).item())


def compute_masked_lpips(pred, gt, mask, device):
    pred_t = masked_tensor(pred * mask).to(device)
    gt_t = masked_tensor(gt * mask).to(device)
    return float(lpips(pred_t, gt_t, net_type="vgg").mean().item())


def compute_masked_mse(pred, gt, mask):
    pred_t = masked_tensor(pred * mask)
    gt_t = masked_tensor(gt * mask)
    return float(mse(pred_t, gt_t).mean().item())


def safe_mean(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return float(np.mean(values))


def main():
    args = parse_args()

    pred_files = list_image_files(args.pred_dir)
    if not pred_files:
        raise FileNotFoundError(f"No predicted images found in {args.pred_dir}")

    out_json = args.out_json or os.path.join(args.pred_dir, "metrics_albedo.json")
    out_txt = args.out_txt or os.path.join(args.pred_dir, "metrics_albedo.txt")
    out_json_dir = os.path.dirname(out_json)
    out_txt_dir = os.path.dirname(out_txt)
    if out_json_dir:
        os.makedirs(out_json_dir, exist_ok=True)
    if out_txt_dir:
        os.makedirs(out_txt_dir, exist_ok=True)

    lpips_enabled = not args.skip_lpips
    if lpips_enabled:
        try:
            _ = torch.zeros(1)
        except Exception:
            lpips_enabled = False

    per_image = []
    skipped = []
    warnings_list = []

    for pred_path in pred_files:
        stem = Path(pred_path).stem
        record = {
            "pred_path": pred_path,
            "stem": stem,
        }

        gt_albedo_path = resolve_sample_path(args.gt_dir, stem, args.gt_name)
        if gt_albedo_path is None:
            skipped.append({"pred_path": pred_path, "reason": "missing_gt_albedo"})
            continue

        pred_rgba = load_image_any(pred_path)
        pred_rgb, pred_alpha_from_pred = split_rgb_alpha(pred_rgba)
        pred_rgb = clamp01(pred_rgb)

        gt_rgba = load_image_any(gt_albedo_path)
        gt_rgb, gt_alpha_from_gt = split_rgb_alpha(gt_rgba)
        gt_rgb = clamp01(gt_rgb)

        alpha_path = resolve_sample_path(args.alpha_dir, stem, f"{stem}.png") if args.alpha_dir else None
        pred_alpha = None
        if alpha_path is not None:
            pred_alpha = load_image_any(alpha_path)[..., :1]
        elif pred_alpha_from_pred is not None:
            pred_alpha = pred_alpha_from_pred

        gt_mask_path = resolve_sample_path(args.mask_dir, stem, args.mask_name) if args.mask_dir else None
        gt_mask = None
        if gt_mask_path is not None:
            mask_rgba = load_image_any(gt_mask_path)
            if mask_rgba.shape[-1] >= 4:
                gt_mask = mask_rgba[..., 3:4]
            else:
                gt_mask = mask_rgba[..., :1]
        elif gt_alpha_from_gt is not None:
            gt_mask = gt_alpha_from_gt

        target_hw = pred_rgb.shape[:2] if args.target_resolution == "pred" else gt_rgb.shape[:2]
        if pred_rgb.shape[:2] != target_hw:
            warnings_list.append(
                f"[Warning] Resize pred albedo {pred_path} from {pred_rgb.shape[:2]} to {target_hw} with INTER_AREA."
            )
            pred_rgb = resize_image(pred_rgb, target_hw, is_mask=False)
        if gt_rgb.shape[:2] != target_hw:
            warnings_list.append(
                f"[Warning] Resize gt albedo {gt_albedo_path} from {gt_rgb.shape[:2]} to {target_hw} with INTER_AREA."
            )
            gt_rgb = resize_image(gt_rgb, target_hw, is_mask=False)
            if gt_alpha_from_gt is not None and gt_mask_path is None:
                gt_alpha_from_gt = resize_image(gt_alpha_from_gt, target_hw, is_mask=True)
        if gt_mask is not None and gt_mask.shape[:2] != target_hw:
            warnings_list.append(
                f"[Warning] Resize gt mask {gt_mask_path or gt_albedo_path} from {gt_mask.shape[:2]} to {target_hw} with INTER_NEAREST."
            )
            gt_mask = resize_image(gt_mask, target_hw, is_mask=True)

        if pred_alpha is not None and pred_alpha.shape[:2] != target_hw:
            warnings_list.append(
                f"[Warning] Resize pred alpha for {pred_path} from {pred_alpha.shape[:2]} to {target_hw} with INTER_NEAREST."
            )
            pred_alpha = resize_image(pred_alpha, target_hw, is_mask=True)

        mask, mask_source = make_mask(gt_mask, pred_alpha, gt_rgb, args.alpha_thresh)
        if mask.sum() <= 0:
            skipped.append({"pred_path": pred_path, "reason": "empty_mask"})
            continue

        record["gt_albedo_path"] = gt_albedo_path
        record["mask_path"] = gt_mask_path
        record["mask_source"] = mask_source
        record["mask_pixels"] = int(mask.sum())
        record["albedo_psnr"] = compute_masked_psnr(pred_rgb, gt_rgb, mask)
        record["albedo_ssim"] = compute_masked_ssim(pred_rgb, gt_rgb, mask)
        record["albedo_lpips"] = compute_masked_lpips(pred_rgb, gt_rgb, mask, args.device) if lpips_enabled else None

        if args.pred_roughness_dir and args.gt_roughness_dir:
            pred_rough_path = resolve_sample_path(args.pred_roughness_dir, stem, f"{stem}.png")
            gt_rough_path = resolve_sample_path(args.gt_roughness_dir, stem, args.gt_roughness_name)
            if pred_rough_path and gt_rough_path:
                pred_rough = clamp01(load_image_any(pred_rough_path)[..., :1])
                gt_rough = clamp01(load_image_any(gt_rough_path)[..., :1])
                if pred_rough.shape[:2] != target_hw:
                    warnings_list.append(
                        f"[Warning] Resize pred roughness {pred_rough_path} from {pred_rough.shape[:2]} to {target_hw} with INTER_AREA."
                    )
                    pred_rough = resize_image(pred_rough, target_hw, is_mask=False)
                if gt_rough.shape[:2] != target_hw:
                    warnings_list.append(
                        f"[Warning] Resize gt roughness {gt_rough_path} from {gt_rough.shape[:2]} to {target_hw} with INTER_AREA."
                    )
                    gt_rough = resize_image(gt_rough, target_hw, is_mask=False)
                record["roughness_mse"] = compute_masked_mse(pred_rough, gt_rough, mask)
            else:
                record["roughness_mse"] = None
        else:
            record["roughness_mse"] = None

        if args.pred_rgb_dir and args.gt_rgb_dir:
            pred_rgb_path = resolve_sample_path(args.pred_rgb_dir, stem, f"{stem}.png")
            gt_rgb_path = resolve_sample_path(args.gt_rgb_dir, stem, f"{stem}.png")
            if pred_rgb_path and gt_rgb_path:
                pred_render, _ = split_rgb_alpha(load_image_any(pred_rgb_path))
                gt_render, _ = split_rgb_alpha(load_image_any(gt_rgb_path))
                pred_render = clamp01(pred_render)
                gt_render = clamp01(gt_render)
                if pred_render.shape[:2] != target_hw:
                    warnings_list.append(
                        f"[Warning] Resize pred rgb {pred_rgb_path} from {pred_render.shape[:2]} to {target_hw} with INTER_AREA."
                    )
                    pred_render = resize_image(pred_render, target_hw, is_mask=False)
                if gt_render.shape[:2] != target_hw:
                    warnings_list.append(
                        f"[Warning] Resize gt rgb {gt_rgb_path} from {gt_render.shape[:2]} to {target_hw} with INTER_AREA."
                    )
                    gt_render = resize_image(gt_render, target_hw, is_mask=False)
                record["rgb_psnr"] = compute_masked_psnr(pred_render, gt_render, mask)
            else:
                record["rgb_psnr"] = None
        else:
            record["rgb_psnr"] = None

        per_image.append(record)

    summary = {
        "num_pred_files": len(pred_files),
        "num_evaluated": len(per_image),
        "num_skipped": len(skipped),
        "lpips_enabled": lpips_enabled,
        "target_resolution": args.target_resolution,
        "averages": {
            "albedo_psnr": safe_mean([x["albedo_psnr"] for x in per_image]),
            "albedo_ssim": safe_mean([x["albedo_ssim"] for x in per_image]),
            "albedo_lpips": safe_mean([x["albedo_lpips"] for x in per_image]),
            "roughness_mse": safe_mean([x["roughness_mse"] for x in per_image]),
            "rgb_psnr": safe_mean([x["rgb_psnr"] for x in per_image]),
        },
        "warnings": warnings_list,
        "skipped": skipped,
    }

    payload = {
        "summary": summary,
        "per_image": per_image,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Albedo Evaluation Summary\n")
        f.write(f"pred_dir: {args.pred_dir}\n")
        f.write(f"gt_dir: {args.gt_dir}\n")
        f.write(f"mask_dir: {args.mask_dir}\n")
        f.write(f"alpha_dir: {args.alpha_dir}\n")
        f.write(f"num_pred_files: {summary['num_pred_files']}\n")
        f.write(f"num_evaluated: {summary['num_evaluated']}\n")
        f.write(f"num_skipped: {summary['num_skipped']}\n")
        f.write(f"lpips_enabled: {summary['lpips_enabled']}\n")
        f.write(f"target_resolution: {summary['target_resolution']}\n")
        for key, value in summary["averages"].items():
            f.write(f"{key}: {value}\n")
        if warnings_list:
            f.write("\nWarnings:\n")
            for item in warnings_list:
                f.write(f"- {item}\n")
        if skipped:
            f.write("\nSkipped:\n")
            for item in skipped:
                f.write(f"- {item['pred_path']}: {item['reason']}\n")

    print(f"[AlbedoEval] evaluated={summary['num_evaluated']} skipped={summary['num_skipped']} lpips_enabled={lpips_enabled}")
    for key, value in summary["averages"].items():
        print(f"[AlbedoEval] {key}={value}")
    if warnings_list:
        for item in warnings_list:
            print(item)
    if skipped:
        for item in skipped:
            print(f"[AlbedoEval][Skip] {item['pred_path']} :: {item['reason']}")


if __name__ == "__main__":
    main()



