from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_EXPERIMENT_ROOT = Path(
    r"D:\Grade Three Down\Experiment\SVG-IR\output\TensoIR\armadillo\render_relight_local_r4_i50000"
)
DEFAULT_OUTPUT_DIRNAME = "paper_figures"
IMAGE_SIZE = (320, 320)

METRICS = {
    "PSNR_PBR": {"fireplace": 32.0956, "forest": 30.0573, "night": 32.1737},
    "SSIM_PBR": {"fireplace": 0.9422, "forest": 0.9349, "night": 0.9276},
    "LPIPS_PBR": {"fireplace": 0.0781, "forest": 0.0863, "night": 0.0889},
}


def log_info(message: str) -> None:
    print(f"[INFO] {message}")


def log_warning(message: str) -> None:
    print(f"[WARNING] {message}")


def log_error(message: str) -> None:
    print(f"[ERROR] {message}")


def natural_sort_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", path.stem.lower())
    key: list[object] = []
    for part in parts:
        key.append(int(part) if part.isdigit() else part)
    key.append(path.suffix.lower())
    return key


def list_images(directory: Path, required: bool = True) -> list[Path]:
    if not directory.exists():
        message = f"Directory does not exist: {directory}"
        if required:
            raise FileNotFoundError(message)
        log_warning(message)
        return []
    if not directory.is_dir():
        message = f"Path is not a directory: {directory}"
        if required:
            raise NotADirectoryError(message)
        log_warning(message)
        return []

    images = sorted(
        [path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=natural_sort_key,
    )
    if not images:
        message = f"No images found in directory: {directory}"
        if required:
            raise FileNotFoundError(message)
        log_warning(message)
        return []
    return images


def find_common_filenames(directories: Sequence[Path], required: bool = True) -> list[str]:
    filename_sets: list[set[str]] = []
    for directory in directories:
        images = list_images(directory, required=required)
        if not images:
            if required:
                raise FileNotFoundError(f"No images available for directory: {directory}")
            return []
        filename_sets.append({image.name for image in images})

    common = set.intersection(*filename_sets) if filename_sets else set()
    if not common:
        message = "No common image filenames across directories:\n" + "\n".join(str(path) for path in directories)
        if required:
            raise FileNotFoundError(message)
        log_warning(message)
        return []

    return sorted((Path(name) for name in common), key=natural_sort_key)


def sample_evenly(items: Sequence[str], count: int) -> list[str]:
    if not items:
        return []
    if len(items) <= count:
        return list(items)
    indexes = np.linspace(0, len(items) - 1, count, dtype=int)
    return [items[index] for index in indexes]


def prepare_image(image_path: Path, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    with Image.open(image_path) as image:
        rgb_image = image.convert("RGB")
        fitted = ImageOps.contain(rgb_image, image_size, method=Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", image_size, color=(255, 255, 255))
        x_offset = (image_size[0] - fitted.width) // 2
        y_offset = (image_size[1] - fitted.height) // 2
        canvas.paste(fitted, (x_offset, y_offset))
        return np.asarray(canvas)


def save_grid_figure(
    image_rows: Sequence[Sequence[Path | None]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    output_path: Path,
    title: str,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> None:
    n_rows = len(image_rows)
    n_cols = len(col_labels)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 2.8, n_rows * 2.8 + 0.8),
        squeeze=False,
        facecolor="white",
    )
    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.06, wspace=0.05, hspace=0.10)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    for row_index, row in enumerate(image_rows):
        for col_index, image_path in enumerate(row):
            ax = axes[row_index][col_index]
            ax.set_facecolor("white")
            ax.axis("off")

            if image_path is None:
                blank = np.full((image_size[1], image_size[0], 3), 255, dtype=np.uint8)
                ax.imshow(blank)
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="gray",
                    transform=ax.transAxes,
                )
            else:
                ax.imshow(prepare_image(image_path, image_size=image_size))

            if row_index == 0:
                ax.set_title(col_labels[col_index], fontsize=12, pad=10)
            if col_index == 0:
                ax.set_ylabel(row_labels[row_index], rotation=0, fontsize=11, labelpad=38, va="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved figure: {output_path}")


def make_nvs_comparison_figure(experiment_root: Path, output_dir: Path) -> None:
    base_dir = experiment_root / "test" / "ours_50000"
    columns = [
        ("GT", base_dir / "gt"),
        ("Render", base_dir / "renders"),
        ("Normal", base_dir / "normal"),
        ("Base Color", base_dir / "base_color"),
        ("Roughness", base_dir / "roughness"),
        ("Visibility", base_dir / "visibility"),
    ]
    common_filenames = find_common_filenames([path for _, path in columns], required=True)
    selected_filenames = sample_evenly(common_filenames, count=4)
    if len(selected_filenames) < 4:
        log_warning(f"NVS figure uses {len(selected_filenames)} views because fewer than 4 common images were found.")

    image_rows = []
    row_labels = []
    for filename in selected_filenames:
        image_rows.append([directory / filename for _, directory in columns])
        row_labels.append(Path(filename).stem)

    save_grid_figure(
        image_rows=image_rows,
        row_labels=row_labels,
        col_labels=[label for label, _ in columns],
        output_path=output_dir / "nvs_comparison_grid.png",
        title="Novel View Synthesis Comparison",
    )


def choose_reference_filename(experiment_root: Path) -> str:
    directories = []
    for environment in ("fireplace", "forest", "night"):
        env_root = experiment_root / "test_rli" / environment
        directories.extend([env_root / "gt", env_root / "pbr"])
    common_filenames = find_common_filenames(directories, required=True)
    return common_filenames[len(common_filenames) // 2]


def make_relighting_env_comparison_figure(experiment_root: Path, output_dir: Path, reference_filename: str) -> None:
    environments = ("fireplace", "forest", "night")
    image_rows = []
    for environment in environments:
        env_root = experiment_root / "test_rli" / environment
        gt_path = env_root / "gt" / reference_filename
        pbr_path = env_root / "pbr" / reference_filename
        if not gt_path.exists():
            raise FileNotFoundError(f"Missing GT image for {environment}: {gt_path}")
        if not pbr_path.exists():
            raise FileNotFoundError(f"Missing PBR image for {environment}: {pbr_path}")
        image_rows.append([gt_path, pbr_path])

    save_grid_figure(
        image_rows=image_rows,
        row_labels=list(environments),
        col_labels=["GT", "PBR Relighting"],
        output_path=output_dir / "relighting_env_comparison.png",
        title=f"Relighting Comparison ({Path(reference_filename).stem})",
    )


def resolve_optional_image(optional_dir: Path, reference_filename: str, fallback_index: int) -> Path | None:
    if not optional_dir.exists():
        log_warning(f"Optional directory missing, skipped: {optional_dir}")
        return None
    images = list_images(optional_dir, required=False)
    if not images:
        return None

    direct_match = optional_dir / reference_filename
    if direct_match.exists():
        return direct_match

    if fallback_index < len(images):
        log_warning(
            f"Filename {reference_filename} not found in {optional_dir}. "
            f"Using same sorted index position {fallback_index} instead."
        )
        return images[fallback_index]

    log_warning(
        f"Filename {reference_filename} not found in {optional_dir}, and fallback index {fallback_index} is out of range."
    )
    return None


def make_relighting_decomposition_figure(experiment_root: Path, output_dir: Path, reference_filename: str) -> None:
    environments = ("fireplace", "forest", "night")
    column_specs = [
        ("PBR", "pbr", True),
        ("Base Color", "base_color", True),
        ("Visibility", "visibility", True),
        ("Direct", "direct", False),
        ("Local Lights", "local_lights", False),
    ]
    fallback_index = int(Path(reference_filename).stem)
    image_rows: list[list[Path | None]] = []

    for environment in environments:
        env_root = experiment_root / "test_rli" / environment
        row: list[Path | None] = []
        for label, dirname, required in column_specs:
            image_dir = env_root / dirname
            if required:
                image_path = image_dir / reference_filename
                if not image_dir.exists():
                    raise FileNotFoundError(f"Required directory missing for decomposition: {image_dir}")
                if not image_path.exists():
                    available = list_images(image_dir, required=True)
                    if fallback_index < len(available):
                        log_warning(
                            f"Required file {image_path.name} missing in {image_dir}. "
                            f"Using sorted index position {fallback_index}."
                        )
                        image_path = available[fallback_index]
                    else:
                        raise FileNotFoundError(
                            f"Required file missing and fallback index out of range for directory: {image_dir}"
                        )
                row.append(image_path)
            else:
                row.append(resolve_optional_image(image_dir, reference_filename, fallback_index))
        image_rows.append(row)

    save_grid_figure(
        image_rows=image_rows,
        row_labels=list(environments),
        col_labels=[label for label, _, _ in column_specs],
        output_path=output_dir / "relighting_decomposition_grid.png",
        title=f"Relighting Decomposition ({Path(reference_filename).stem})",
    )


def save_metric_bar_chart(metric_name: str, values: dict[str, float], output_dir: Path) -> None:
    environments = list(values.keys())
    scores = list(values.values())
    colors = ["#4C78A8", "#72B7B2", "#F58518"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6), facecolor="white")
    ax.set_facecolor("white")
    bars = ax.bar(environments, scores, color=colors[: len(environments)], width=0.62)
    ax.set_title(metric_name.replace("_", " "), fontsize=15, fontweight="bold")
    ax.set_ylabel(metric_name.split("_", 1)[0], fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.set_axisbelow(True)

    min_score = min(scores)
    max_score = max(scores)
    margin = max((max_score - min_score) * 0.25, 0.02 if max_score <= 1.5 else 0.4)
    ax.set_ylim(max(0.0, min_score - margin), max_score + margin)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + margin * 0.15,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    output_path = output_dir / f"metric_{metric_name.lower()}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    log_info(f"Saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper/PPT-ready figures from SVG-IR experiment outputs.")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=DEFAULT_EXPERIMENT_ROOT,
        help="Root directory of the experiment output.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save generated figures. Defaults to <experiment-root>/paper_figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    experiment_root = args.experiment_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else experiment_root / DEFAULT_OUTPUT_DIRNAME

    if not experiment_root.exists():
        log_error(f"Experiment root does not exist: {experiment_root}")
        return 1

    log_info(f"Experiment root: {experiment_root}")
    log_info(f"Output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0

    try:
        make_nvs_comparison_figure(experiment_root, output_dir)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        log_error(f"Failed to generate NVS comparison figure: {exc}")

    reference_filename = None
    try:
        reference_filename = choose_reference_filename(experiment_root)
        log_info(f"Selected reference relighting view: {reference_filename}")
    except Exception as exc:  # noqa: BLE001
        failures += 1
        log_error(f"Failed to choose relighting reference image: {exc}")

    if reference_filename is not None:
        try:
            make_relighting_env_comparison_figure(experiment_root, output_dir, reference_filename)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log_error(f"Failed to generate relighting environment comparison figure: {exc}")

        try:
            make_relighting_decomposition_figure(experiment_root, output_dir, reference_filename)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            log_error(f"Failed to generate relighting decomposition figure: {exc}")

    try:
        for metric_name, values in METRICS.items():
            save_metric_bar_chart(metric_name, values, output_dir)
    except Exception as exc:  # noqa: BLE001
        failures += 1
        log_error(f"Failed to generate metric bar charts: {exc}")

    if failures:
        log_error(f"Completed with {failures} failure(s). Check the messages above.")
        return 1

    log_info("All paper figures generated successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
