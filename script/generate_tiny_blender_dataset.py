import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "dataset" / "tiny_blender" / "cube"
WIDTH = 64
HEIGHT = 64
FOV_X = math.radians(60.0)


def normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return v
    return v / n


def look_at(camera_pos, target=np.zeros(3, dtype=np.float32), up=np.array([0, 1, 0], dtype=np.float32)):
    camera_pos = np.asarray(camera_pos, dtype=np.float32)
    forward = normalize(target - camera_pos)
    right = normalize(np.cross(forward, up))
    true_up = normalize(np.cross(right, forward))

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = camera_pos
    return c2w


def cube_points():
    coords = np.linspace(-0.6, 0.6, 22, dtype=np.float32)
    points = []
    colors = []
    normals = []
    for axis in range(3):
        other = [i for i in range(3) if i != axis]
        for sign in (-1.0, 1.0):
            for u in coords:
                for v in coords:
                    p = np.zeros(3, dtype=np.float32)
                    p[axis] = sign * 0.6
                    p[other[0]] = u
                    p[other[1]] = v
                    points.append(p)
                    normals.append(normalize(np.sign(p) + 1e-4))
                    colors.append((p + 0.7) / 1.4)

    points = np.asarray(points, dtype=np.float32)
    colors = np.clip(np.asarray(colors, dtype=np.float32), 0.0, 1.0)
    normals = np.asarray(normals, dtype=np.float32)
    return points, colors, normals


def project(points, c2w):
    fx = 0.5 * WIDTH / math.tan(FOV_X * 0.5)
    fy = fx
    cx = WIDTH / 2.0
    cy = HEIGHT / 2.0

    w2c = np.linalg.inv(c2w)
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (w2c @ points_h.T).T[:, :3]

    visible = cam[:, 2] < -0.1
    cam = cam[visible]
    pts = points[visible]

    x = (-cam[:, 0] / cam[:, 2]) * fx + cx
    y = (cam[:, 1] / -cam[:, 2]) * fy + cy
    z = -cam[:, 2]
    return pts, x, y, z, visible


def render_frame(points, colors, c2w):
    image = np.zeros((HEIGHT, WIDTH, 4), dtype=np.float32)
    depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)

    visible_points, xs, ys, zs, visible = project(points, c2w)
    visible_colors = colors[visible]

    for point, color, px, py, pz in zip(visible_points, visible_colors, xs, ys, zs):
        ix = int(round(px))
        iy = int(round(py))
        if ix < 1 or ix >= WIDTH - 1 or iy < 1 or iy >= HEIGHT - 1:
            continue

        shade = 0.55 + 0.45 * np.clip(normalize(point + np.array([0.3, 0.4, 1.0], dtype=np.float32))[2], 0.0, 1.0)
        shaded = np.clip(color * shade, 0.0, 1.0)
        for oy in (-1, 0, 1):
            for ox in (-1, 0, 1):
                xx = ix + ox
                yy = iy + oy
                if pz < depth[yy, xx]:
                    depth[yy, xx] = pz
                    image[yy, xx, :3] = shaded
                    image[yy, xx, 3] = 1.0

    image[..., :3] = np.where(image[..., 3:4] > 0, image[..., :3], 1.0)
    return (image * 255.0).clip(0, 255).astype(np.uint8)


def write_ply(path, points, colors, normals):
    vertices = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"] = points[:, 0]
    vertices["y"] = points[:, 1]
    vertices["z"] = points[:, 2]
    vertices["nx"] = normals[:, 0]
    vertices["ny"] = normals[:, 1]
    vertices["nz"] = normals[:, 2]
    rgb = (colors * 255.0).clip(0, 255).astype(np.uint8)
    vertices["red"] = rgb[:, 0]
    vertices["green"] = rgb[:, 1]
    vertices["blue"] = rgb[:, 2]
    PlyData([PlyElement.describe(vertices, "vertex")]).write(path)


def frame_paths(split, count):
    return [f"{split}/{i:03d}" for i in range(count)]


def orbit_poses(count, radius, elevation_deg):
    poses = []
    elevation = math.radians(elevation_deg)
    for i in range(count):
        theta = 2.0 * math.pi * i / count
        pos = np.array(
            [
                radius * math.cos(theta) * math.cos(elevation),
                radius * math.sin(elevation),
                radius * math.sin(theta) * math.cos(elevation),
            ],
            dtype=np.float32,
        )
        poses.append(look_at(pos))
    return poses


def write_transforms(path, file_paths, poses):
    payload = {
        "camera_angle_x": FOV_X,
        "frames": [{"file_path": fp.replace("\\", "/"), "transform_matrix": pose.tolist()} for fp, pose in zip(file_paths, poses)],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    train_dir = DATASET_ROOT / "train"
    test_dir = DATASET_ROOT / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    points, colors, normals = cube_points()
    write_ply(DATASET_ROOT / "points3d.ply", points, colors, normals)

    train_files = frame_paths("train", 8)
    test_files = frame_paths("test", 2)
    train_poses = orbit_poses(8, radius=2.5, elevation_deg=18.0)
    test_poses = orbit_poses(2, radius=2.8, elevation_deg=10.0)

    for rel_path, pose in zip(train_files, train_poses):
        image = render_frame(points, colors, pose)
        Image.fromarray(image, mode="RGBA").save(DATASET_ROOT / f"{rel_path}.png")

    for rel_path, pose in zip(test_files, test_poses):
        image = render_frame(points, colors, pose)
        Image.fromarray(image, mode="RGBA").save(DATASET_ROOT / f"{rel_path}.png")

    write_transforms(DATASET_ROOT / "transforms_train.json", train_files, train_poses)
    write_transforms(DATASET_ROOT / "transforms_test.json", test_files, test_poses)
    print(f"Wrote tiny dataset to {DATASET_ROOT}")


if __name__ == "__main__":
    main()
