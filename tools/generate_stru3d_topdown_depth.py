#!/usr/bin/env python3
"""Generate RoomFormer-aligned top-down depth maps from raw Structured3D panoramas."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


INVALID_SCENE_IDS = {
    49, 80, 323, 325, 334, 344, 440, 491, 593, 659, 713, 779, 793, 861, 864,
    969, 986, 1058, 1156, 1207, 1280, 1354, 1356, 1378, 1445, 1473, 1523, 1544,
    1634, 1696, 1741, 1835, 1850, 1938, 2033, 2047, 2056, 2107, 2165, 2178, 2186,
    2263, 2290, 2332, 2357, 2367, 2374, 2421, 2454, 2471, 2479, 2486, 2519, 2544,
    2606, 2630, 2656, 2685, 2713, 2742, 2762, 2797, 2860, 2868, 2877, 2921, 2927,
    2940, 2947, 2951, 2961, 2964, 2976, 2980, 2985, 2996, 3066, 3127, 3235, 3256,
    3271, 3296, 3342, 3387, 3398, 3421, 3426, 3427, 3429, 3437, 3443, 3457, 3478,
    3480, 3481, 3489, 3505,
}


def split_for_scene(scene_key):
    sid = int(scene_key)
    if sid < 3000:
        return "train"
    if sid < 3250:
        return "val"
    return "test"


def load_processed_scenes(processed_root):
    scenes = {}
    annotations_dir = processed_root / "annotations"
    for split in ("train", "val", "test"):
        ann_path = annotations_dir / f"{split}.json"
        if not ann_path.exists():
            continue
        with ann_path.open("r") as f:
            data = json.load(f)
        for image in data.get("images", []):
            file_name = image["file_name"]
            stem = Path(file_name).stem
            try:
                scene_key = int(stem)
            except ValueError:
                scene_key = int(image["id"])
            scenes[scene_key] = {"split": split, "file_name": file_name}
    return scenes


def camera_center(path):
    with path.open("r") as f:
        return np.asarray([float(v) for v in f.readline().strip().split()], dtype=np.float32)


def angle_grids(shape):
    height, width = shape
    x_tick = 180.0 / height
    y_tick = 360.0 / width
    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    alpha = np.deg2rad(90.0 - rows * x_tick)
    beta = np.deg2rad(cols * y_tick - 180.0)
    return alpha, beta


def panorama_points(scene_path, resolution):
    coords = []
    cache = {}
    render_root = scene_path / "2D_rendering"
    if not render_root.exists():
        return None

    for section in sorted(render_root.iterdir()):
        if not section.is_dir():
            continue
        pano_root = section / "panorama"
        depth_path = pano_root / resolution / "depth.png"
        camera_path = pano_root / "camera_xyz.txt"
        if not depth_path.exists() or not camera_path.exists():
            continue

        depth = np.asarray(Image.open(depth_path), dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        valid = depth > 500.0
        if not np.any(valid):
            continue

        if depth.shape not in cache:
            cache[depth.shape] = angle_grids(depth.shape)
        alpha, beta = cache[depth.shape]
        center = camera_center(camera_path)

        xy_offset = depth * np.cos(alpha)
        points = np.stack(
            [
                xy_offset * np.sin(beta),
                xy_offset * np.cos(beta),
                depth * np.sin(alpha),
            ],
            axis=-1,
        )
        coords.append(points[valid] + center)

    if not coords:
        return None

    coords = np.concatenate(coords, axis=0)
    coords[:, :2] = np.round(coords[:, :2] / 10.0) * 10.0
    coords[:, 2] = np.round(coords[:, 2] / 100.0) * 100.0
    coords = np.unique(coords, axis=0)
    return coords


def topdown_depth_map(points, width, height, stat):
    ps = points * -1.0
    ps[:, 0] *= -1.0
    ps[:, 1] *= -1.0

    image_res = np.array((width, height), dtype=np.float32)
    max_coords = np.max(ps, axis=0)
    min_coords = np.min(ps, axis=0)
    span = max_coords - min_coords
    max_coords = max_coords + 0.1 * span
    min_coords = min_coords - 0.1 * span

    denom_xy = np.maximum(max_coords[:2] - min_coords[:2], 1e-6)
    coords = np.round((ps[:, :2] - min_coords[None, :2]) / denom_xy[None, :] * image_res[None, :])
    coords = np.minimum(np.maximum(coords, np.zeros_like(image_res)), image_res - 1).astype(np.int32)

    z_span = max(max_coords[2] - min_coords[2], 1e-6)
    values = np.clip((ps[:, 2] - min_coords[2]) / z_span, 0.0, 1.0).astype(np.float32)
    flat = coords[:, 1] * width + coords[:, 0]

    if stat == "max":
        out = np.zeros(height * width, dtype=np.float32)
        np.maximum.at(out, flat, values)
    elif stat == "min":
        out = np.ones(height * width, dtype=np.float32)
        np.minimum.at(out, flat, values)
        out[out == 1.0] = 0.0
    elif stat == "range":
        hi = np.zeros(height * width, dtype=np.float32)
        lo = np.ones(height * width, dtype=np.float32)
        np.maximum.at(hi, flat, values)
        np.minimum.at(lo, flat, values)
        out = hi - np.where(lo == 1.0, hi, lo)
    else:
        sums = np.zeros(height * width, dtype=np.float32)
        counts = np.zeros(height * width, dtype=np.float32)
        np.add.at(sums, flat, values)
        np.add.at(counts, flat, 1.0)
        out = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)

    return (out.reshape(height, width) * 255.0).clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, type=Path,
                        help="Directory containing scene_xxxxx folders from raw Structured3D.")
    parser.add_argument("--processed_root", default="data/stru3d", type=Path,
                        help="Processed RoomFormer Structured3D root with annotations/*.json.")
    parser.add_argument("--output_root", default="data/stru3d_depth", type=Path)
    parser.add_argument("--resolution", default="full", choices=["full", "empty", "simple"])
    parser.add_argument("--stat", default="mean", choices=["mean", "max", "min", "range"])
    parser.add_argument("--width", default=256, type=int)
    parser.add_argument("--height", default=256, type=int)
    parser.add_argument("--limit", default=0, type=int)
    args = parser.parse_args()

    processed_scenes = load_processed_scenes(args.processed_root)
    for split in ("train", "val", "test"):
        (args.output_root / split).mkdir(parents=True, exist_ok=True)

    scenes = sorted(args.raw_root.glob("scene_*"))
    written = skipped = 0
    for scene_path in scenes:
        scene_id = scene_path.name.split("_")[-1]
        scene_key = int(scene_id)
        if processed_scenes and scene_key not in processed_scenes:
            skipped += 1
            continue
        if scene_key in INVALID_SCENE_IDS:
            skipped += 1
            continue

        processed = processed_scenes.get(
            scene_key,
            {"split": split_for_scene(scene_key), "file_name": f"{scene_id}.png"},
        )
        split = processed["split"]
        out_path = args.output_root / split / processed["file_name"]
        if out_path.exists():
            skipped += 1
            continue

        points = panorama_points(scene_path, args.resolution)
        if points is None:
            skipped += 1
            continue

        depth = topdown_depth_map(points, args.width, args.height, args.stat)
        Image.fromarray(depth).save(out_path)
        written += 1
        print(f"[{written}] wrote {out_path} from {scene_path.name}")

        if args.limit and written >= args.limit:
            break

    print(f"done: wrote={written}, skipped={skipped}, output_root={args.output_root}")


if __name__ == "__main__":
    main()
