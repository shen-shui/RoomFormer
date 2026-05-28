#!/usr/bin/env python3
import argparse
import json
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser("Create a small COCO-style subset dataset.")
    parser.add_argument("--src_root", required=True, help="Source dataset root, e.g. data/stru3d")
    parser.add_argument("--dst_root", required=True, help="Output subset root, e.g. data/stru3d_subset")
    parser.add_argument("--train", type=int, default=300, help="Number of train images")
    parser.add_argument("--val", type=int, default=80, help="Number of val images")
    parser.add_argument("--test", type=int, default=0, help="Number of test images; 0 means keep none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="Copy images instead of creating symlinks")
    return parser.parse_args()


def subset_split(src_root, dst_root, split, count, seed, copy_files):
    ann_path = src_root / "annotations" / f"{split}.json"
    src_img_dir = src_root / split
    dst_img_dir = dst_root / split
    dst_ann_dir = dst_root / "annotations"
    dst_ann_path = dst_ann_dir / f"{split}.json"

    if not ann_path.exists():
        print(f"[skip] {split}: {ann_path} does not exist")
        return

    data = json.loads(ann_path.read_text())
    images = data.get("images", [])
    rng = random.Random(seed + hash(split) % 10000)
    selected = images[:] if count <= 0 else rng.sample(images, min(count, len(images)))
    selected_ids = {img["id"] for img in selected}
    selected_files = {img["file_name"] for img in selected}

    anns = [ann for ann in data.get("annotations", []) if ann.get("image_id") in selected_ids]
    subset = dict(data)
    subset["images"] = selected
    subset["annotations"] = anns

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_ann_dir.mkdir(parents=True, exist_ok=True)
    for img in selected:
        rel = Path(img["file_name"])
        src = src_img_dir / rel
        dst = dst_img_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            continue
        if copy_files:
            shutil.copy2(src, dst)
        else:
            os.symlink(os.path.relpath(src, dst.parent), dst)

    dst_ann_path.write_text(json.dumps(subset))
    print(
        f"[ok] {split}: {len(selected)} images, {len(anns)} annotations, "
        f"{len(selected_files)} files -> {dst_ann_path}"
    )


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    assert src_root.exists(), f"{src_root} does not exist"

    subset_split(src_root, dst_root, "train", args.train, args.seed, args.copy)
    subset_split(src_root, dst_root, "val", args.val, args.seed, args.copy)
    if args.test > 0:
        subset_split(src_root, dst_root, "test", args.test, args.seed, args.copy)


if __name__ == "__main__":
    main()
