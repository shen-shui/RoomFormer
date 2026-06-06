import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from shapely.geometry import Polygon
from torch.utils.data import DataLoader

from datasets import build_dataset
from models import build_model
from util.plot_utils import plot_floorplan_with_regions, plot_room_map


def get_args_parser():
    parser = argparse.ArgumentParser("RoomFormer density-only inference", add_help=False)
    parser.add_argument("--batch_size", default=1, type=int)

    # Backbone.
    parser.add_argument("--backbone", default="resnet50", type=str)
    parser.add_argument("--lr_backbone", default=0, type=float)
    parser.add_argument("--dilation", action="store_true")
    parser.add_argument("--position_embedding", default="sine", type=str, choices=("sine", "learned"))
    parser.add_argument("--position_embedding_scale", default=2 * np.pi, type=float)
    parser.add_argument("--num_feature_levels", default=4, type=int)

    # Transformer.
    parser.add_argument("--enc_layers", default=6, type=int)
    parser.add_argument("--dec_layers", default=6, type=int)
    parser.add_argument("--dim_feedforward", default=1024, type=int)
    parser.add_argument("--hidden_dim", default=256, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--nheads", default=8, type=int)
    parser.add_argument("--num_queries", default=800, type=int)
    parser.add_argument("--num_polys", default=20, type=int)
    parser.add_argument("--dec_n_points", default=4, type=int)
    parser.add_argument("--enc_n_points", default=4, type=int)
    parser.add_argument("--query_pos_type", default="sine", type=str, choices=("static", "sine", "none"))
    parser.add_argument("--with_poly_refine", default=True, action="store_true")
    parser.add_argument("--masked_attn", default=False, action="store_true")
    parser.add_argument("--semantic_classes", default=-1, type=int)

    # Aux flag kept for checkpoint/model compatibility.
    parser.add_argument("--no_aux_loss", dest="aux_loss", action="store_true")

    # Dataset parameters.
    parser.add_argument("--dataset_name", default="floornet")
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--eval_set", default="test", type=str)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--room_score_threshold", default=0.5, type=float)
    parser.add_argument("--min_room_area", default=100.0, type=float)
    return parser


def trivial_batch_collator(batch):
    return batch


def polygons_from_outputs(outputs, room_score_threshold, min_room_area):
    pred_logits = outputs["pred_logits"]
    pred_corners = outputs["pred_coords"]
    fg_mask = torch.sigmoid(pred_logits) > room_score_threshold
    batch_polys = []

    for i in range(pred_logits.shape[0]):
        room_polys = []
        for j in range(fg_mask[i].shape[0]):
            valid_corners = pred_corners[i][j][fg_mask[i][j]]
            if len(valid_corners) == 0:
                continue

            corners = np.around((valid_corners * 255).cpu().numpy()).astype(np.int32)
            if len(corners) >= 4 and Polygon(corners).area >= min_room_area:
                room_polys.append(corners)
        batch_polys.append(room_polys)

    return batch_polys


def save_prediction_maps(output_dir, scene_id, sample, room_polys):
    floorplan_map = plot_floorplan_with_regions([np.array(r) for r in room_polys], scale=1000)
    cv2.imwrite(str(output_dir / f"{scene_id}_pred_floorplan.png"), floorplan_map)

    density_map = np.transpose((sample * 255).cpu().numpy(), [1, 2, 0])
    density_map = np.repeat(density_map, 3, axis=2)

    pred_room_map = np.zeros([256, 256, 3])
    for room_poly in room_polys:
        pred_room_map = plot_room_map(room_poly, pred_room_map)

    pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
    cv2.imwrite(str(output_dir / f"{scene_id}_pred_room_map.png"), pred_room_map)


@torch.no_grad()
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(args, train=False)
    model.to(device)
    model.eval()

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    unexpected_keys = [
        k for k in unexpected_keys
        if not (k.endswith("total_params") or k.endswith("total_ops"))
    ]
    if missing_keys:
        print("Missing Keys:", missing_keys)
    if unexpected_keys:
        print("Unexpected Keys:", unexpected_keys)
    print("loaded checkpoint:", args.checkpoint)

    dataset_eval = build_dataset(image_set=args.eval_set, args=args)
    data_loader_eval = DataLoader(
        dataset_eval,
        args.batch_size,
        sampler=torch.utils.data.SequentialSampler(dataset_eval),
        drop_last=False,
        collate_fn=trivial_batch_collator,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print("images:", len(dataset_eval))

    results = []
    for batched_inputs in data_loader_eval:
        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        file_names = [x["file_name"] for x in batched_inputs]

        outputs = model(samples)
        batch_polys = polygons_from_outputs(
            outputs,
            room_score_threshold=args.room_score_threshold,
            min_room_area=args.min_room_area,
        )

        for sample, scene_id, file_name, room_polys in zip(samples, scene_ids, file_names, batch_polys):
            save_prediction_maps(output_dir, scene_id, sample, room_polys)
            results.append({
                "image_id": int(scene_id),
                "file_name": file_name,
                "num_rooms": len(room_polys),
                "polygons": [poly.tolist() for poly in room_polys],
            })
            print(f"scene {scene_id}: rooms={len(room_polys)} file={file_name}")

    predictions_path = output_dir / "predictions.json"
    predictions_path.write_text(json.dumps(results, indent=2))
    print("saved predictions:", predictions_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "RoomFormer density-only inference script",
        parents=[get_args_parser()],
    )
    main(parser.parse_args())
