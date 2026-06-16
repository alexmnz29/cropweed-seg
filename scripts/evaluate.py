"""Evaluate a trained checkpoint on a split, reporting per-class IoU.

Reads runs/<name>/config.json to rebuild the right architecture, so no need to
pass the architecture by hand. Reusable across the test split and future
cross-dataset evaluation. Run with:
    uv run scripts/evaluate.py --run focal_dice_s42 --split test
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from cropweed_seg.dataset import PeanutDataset
from cropweed_seg.engine import evaluate
from cropweed_seg.losses import build_criterion
from cropweed_seg.metrics import ConfusionMatrixMetric
from cropweed_seg.model import build_model
from cropweed_seg.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        required=True,
        help="run directory name under runs/ (contains model.pt)",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    run_dir = ROOT / "runs" / args.run
    checkpoint = run_dir / "model.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"{checkpoint} not found")

    config_path = run_dir / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        architecture = config["architecture"]
        encoder = config.get("encoder", "resnet34")
    else:
        architecture, encoder = "unet", "resnet34"  # runs from before config.json
        print("no config.json found, assuming unet / resnet34")

    model = build_model(architecture=architecture, encoder_name=encoder).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    print(f"loaded {checkpoint} (arch: {architecture}, encoder: {encoder})")

    loader = DataLoader(
        PeanutDataset(ROOT, args.split, build_transforms(args.split)),
        batch_size=4,
        shuffle=False,
    )

    # criterion only satisfies evaluate()'s signature; its loss is ignored
    criterion = build_criterion("focal_dice")
    metric = ConfusionMatrixMetric()
    _, iou = evaluate(model, loader, criterion, metric, device)

    print(f"\n{args.split} results for run '{args.run}':")
    for k, v in iou.items():
        print(f"  {k:<16} {v:.4f}")


if __name__ == "__main__":
    main()
