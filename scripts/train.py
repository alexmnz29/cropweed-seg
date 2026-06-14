"""Train the baseline U-Net on the peanut dataset.

U-Net with a ResNet34 encoder pretrained on ImageNet, cross-entropy weighted
by inverse class frequency (computed from the train split), per-class IoU on
validation each epoch. Best checkpoint is kept by validation mIoU; weed IoU is
reported as the bottleneck class to watch but is too noisy per-epoch to select
on. Run with: uv run scripts/train.py
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from cropweed_seg.dataset import PeanutDataset
from cropweed_seg.engine import (
    compute_class_weights,
    evaluate,
    set_seed,
    train_one_epoch,
    write_metrics_csv,
)
from cropweed_seg.metrics import ConfusionMatrixMetric
from cropweed_seg.model import build_model
from cropweed_seg.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent

RUN_NAME = "baseline"
SEED = 42
BATCH_SIZE = 8
N_EPOCHS = 15
LEARNING_RATE = 1e-4


def main() -> None:
    set_seed(SEED)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    run_dir = ROOT / "runs" / RUN_NAME
    generator = torch.Generator()
    generator.manual_seed(SEED)

    train_loader = DataLoader(
        PeanutDataset(ROOT, "train", build_transforms("train")),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        PeanutDataset(ROOT, "val", build_transforms("val")),
        batch_size=4,
        shuffle=False,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=compute_class_weights(ROOT, device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    metric = ConfusionMatrixMetric()

    best_miou = -1.0
    history = []

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, iou = evaluate(model, val_loader, criterion, metric, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 4),
                "val_loss": round(val_loss, 4),
                "iou_background": round(iou["iou_background"], 4),
                "iou_crop": round(iou["iou_crop"], 4),
                "iou_weed": round(iou["iou_weed"], 4),
                "miou": round(iou["miou"], 4),
            }
        )

        marker = ""
        if iou["miou"] > best_miou:
            best_miou = iou["miou"]
            run_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), run_dir / "model.pt")
            marker = "  <- best mIoU, saved"

        print(
            f"epoch {epoch:2d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"bg {iou['iou_background']:.3f} crop {iou['iou_crop']:.3f} "
            f"weed {iou['iou_weed']:.3f} | mIoU {iou['miou']:.3f}{marker}"
        )

    write_metrics_csv(run_dir / "metrics.csv", history)
    print(f"\nbest mIoU: {best_miou:.4f}")
    print(f"checkpoint and metrics in {run_dir}")


if __name__ == "__main__":
    main()
