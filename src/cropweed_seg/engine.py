"""Training and evaluation loops for the segmentation model.

Pure logic, shared by scripts/train.py and the training notebook. Construction
of model, data, and optimizer lives in the caller; these functions take what
they need as arguments.
"""

import csv
import random
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torch.utils.data import DataLoader

from cropweed_seg.masks import N_CLASSES
from cropweed_seg.metrics import ConfusionMatrixMetric


def compute_class_weights(root: Path, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights from the train split, normalized to mean 1."""
    labels_dir = root / "data" / "processed" / "labels"
    train_stems = (root / "data" / "splits" / "train.txt").read_text().split()

    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for stem in train_stems:
        mask = np.asarray(Image.open(labels_dir / f"{stem}.png"))
        counts += np.bincount(mask.ravel(), minlength=N_CLASSES)

    inv_freq = counts.sum() / counts
    weights = inv_freq / inv_freq.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    metric: ConfusionMatrixMetric,
    device: torch.device,
) -> tuple[float, dict]:
    model.eval()
    metric.reset()
    running_loss = 0.0
    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device).long()

        logits = model(images)
        running_loss += criterion(logits, masks).item()
        metric.update(logits.argmax(dim=1), masks)
    return running_loss / len(loader), metric.compute()


def set_seed(seed: int = 42) -> None:
    """Seed Python, numpy and torch for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_metrics_csv(path: Path, rows: list[dict]) -> None:
    """Write per-epoch metrics to CSV. One row per epoch."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
