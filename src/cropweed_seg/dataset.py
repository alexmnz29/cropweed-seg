"""PyTorch dataset for the peanut crop-weed segmentation task.

Reads images from data/raw/images/ and converted index masks from
data/processed/labels/, filtered by the split files in data/splits/.
Run scripts/prepare_data.py first to produce the masks and splits.
"""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F


class PeanutDataset(Dataset):
    """Crop-weed segmentation samples for one split.

    Returns (image, mask) where image is a normalized float tensor (3, H, W)
    and mask is an integer tensor (H, W) with values in {0, 1, 2}:
    0 background, 1 crop, 2 weed.
    """

    def __init__(self, root: Path, split: str, transform: v2.Compose) -> None:
        self.images_dir = root / "data" / "raw" / "images"
        self.labels_dir = root / "data" / "processed" / "labels"
        self.transform = transform

        split_file = root / "data" / "splits" / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(
                f"{split_file} not found. Run scripts/prepare_data.py first."
            )
        self.stems = split_file.read_text().split()

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        stem = self.stems[idx]

        image = F.to_image(Image.open(self.images_dir / f"{stem}.jpg"))
        mask = tv_tensors.Mask(
            F.pil_to_tensor(Image.open(self.labels_dir / f"{stem}.png")).squeeze(0)
        )

        image, mask = self.transform(image, mask)
        return image, mask
