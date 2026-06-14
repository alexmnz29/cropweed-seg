"""Transform pipelines for the peanut dataset.

train uses a 512x512 random crop; val/test use the full 960x720 frame.
Normalization uses ImageNet statistics and applies to the image only.
See docs/decisions/0001-imagenet-normalization.md for why ImageNet stats
and not dataset-specific ones. Joint image/mask behavior validated in
notebooks/02_transforms_validation.ipynb.
"""

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

TRAIN_CROP_SIZE = (512, 512)


def build_transforms(split: str) -> v2.Compose:
    """Return the transform pipeline for a given split.

    train: random crop -> float scale -> ImageNet normalize
    val/test: float scale -> ImageNet normalize (full frame, no crop)
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split!r}")

    steps = []
    if split == "train":
        steps.append(v2.RandomCrop(size=TRAIN_CROP_SIZE))
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(steps)
