"""Transform pipelines for the peanut dataset.

train uses a random crop; val/test use the full 960x720 frame. Normalization
uses ImageNet statistics and applies to the image only. See
docs/decisions/0001-imagenet-normalization.md for why ImageNet stats and not
dataset-specific ones. Joint image/mask behavior validated in
notebooks/02_transforms_validation.ipynb.

augment selects the train-time augmentation stack:
  "none"  no augmentation; the champion (weed 0.670) trained with this.
  "flips" horizontal + vertical flips (p=0.5 each). Top-down field images
          have no privileged orientation, so both are label-preserving.
          Applied jointly to image and mask via torchvision v2, which handles
          tv_tensors.Mask without interpolating labels.
  "full"  flips plus mild ColorJitter (brightness/contrast/saturation 0.2,
          no hue). Targets the diagnosed photometric failure mode: thin,
          low-contrast weed against pale soil (notebook 05). Hue is left out
          on purpose: crop/weed discrimination is partly chromatic, and
          shifting the green tone would perturb that signal rather than
          simulate lighting diversity. ColorJitter applies to the image only;
          v2 leaves Mask untouched by color ops.
"""

import torch
from torchvision.transforms import v2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEFAULT_CROP_SIZE = 512

AUGMENT_MODES = ("none", "flips", "full")


def build_transforms(
    split: str, crop_size: int = DEFAULT_CROP_SIZE, augment: str = "none"
) -> v2.Compose:
    """Return the transform pipeline for a given split.

    train: random crop -> (flips) -> (color jitter) -> float scale -> normalize
    val/test: float scale -> ImageNet normalize (full frame, no crop)
    crop_size and augment only affect the train split.
    """
    if split not in {"train", "val", "test"}:
        raise ValueError(f"unknown split: {split!r}")
    if augment not in AUGMENT_MODES:
        raise ValueError(f"unknown augment mode: {augment!r}. Options: {AUGMENT_MODES}")

    steps = []
    if split == "train":
        steps.append(v2.RandomCrop(size=(crop_size, crop_size)))
        if augment in {"flips", "full"}:
            steps.append(v2.RandomHorizontalFlip(p=0.5))
            steps.append(v2.RandomVerticalFlip(p=0.5))
        if augment == "full":
            steps.append(v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2))
    steps.append(v2.ToDtype(torch.float32, scale=True))
    steps.append(v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return v2.Compose(steps)
