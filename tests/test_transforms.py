"""Tests for cropweed_seg.transforms.

No data required: these check the pipeline composition and contract,
not behavior on real images.
"""

import pytest
import torch
from torchvision.transforms import v2

from cropweed_seg.transforms import build_transforms


def test_train_includes_random_crop():
    transform = build_transforms("train")
    transform_types = [type(t) for t in transform.transforms]
    assert v2.RandomCrop in transform_types


def test_val_and_test_have_no_crop():
    for split in ("val", "test"):
        transform = build_transforms(split)
        transform_types = [type(t) for t in transform.transforms]
        assert v2.RandomCrop not in transform_types


def test_all_splits_normalize():
    for split in ("train", "val", "test"):
        transform = build_transforms(split)
        transform_types = [type(t) for t in transform.transforms]
        assert v2.Normalize in transform_types


def test_augment_adds_flips_to_train_only():
    transform = build_transforms("train", augment=True)
    transform_types = [type(t) for t in transform.transforms]
    assert v2.RandomHorizontalFlip in transform_types
    assert v2.RandomVerticalFlip in transform_types

    for split in ("val", "test"):
        transform = build_transforms(split, augment=True)
        transform_types = [type(t) for t in transform.transforms]
        assert v2.RandomHorizontalFlip not in transform_types
        assert v2.RandomVerticalFlip not in transform_types


def test_no_flips_by_default():
    transform = build_transforms("train")
    transform_types = [type(t) for t in transform.transforms]
    assert v2.RandomHorizontalFlip not in transform_types
    assert v2.RandomVerticalFlip not in transform_types


def test_unknown_split_raises():
    with pytest.raises(ValueError, match="unknown split"):
        build_transforms("trian")  # typo on purpose


def test_normalize_leaves_mask_untouched():
    """A Normalize step must not alter integer mask values."""
    from torchvision import tv_tensors

    image = tv_tensors.Image(torch.randint(0, 256, (3, 64, 64), dtype=torch.uint8))
    mask = tv_tensors.Mask(torch.randint(0, 3, (64, 64), dtype=torch.uint8))

    transform = build_transforms("val")  # no crop, so shapes are preserved
    _, mask_out = transform(image, mask)

    assert mask_out.dtype == torch.uint8
    assert torch.equal(torch.unique(mask_out), torch.unique(mask))
