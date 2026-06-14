"""Tests for cropweed_seg.dataset.

Marked requires_data: these load the prepared dataset from disk and skip
cleanly where it is absent (fresh clone, CI without data). Run
scripts/prepare_data.py first to exercise them.
"""

from pathlib import Path

import pytest
import torch

from cropweed_seg.dataset import PeanutDataset
from cropweed_seg.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent

requires_data = pytest.mark.skipif(
    not (ROOT / "data" / "processed" / "labels").exists(),
    reason="requires prepared dataset on disk (run scripts/prepare_data.py)",
)


@requires_data
def test_train_sample_is_cropped_and_normalized():
    ds = PeanutDataset(ROOT, "train", build_transforms("train"))
    image, mask = ds[0]

    assert image.shape == (3, 512, 512)
    assert image.dtype == torch.float32
    assert mask.shape == (512, 512)
    assert set(torch.unique(mask).tolist()) <= {0, 1, 2}


@requires_data
def test_val_sample_is_full_frame():
    ds = PeanutDataset(ROOT, "val", build_transforms("val"))
    image, mask = ds[0]

    assert image.shape == (3, 720, 960)
    assert mask.shape == (720, 960)
    assert set(torch.unique(mask).tolist()) <= {0, 1, 2}


@requires_data
def test_lengths_match_splits():
    train = PeanutDataset(ROOT, "train", build_transforms("train"))
    val = PeanutDataset(ROOT, "val", build_transforms("val"))
    test = PeanutDataset(ROOT, "test", build_transforms("test"))

    assert len(train) == 280
    assert len(val) == 60
    assert len(test) == 60
    assert len(train) + len(val) + len(test) == 400


def test_missing_split_file_raises(tmp_path):
    """No data needed: an empty root must produce a clear error."""
    with pytest.raises(FileNotFoundError, match="prepare_data.py"):
        PeanutDataset(tmp_path, "train", build_transforms("train"))
