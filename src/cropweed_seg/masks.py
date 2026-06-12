"""Mask conversion between the dataset's RGB encoding and index masks.

Index convention: 0 = background, 1 = crop, 2 = weed.
Evidence for the conversion rule lives in notebooks/01_data_exploration.ipynb:
the dataset contains exactly (0,0,0), (0,255,0) and (255,0,0), and the
round-trip RGB -> index -> RGB is lossless on all 400 masks.
"""

import numpy as np

PALETTE = np.array(
    [[0, 0, 0], [0, 255, 0], [255, 0, 0]],
    dtype=np.uint8,
)

N_CLASSES = 3
CLASS_NAMES = ("background", "crop", "weed")


def rgb_mask_to_index(mask_rgb: np.ndarray) -> np.ndarray:
    """Convert an (H, W, 3) RGB mask to an (H, W) uint8 index mask.

    Rule: channel dominance. G > R -> 1 (crop), R > G -> 2 (weed),
    else 0 (background). Exact on this dataset, robust on noisier ones.
    """
    r = mask_rgb[..., 0].astype(np.int16)
    g = mask_rgb[..., 1].astype(np.int16)
    out = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    out[g > r] = 1
    out[r > g] = 2
    return out


def index_mask_to_rgb(mask_idx: np.ndarray) -> np.ndarray:
    """Convert an (H, W) index mask back to (H, W, 3) RGB using PALETTE."""
    return PALETTE[mask_idx]
