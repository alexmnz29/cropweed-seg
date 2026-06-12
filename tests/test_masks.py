"""Tests for cropweed_seg.masks.

The round-trip property was verified on all 400 real masks in
notebooks/01_data_exploration.ipynb; here it is pinned with synthetic
fixtures so it runs without the dataset on disk.
"""

import numpy as np

from cropweed_seg.masks import PALETTE, index_mask_to_rgb, rgb_mask_to_index


def test_round_trip_is_lossless_on_clean_colors():
    """RGB -> index -> RGB must reproduce a mask that uses the exact palette."""
    mask_idx = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.uint8,
    )
    mask_rgb = index_mask_to_rgb(mask_idx)

    recovered_idx = rgb_mask_to_index(mask_rgb)
    reconstructed_rgb = index_mask_to_rgb(recovered_idx)

    np.testing.assert_array_equal(recovered_idx, mask_idx)
    np.testing.assert_array_equal(reconstructed_rgb, mask_rgb)


def test_dominance_handles_intermediate_values():
    """Noisy pixels must resolve by channel dominance, as the docstring promises."""
    pixels = np.array(
        [
            [
                [10, 200, 0],  # G dominates -> crop
                [180, 30, 0],  # R dominates -> weed
                [0, 0, 0],  # neither -> background
                [7, 7, 250],  # R == G, B irrelevant -> background
            ]
        ],
        dtype=np.uint8,
    )
    expected = np.array([[1, 2, 0, 0]], dtype=np.uint8)
    np.testing.assert_array_equal(rgb_mask_to_index(pixels), expected)


def test_palette_matches_class_convention():
    """Index order is the contract: 0 background, 1 crop (green), 2 weed (red)."""
    np.testing.assert_array_equal(PALETTE[0], [0, 0, 0])
    np.testing.assert_array_equal(PALETTE[1], [0, 255, 0])
    np.testing.assert_array_equal(PALETTE[2], [255, 0, 0])
