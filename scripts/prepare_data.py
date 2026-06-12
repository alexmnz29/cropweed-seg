"""Prepare the peanut dataset: verify raw data, convert masks, write splits.

Assumes data/raw/{images,labels}/ is already populated (see README for the
one-time download). Produces:
  data/processed/labels/   single-channel index masks (0/1/2), PNG
  data/splits/{train,val,test}.txt   one stem per line, crop-stratified

All decisions implemented here are justified in
notebooks/01_data_exploration.ipynb.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from cropweed_seg.masks import rgb_mask_to_index

ROOT = Path(__file__).resolve().parent.parent
RAW_IMAGES = ROOT / "data" / "raw" / "images"
RAW_LABELS = ROOT / "data" / "raw" / "labels"
OUT_LABELS = ROOT / "data" / "processed" / "labels"
OUT_SPLITS = ROOT / "data" / "splits"

SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def verify_raw() -> list[str]:
    """Check raw data integrity and return the sorted list of stems."""
    if not RAW_IMAGES.is_dir() or not RAW_LABELS.is_dir():
        sys.exit(f"Raw data not found under {ROOT / 'data' / 'raw'}. See README.")

    image_stems = {f.stem for f in RAW_IMAGES.glob("*.jpg") if f.is_file()}
    label_stems = {f.stem for f in RAW_LABELS.glob("*.png") if f.is_file()}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems
    if missing_labels or missing_images:
        sys.exit(
            f"Pairing broken. Images without label: {sorted(missing_labels)}; "
            f"labels without image: {sorted(missing_images)}"
        )
    print(f"Verified {len(image_stems)} image/label pairs.")
    return sorted(image_stems)


def convert_masks(stems: list[str], force: bool) -> np.ndarray:
    """Convert RGB masks to index PNGs. Returns per-image crop presence."""
    if OUT_LABELS.exists() and any(OUT_LABELS.iterdir()):
        if not force:
            sys.exit(f"{OUT_LABELS} is not empty. Use --force to regenerate.")
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    has_crop = np.zeros(len(stems), dtype=bool)
    for i, stem in enumerate(stems):
        mask_rgb = np.asarray(Image.open(RAW_LABELS / f"{stem}.png"))
        mask_idx = rgb_mask_to_index(mask_rgb)
        has_crop[i] = bool((mask_idx == 1).any())
        Image.fromarray(mask_idx, mode="L").save(OUT_LABELS / f"{stem}.png")
    print(f"Converted {len(stems)} masks -> {OUT_LABELS}")
    print(f"Images with crop: {has_crop.sum()}, without: {(~has_crop).sum()}")
    return has_crop


def stratified_split(
    stems: list[str], has_crop: np.ndarray, seed: int
) -> dict[str, list[str]]:
    """Split stems into train/val/test, stratified by crop presence."""
    rng = np.random.default_rng(seed)
    splits: dict[str, list[str]] = {name: [] for name in SPLIT_FRACTIONS}

    stems_arr = np.array(stems)
    for stratum_mask in (has_crop, ~has_crop):
        stratum = stems_arr[stratum_mask]
        rng.shuffle(stratum)
        n = len(stratum)
        n_train = round(n * SPLIT_FRACTIONS["train"])
        n_val = round(n * SPLIT_FRACTIONS["val"])
        splits["train"].extend(stratum[:n_train])
        splits["val"].extend(stratum[n_train : n_train + n_val])
        splits["test"].extend(stratum[n_train + n_val :])

    return {name: sorted(members) for name, members in splits.items()}


def write_splits(splits: dict[str, list[str]]) -> None:
    OUT_SPLITS.mkdir(parents=True, exist_ok=True)
    for name, members in splits.items():
        path = OUT_SPLITS / f"{name}.txt"
        path.write_text("\n".join(members) + "\n")
        print(f"{name}: {len(members)} stems -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--force", action="store_true", help="regenerate processed labels"
    )
    args = parser.parse_args()

    stems = verify_raw()
    has_crop = convert_masks(stems, force=args.force)
    splits = stratified_split(stems, has_crop, seed=args.seed)
    write_splits(splits)


if __name__ == "__main__":
    main()
