"""Generate fractional train splits for the learning-curve experiment.

Reads data/splits/train.txt and writes train_p25.txt, train_p50.txt and
train_p75.txt. Two properties matter for a clean curve:

Stratified: each fraction keeps the same proportion of crop-present and
crop-absent images as the full train split, same rule as prepare_data.py.
Otherwise a small fraction could over- or under-sample the no-crop stratum
and the curve would confound data volume with data composition.

Nested: the 25% subset is contained in the 50%, which is contained in the
75%, which is contained in the full split. Each point on the curve adds
images to the previous point instead of drawing a fresh sample, so the
comparison is "more of the same data", not "different data". One seeded
shuffle per stratum, then prefixes.

The val and test splits are untouched: every point on the curve is measured
against the identical validation set.

Run once: uv run scripts/make_learning_curve_splits.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SPLITS_DIR = ROOT / "data" / "splits"
LABELS_DIR = ROOT / "data" / "processed" / "labels"

FRACTIONS = (0.25, 0.50, 0.75)
SEED = 42


def main() -> None:
    stems = SPLITS_DIR.joinpath("train.txt").read_text().split()

    has_crop = np.array(
        [
            bool((np.asarray(Image.open(LABELS_DIR / f"{s}.png")) == 1).any())
            for s in stems
        ]
    )
    stems_arr = np.array(stems)
    print(
        f"train: {len(stems)} stems ({has_crop.sum()} with crop, {(~has_crop).sum()} without)"
    )

    rng = np.random.default_rng(SEED)
    shuffled = {}
    for name, mask in (("crop", has_crop), ("nocrop", ~has_crop)):
        stratum = stems_arr[mask].copy()
        rng.shuffle(stratum)
        shuffled[name] = stratum

    for frac in FRACTIONS:
        members: list[str] = []
        for stratum in shuffled.values():
            n = round(len(stratum) * frac)
            members.extend(stratum[:n])
        members.sort()
        out = SPLITS_DIR / f"train_p{int(frac * 100)}.txt"
        out.write_text("\n".join(members) + "\n")
        print(f"{out.name}: {len(members)} stems")


if __name__ == "__main__":
    main()
