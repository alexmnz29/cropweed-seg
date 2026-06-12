# cropweed-seg

Semantic segmentation of crop vs weed on the peanut dataset
(Tran & Phan, IEEE Access 2023), built end to end: data exploration,
reproducible preprocessing, training, per-class evaluation, and
edge deployment on Jetson as the final target.

## Status

Data pipeline complete. Next: PyTorch dataset and baseline model.

- [x] Exploration notebook with evidence for every preprocessing decision
- [x] Lossless RGB-to-index mask conversion, unit tested
- [x] Crop-stratified 70/15/15 split, seeded and versioned
- [ ] Dataset + dataloader (512×512 random crops, full-res eval)
- [ ] Baseline model + training loop
- [ ] Per-class IoU evaluation
- [ ] Cross-dataset generalization test (Bonn sugar beets)
- [ ] Jetson Orin Nano deployment (ONNX → TensorRT → INT8)

## Setup

Requires [uv](https://docs.astral.sh/uv/).

    git clone git@github.com:alexmnz29/cropweed-seg.git
    cd cropweed-seg
    uv sync

## Data

Download the [Peanut dataset](https://github.com/ptdkhoa/Peanut-dataset)
(CC BY-SA 4.0) and copy `images/` and `labels/` into `data/raw/`. Then:

    uv run scripts/prepare_data.py

This verifies integrity, converts RGB masks to single-channel index masks
(0 background, 1 crop, 2 weed), and writes the stratified splits. Evidence
for each decision is in `notebooks/01_data_exploration.ipynb`.

## Dataset notes

400 RGB images, 960×720. Class balance: 84.0% background, 12.3% crop,
3.7% weed. Weed appears in every image; crop is absent from 29.5% of them,
which is why the split stratifies by crop presence. The imbalance is why
evaluation reports per-class IoU, never aggregate IoU alone.

## Tests

    uv run pytest
