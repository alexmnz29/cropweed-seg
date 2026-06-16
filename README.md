# cropweed-seg

Semantic segmentation of crop vs weed in peanut fields, built end to end: from
raw data to a deployable edge model. The project covers data exploration,
reproducible preprocessing, a measured loss-function study, honest per-class
evaluation, and edge deployment on Jetson.

**Headline result:** weed IoU 0.670, mIoU 0.850 on the held-out test set, with a
U-Net trained on a 3.7%-weed dataset. weed is the hard class, and the result is
competitive with what published work reports on the same dataset.

## Pipeline

```mermaid
flowchart TD
    A[Raw dataset<br>400 RGB 960x720] --> B[prepare_data.py<br>RGB to index masks<br>stratified split]
    B --> C[PeanutDataset<br>512 crops + norm]
    C --> D[Training<br>U-Net ResNet34<br>focal+dice]
    D --> E[Evaluation<br>per-class IoU]
    E --> F[Champion<br>test weed 0.670]
    F --> G[Edge deploy<br>ONNX-TensorRT-INT8<br>Jetson Orin Nano]
```

## The problem

Peanut yields drop sharply when weeds compete during the early growth phase, so
detecting weed against crop and soil is a real precision-agriculture task. The
hard part is imbalance: in this dataset weed is 3.7% of pixels, crop 12.3%, and
background 84%. A model can score well on average while missing weed almost
entirely, so the imbalance shapes every decision here, from the loss to the
evaluation metric.

## Data

The [peanut dataset](https://github.com/ptdkhoa/Peanut-dataset) (Tran & Phan,
IEEE Access 2023, CC BY-SA 4.0) is 400 RGB images at 960x720 from fields near Da
Nang, Vietnam. Notebook 01 validates it end to end. Key findings:

- Masks contain exactly three clean colors across all 276M pixels. No
  antialiasing, no ambiguous pixels. Conversion to index masks is lossless.
- Class balance: background 84.0%, crop 12.3%, weed 3.7%.
- weed appears in every image; crop is absent from 29.5% of them. The split
  therefore stratifies by crop presence, not weed.

Splits are seeded and versioned so the experiments are reproducible.

## Method

- **Model:** U-Net with a ResNet34 encoder pretrained on ImageNet, via
  segmentation-models-pytorch. A standard, well-understood baseline rather than a
  hand-rolled architecture.
- **Training:** 512x512 random crops at native resolution, full-frame evaluation,
  ImageNet normalization, Adam, seeded runs.
- **Loss:** focal + Dice, chosen by a measured study (below).
- **Evaluation:** per-class IoU from a confusion matrix accumulated over the
  whole split, never aggregate mIoU alone. With 29.5% of images lacking crop,
  per-image IoU would be undefined for absent classes; accumulating sidesteps
  that.

## Results

### Loss study

Four imbalance-handling strategies, all at the best mIoU epoch, same split and
seed handling. focal and focal+dice were run to convergence at 25 epochs.

| loss | mIoU | weed IoU |
| --- | --- | --- |
| weighted cross-entropy | 0.768 | 0.496 |
| focal | 0.840 | 0.650 |
| dice | 0.838 | 0.643 |
| **focal + dice** | **0.849** | **0.670** |

focal+dice wins, reproducible across two seeds (weed 0.670 both times). The
improvement over the weighted cross-entropy baseline is +0.174 weed IoU.

### Test set

The champion, evaluated once on the held-out test split:

| class | IoU |
| --- | --- |
| background | 0.971 |
| crop | 0.910 |
| weed | 0.670 |
| **mIoU** | **0.850** |

Test matches validation almost exactly (weed 0.670 vs 0.670), so the estimate is
reliable and the method holds: stratified split, selection on validation, test
untouched until the end.

### Error analysis

![Error gallery: worst cases by weed IoU](docs/img/error_gallery.png)

The worst cases share a pattern: the model misses thin, filamentous,
low-contrast weed (teal in the error map), while segmenting compact weed and
crop well. Errors are dominated by false negatives on fine structures, not class
confusion.

## The ceiling

```mermaid
flowchart TD
    A[Baseline<br>weighted CE<br>weed 0.496] --> B{Loss study}
    B --> C[focal<br>weed 0.650]
    B --> D[dice<br>weed 0.643]
    B --> E[focal+dice<br>weed 0.670]
    E --> F{Push further?}
    F --> G[DeepLabV3+<br>weed 0.655<br>no gain]
    F --> H[crop 720<br>weed 0.659<br>no gain]
    G --> I[Ceiling ~0.67<br>three levers converge<br>limit is data + ambiguity]
    H --> I
    E --> I
```

weed converges around 0.67 across three independent levers: loss (the study
above), architecture (DeepLabV3+, no improvement), and input resolution (720
crops, no improvement). Three levers converging is strong evidence the limit is
the task on this dataset, not a single model choice.

The likely cause is intrinsic ambiguity plus data size. Thin low-contrast stems
are genuinely hard to separate, and 400 images give few examples to learn from.
This reading matches the dataset authors and later work on the same data:

- PSPEdgeWeedNet (Pai et al., Sci Rep 2025) reports weed as the lowest-scoring
  class even with an edge-aware architecture and CRF post-processing, with weed
  around 0.60 to 0.69 depending on metric.
- The same work names the small, single-region dataset as a key limitation and
  notes these methods usually need thousands of images.
- The error mode they describe, missed small weed from lost spatial resolution,
  is the same one this project finds independently.

So weed 0.670 from a plain U-Net with focal+dice, no edge branch and no CRF, sits
in the competitive range for this dataset. The number is read as near the
practical ceiling, not as a weak result.

## Deployment

> Edge deployment on Jetson Orin Nano (ONNX → TensorRT → INT8): in progress.
> Will report latency benchmarks and per-class accuracy drop after quantization.

## Scope decisions and future work

- **Experiment tracking** used versioned CSVs, not MLflow, given the small number
  of runs. A larger sweep would justify a tracking framework.
- **No further model tuning.** Three levers showed the ceiling; more backbones or
  losses have diminishing returns.
- **Cross-dataset generalization (Bonn sugar beets)** is open as future work. It
  needs label-scheme and spectral remapping, and a weed-only framing would
  isolate generalization from class mismatch.
- **Edge-guided segmentation** (a boundary loss or edge branch) could target the
  fine-structure errors, but faces the same data ceiling.

## Reproducibility

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone git@github.com:alexmnz29/cropweed-seg.git
cd cropweed-seg
uv sync
```

Download the [peanut dataset](https://github.com/ptdkhoa/Peanut-dataset) into
`data/raw/{images,labels}/`, then:

```bash
uv run scripts/prepare_data.py          # verify, convert masks, write splits
uv run scripts/train.py                 # train (config in the script header)
uv run scripts/evaluate.py --run focal_dice_s42 --split test
uv run pytest                           # run the test suite
```

Each training run writes its config, checkpoint, and per-epoch metrics to
`runs/<name>/`. Decisions are documented in `docs/decisions/`.
