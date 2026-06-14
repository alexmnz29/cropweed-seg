# 0002 · Baseline architecture and checkpoint selection

Status: accepted
Date: 2026-06-14

## Context

The baseline needs a segmentation model and a rule for which checkpoint to keep
across epochs. Two questions: what architecture, and selected on which metric.

## Decision

U-Net with a ResNet34 encoder pretrained on ImageNet, from
segmentation-models-pytorch (SMP). Instantiated through the library rather than
implemented by hand.

Checkpoint kept by best validation mIoU, not by weed IoU.

## Alternatives considered

Implementing U-Net from scratch. Rejected for the project baseline: nobody
reimplements standard architectures in production, and the engineering value
here is the data pipeline, honest per-class evaluation, imbalance handling, and
edge deployment, not rewriting a known model. A from-scratch U-Net is worth a
separate teaching notebook, not the baseline.

Heavier encoders (ResNet50) or lighter ones (ResNet18). ResNet34 measured at
24.4M params and around 6 GB training memory at batch 8, 512x512, which fits the
24 GB machine comfortably. The encoder is not the memory bottleneck; activations
are, and they scale with batch size, which is the knob to turn if memory gets
tight.

Selecting the checkpoint by weed IoU. Weed is the bottleneck class and the
target to improve, so it was the first instinct. Rejected because weed IoU is
noisy per epoch (it bounced between 0.39 and 0.53 across adjacent epochs on a
stable model). Selecting on a noisy metric rewards a lucky spike rather than the
best model. The weed peak at epoch 6 (0.526) was an isolated spike; the stable
plateau at epochs 11 to 13 is the real optimum. mIoU is steadier and its best
epoch (13) lands in that plateau, with a strong weed value (0.496) as a result.

## Consequences

The baseline is mIoU 0.768, with background 0.947, crop 0.862, weed 0.496,
converged by epoch 13, reproducible under seed 42. This is the number future
experiments are measured against.

weed at roughly half the IoU of the other classes is the bottleneck. Experiments
to address it (Dice or focal loss, sampling of non-empty crops) are measured
against this baseline with the same seed and split.

Validation loss starts to flatten and tick up around epochs 14 to 15 while train
loss keeps falling, so training longer would not help. The useful range is the
plateau around epochs 11 to 13.
