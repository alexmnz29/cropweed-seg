# 0006 · Input resolution experiment

Status: accepted
Date: 2026-06-16

## Context

The DeepLabV3+ experiment (ADR 0004) refuted the architecture hypothesis and
moved the diagnosis: if a decoder built to preserve fine detail cannot recover
the thin weed, the detail is probably not in the input to begin with. A weed
stem 2 to 3 pixels wide inside a 512x512 crop carries very little signal, and
no decoder recovers information that is not there. The open lever named in
ADR 0004 was input resolution: give the model more pixels per weed and check
whether the thin-structure false negatives recede.

## Decision

Keep the champion at crop 512. Training at larger crops did not improve
weed IoU.

## Result

Same configuration as the champion (U-Net + ResNet34, focal+dice, seed 42,
batch 8, Adam 1e-4, 25 epochs, selection by best validation mIoU), changing
only the training crop from 512 to 720. Validation:

| crop size | mIoU | weed IoU |
| --- | --- | --- |
| 512 (champion) | 0.849 | 0.670 |
| 720 | 0.842 | 0.659 |

Evaluation runs on full 960x720 frames in both cases, so the comparison
isolates the training crop. The larger crop also raises training memory per
sample and reduces the diversity of random crops per image (a 720 crop covers
most of a 960x720 frame, so successive crops overlap heavily and randomness
adds less variety per epoch).

## Interpretation

The resolution hypothesis is refuted. More pixels per weed at train time did
not recover the thin structures; weed IoU landed at 0.659, within the same
band as every other attempt. Combined with ADR 0004, this closes both halves
of the fine-detail diagnosis: neither a detail-preserving decoder nor more
input resolution recovers what the error analysis shows being missed. The
remaining explanation is that the thin, low-contrast weed is genuinely
ambiguous in this data at any setting reachable here, which points at the
dataset (400 images, one region) rather than the model.

At this point three independent levers had converged on the same score: loss
(ADR 0003, best 0.670), architecture (ADR 0004, 0.655), and resolution
(0.659). Each refutation killed one rival explanation for the plateau. The
augmentation experiment (ADR 0007) later became the fourth.

## Consequences

- The champion configuration stays at crop 512; train.py defaults must match
  it (this was the source of a real slip: the committed train.py kept the
  crop 720 constants from this experiment for a while, so a fresh clone would
  have reproduced the experiment instead of the champion).
- The fine-structure failure mode stops being treated as a model problem.
  Remaining work targets the data side: augmentation (ADR 0007), the learning
  curve (ADR 0008), and eventually data from a second region.
- Edge-guided approaches (boundary loss, edge branch) stay as documented
  future work, with the expectation that they face the same data ceiling.
