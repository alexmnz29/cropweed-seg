# 0007 · Augmentation experiment

Status: accepted
Date: 2026-07-22

## Context

The ceiling analysis rested on three refuted levers: loss (ADR 0003),
architecture (ADR 0004), and input resolution. Train-time augmentation was a
fourth lever, untested: the champion trained with a random crop only, no flips
and no photometric transforms. The error analysis (notebook 05) diagnoses the
failure mode as thin, low-contrast weed against pale soil, which is a
photometric pattern. Mild color jitter simulates exactly the lighting
diversity a single-region dataset lacks, so this was the lever with the most
plausible mechanism left.

## Decision

Keep the champion unchanged. Augmentation, geometric and photometric,
converged to the same weed IoU as the champion and does not replace it.

## Experiments

Two augmentation stacks on the champion configuration (U-Net + ResNet34,
focal+dice, crop 512, seed 42, batch 8, Adam 1e-4), selection by best
validation mIoU as always:

| run | augmentation | epochs | best epoch | mIoU | weed IoU |
| --- | --- | --- | --- | --- | --- |
| champion | none | 25 | 13 | 0.849 | 0.670 |
| flips | hflip + vflip, p=0.5 | 25 | 20 | 0.846 | 0.661 |
| full | flips + color jitter | 35 | 26 | 0.849 | 0.671 |

The full stack adds ColorJitter with brightness, contrast and saturation at
0.2 and no hue shift. Hue is excluded on purpose: crop/weed discrimination is
partly chromatic, and shifting the green tone would perturb that signal
rather than simulate lighting variation. Flips are label-preserving on
top-down field images (no privileged orientation) and are applied jointly to
image and mask via torchvision v2.

The full run uses 35 epochs because augmentation increases effective train
diversity and delays convergence. At 25 epochs the flips run was not clearly
plateaued (val loss still descending at the final epoch), the same trap
documented in ADR 0003 for the loss comparison. At 35 epochs the full run
shows ten final epochs of stable validation, so the comparison is made with
both sides converged.

## Interpretation

The full stack lands on weed 0.671 against the champion's 0.670, agreement to
the third decimal, the same number the loss study, DeepLabV3+ and the
resolution run converged on. The photometric jitter targeted the diagnosed
failure mode directly and still did not move the score. This is the strongest
of the four negative results, because it refutes the most plausible remaining
mechanism: the missing lighting diversity cannot be simulated into the
training set. The wall is not simulable variation. It is real data.

## Consequences

- The champion stands. Because the augmented candidate never beat the
  champion on validation, it never touched the test split; the one-shot test
  protocol survives the experiment intact, and the deployed ONNX, INT8 and
  TensorRT artifacts remain valid unchanged.
- The ceiling argument moves from three refuted levers to four.
- The augment flag stays in transforms.py (modes none, flips, full),
  documented and off by default, so the experiment is reproducible from the
  committed code.
