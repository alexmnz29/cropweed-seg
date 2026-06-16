# 0004 · DeepLabV3+ experiment

Status: accepted
Date: 2026-06-15

## Context

The test error analysis (notebook 05) showed the U-Net champion misses thin,
filamentous, low-contrast weed and segments compact regions well. The failure
looked like a resolution or architecture limit, dominated by false negatives on
fine structures, not class confusion. DeepLabV3+, with its decoder and atrous
convolutions designed to preserve fine detail, was the hypothesis to recover that
lost weed.

## Decision

Keep U-Net as the champion. DeepLabV3+ did not improve on it.

## Result

Same loss (focal+dice), epochs (25), seed (42), and crop (512) as the champion,
changing only the architecture. Validation:

| model | mIoU | weed IoU | plateau |
| --- | --- | --- | --- |
| U-Net | 0.849 | 0.670 | stable |
| DeepLabV3+ | 0.841 | 0.655 | noisy, stdev 0.048, dips to 0.51 |

DeepLabV3+ is slightly worse on peak and clearly less stable, with sharp weed
dips late in training (epochs 22 and 24).

## Interpretation

The hypothesis is refuted, which reorients the diagnosis. If a decoder built to
preserve fine detail does not recover the thin weed, the bottleneck is likely not
the architecture but the information in the input. A 2 to 3 pixel stem at 512x512
has very little signal left to preserve, and no decoder recovers detail that is
not there. The right lever is input resolution, giving the model more pixels per
weed, not more decoder capacity.

The late-training instability fits a second point: with 280 training images, the
heavier model (ASPP, more parameters) overfits more easily, which shows as erratic
validation late in the run.

## Consequences

U-Net focal+dice stays the champion at weed 0.670 (val).

The open lever is input resolution: train U-Net with larger crops (768) or full
960x720 frames and check whether thin weed improves. Edge-guided approaches
(an auxiliary boundary loss or an edge branch, as in PSPEdgeWeedNet) are a
heavier option, kept as future work and justified only if higher resolution does
not resolve the fine-structure errors.

A negative result, interpreted and acted on, is the value here: the error
analysis pointed at architecture, the experiment ruled it out, and the diagnosis
moved to resolution.
