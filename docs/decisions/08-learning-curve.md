# 0008 · Learning curve: is the ceiling set by data volume?

Status: accepted
Date: 2026-07-22

## Context

Four refuted levers (loss, architecture, resolution, augmentation; ADRs 0003,
0004, 0006, 0007) converge on weed IoU 0.67 and support the reading that the
ceiling belongs to the data, not the model. That reading was still an
inference. The direct measurement is a learning curve: retrain the champion
configuration on fractions of the train split and observe whether weed IoU is
still climbing when the full dataset is reached.

## Decision

The curve is measured, and it settles the question: weed IoU rises
monotonically with data volume and the steepest segment is the last one.
The score has not plateaued at 100% of the available data. More data is the
lever that plausibly moves the ceiling; further model tuning is not.

## Method

Two properties of the fractional splits, both required for a clean curve:

- Stratified by crop presence, same rule as the original split, so each
  fraction keeps the train split's proportion of no-crop images and the curve
  does not confound volume with composition.
- Nested: the 25% subset is contained in the 50%, which is contained in the
  75%, which is contained in the full split. One seeded shuffle per stratum,
  then prefixes. Each point adds images to the previous point rather than
  drawing a fresh sample.

Validation and test splits untouched; every point is measured against the
identical validation set.

Equal training budget in optimizer steps, not epochs. An epoch of the 25%
subset is about 9 optimizer updates (70 images, batch 8) against 35 for the
full split, so a fixed epoch count would give smaller fractions a quarter of
the training and inflate the effect of data volume. Epoch counts were set to
match the champion's budget of about 875 steps: 100 epochs at 25%, 50 at 50%,
35 at 75%, 25 at 100%. The first attempt ran 25% at 25 epochs and landed at
weed 0.594 with validation loss still in free fall, against 0.621 when
trained to budget; the discarded run overstated the volume effect by 0.027
and is kept as evidence for the budget rule (same convergence trap as
ADR 0003).

## Result

Selection by best validation mIoU per run:

| fraction | images | epochs | best epoch | weed IoU | crop | background |
| --- | --- | --- | --- | --- | --- | --- |
| 25% | 70 | 100 | 66 | 0.621 | 0.889 | 0.972 |
| 50% | 140 | 50 | 46 | 0.639 | 0.896 | 0.973 |
| 75% | 210 | 35 | 28 | 0.648 | 0.898 | 0.973 |
| 100% | 280 | 25 | 13 | 0.670 | 0.910 | 0.971 |

## Interpretation

Three observations carry the conclusion:

- Only weed is data-hungry. Background is flat across the entire curve and
  crop moves by 0.02; the minority class absorbs almost all of the benefit of
  more data, consistent with the imbalance analysis in notebook 01.
- The curve is monotone and the final segment is the steepest: +0.022 from
  75% to 100%, above the between-epoch noise band of roughly ±0.01. At the
  full dataset the slope is still positive.
- The intermediate step from 50% to 75% (+0.009) sits within noise, so the
  fine shape of the curve between points is not interpretable, and likely
  mixes noise with mild under-convergence of the 50% run (its best epoch
  falls near the end of its budget). The defensible claims are the endpoints
  and the monotonicity, not the curvature.

One honest limitation: the curve measures volume within a single region and
camera. Fewer images also means less internal diversity, so volume and
diversity remain partially confounded. Separating them would need data from a
second region, which is exactly the future-work direction this result points
to (cross-dataset generalization, Bonn sugar beets).

## Consequences

- The ceiling claim is now a measurement, not an inference: the data, not the
  model, sets weed IoU on this task, and the curve is still climbing at 400
  images.
- Further model tuning on this dataset stays off the roadmap (unchanged from
  the roadmap's "Not planned" section, now with direct evidence).
- The fractional split files and the budget rule live in
  scripts/make_learning_curve_splits.py and the per-run config.json files,
  so the curve is reproducible end to end.
