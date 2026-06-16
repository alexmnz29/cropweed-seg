# 0003 · Loss function for class imbalance

Status: accepted
Date: 2026-06-14

## Context

The weighted cross-entropy baseline reached weed IoU 0.496, roughly half the IoU
of background and crop. weed is the minority class at 3.7% of pixels and consists
of small scattered regions, so it is the bottleneck. The question was whether a
different loss handles the imbalance better.

## Decision

Focal + Dice, summed with equal weight. Across two seeds at 25 epochs it reaches
weed IoU 0.670 and mIoU 0.849, both identical to three decimals between seeds.

## Alternatives considered

Four strategies were compared, all measured by per-class IoU at the best mIoU
epoch, same split and seed handling.

| loss | mIoU | weed IoU | notes |
| --- | --- | --- | --- |
| weighted cross-entropy | 0.768 | 0.496 | baseline |
| focal | 0.840 | 0.650 | stable plateau |
| dice | 0.838 | 0.643 | noisy, dipped to 0.45 mid-plateau |
| focal + dice | 0.849 | 0.670 | best, reproducible across two seeds |

Weighted cross-entropy is clearly the weakest. All three alternatives beat it on
weed by a large margin. Focal and Dice tie on peak but Dice is noisier per epoch,
consistent with its known instability on very small objects (small denominators).
Focal + Dice combines pixel-level signal (focal) with region-level overlap (dice),
which are different axes, and it converges to the highest and most stable result.

## The convergence episode

The first comparison ran every loss at 15 epochs and nearly concluded "focal alone,
for simplicity" because focal and focal+dice looked tied at ~0.63 to 0.64. That was
wrong. Neither had converged at 15 epochs. Extending both to 25 epochs at equal
training budget showed focal rising to 0.650 and focal+dice to 0.670, a real and
consistent gap. The lesson: comparing the best of N epochs penalizes the loss that
converges more slowly, and a fair comparison requires equal training budget with
both losses actually converged. Focal+dice converges slower because summing two
objectives gives a mixed gradient landscape early on.

## Consequences

The official training configuration moves to focal+dice at 25 epochs. The model.py
and train.py defaults change accordingly.

weed converges around 0.670 across losses and seeds and stays well below background
and crop. This is read as close to the practical ceiling of the task for this
architecture, not a training failure. Tran & Phan, the dataset authors, report that
classifying weed pixels is harder than the other classes, which is consistent with
this result. The improvement from the original baseline is +0.174 weed IoU
(0.496 to 0.670).

The epoch count is treated as a working value, not an optimum. It interacts with
learning rate, scheduler, and possible early stopping, so it is left to be tuned
together with those when the champion model is trained, not in isolation now.

Remaining axes are not exhausted: non-empty crop sampling, input resolution, and
a DeepLabV3+ architecture are open as future work. The ceiling claim is a reasoned
hypothesis backed by the literature, not a certainty.
