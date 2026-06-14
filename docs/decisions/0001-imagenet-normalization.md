# 0001 · ImageNet normalization for a pretrained backbone

Status: accepted
Date: 2026-06-14

## Context

Input images need normalization before the model. Two options exist for the
mean and std: the statistics of ImageNet, or statistics computed from the
peanut training set itself. The peanut images do not look like ImageNet. They
are top-down shots of soil and seedlings, not the natural-object distribution
ImageNet was built from. By the usual rule of thumb, a dataset that differs
from ImageNet is a reason to compute custom statistics.

## Decision

Normalize with ImageNet statistics: mean [0.485, 0.456, 0.406], std
[0.229, 0.224, 0.225], in RGB order, applied after scaling pixels to [0, 1].

The deciding factor is the pretrained backbone. The baseline uses a backbone
pretrained on ImageNet, and its weights were learned on inputs normalized with
exactly these statistics. Feeding it data in a different distribution would work
against the pretrained features the transfer is meant to exploit. The
convention for transfer learning is to normalize with the statistics of the
pretraining set, not the target set, even when the target domain differs.

Normalization applies to the image only. The mask passes through unchanged, which
was verified in notebooks/02_transforms_validation.ipynb.

## Alternatives considered

Custom statistics from the peanut training set. Correct if the model were
trained from scratch, since there would be no pretrained weights with a prior
expectation to respect. Rejected for the baseline because it conflicts with the
pretrained backbone.

## Consequences

The choice is coupled to the pretrained backbone. If a future experiment trains
a model from scratch, this decision does not apply and custom statistics should
be computed from the training split only, to avoid leaking test information.

A cheap follow-up experiment is open: compute the real per-channel statistics of
the peanut training set and compare them against ImageNet, to quantify how far
the domain sits from the pretraining distribution.
