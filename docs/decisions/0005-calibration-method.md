# NN. INT8 calibration method

Status: Accepted

## Context

The champion is quantized to INT8 with ONNX Runtime `quantize_static` (QDQ
format, static PTQ). Calibration picks the activation range that gets mapped to
int8. The initial run used MinMax, which maps the full observed range: an
outlier widens the range and coarsens the step for every other value.

This left an open question. If MinMax were stretching the range to swallow
activation outliers, it would spend resolution on the tail and lose it on the
bulk. `weed` is the minority, hard class (3.7% of pixels), so its fine
distinctions would be the first to suffer. The hypothesis was that a
tail-clipping method could recover `weed` IoU. This ADR records the test.

## Decision

Keep MinMax static calibration for the deployed INT8 model.

## Alternatives considered

Three calibration methods, run through the same path. Same calibration set for
all three (50 train images, deterministic reader, no val/test leak), same test
split to measure. Only `calibrate_method` varies.

| class | fp32 | minmax | entropy | percentile | Δ(entropy−minmax) |
| --- | --- | --- | --- | --- | --- |
| background | 0.9711 | 0.9710 | 0.9710 | 0.9710 | +0.0000 |
| crop | 0.9096 | 0.9095 | 0.9095 | 0.9093 | +0.0000 |
| weed | 0.6695 | 0.6696 | 0.6696 | 0.6692 | +0.0000 |
| mIoU | 0.8500 | 0.8500 | 0.8500 | 0.8498 | +0.0000 |

`weed` is flat across MinMax and Entropy to four decimals. Entropy matching
MinMax exactly is the informative part: Entropy only gains when there is a tail
worth clipping, so an exact match means there is no such tail. The activations
are well bounded, consistent with a BatchNorm U-Net.

Percentile is the only method that moves, and it moves down: `weed` 0.6692,
mIoU 0.8498. Its default 99.999 threshold clips the very tip of the tail by
construction, so with no real outliers present it removes a sliver of
legitimate signal and loses a fraction. This is coherent, not a bug: the method
that clips blindly loses a little, the method that decides when to clip loses
nothing.

## Consequences

- The original MinMax choice holds, now confirmed by three methods converging on
  the same result rather than assumed. This mirrors the ceiling analysis, where
  three independent levers converged on `weed` 0.67.
- The accuracy effect of calibration is resolved off-device. On Jetson,
  TensorRT's default entropy calibrator (`IInt8EntropyCalibrator2`) is fine to
  use; on-device work validates mIoU as a sanity check, not a method comparison.
- The ORT INT8 model and the TensorRT engine are numerically distinct artifacts.
  This ADR isolates the method effect within ORT. The engine gets its own mIoU
  validation on device. Separate tables, separate checks.
- `weed` survives INT8 without measurable loss regardless of method, consistent
  with the near-zero per-class drop recorded in the quantization step.
