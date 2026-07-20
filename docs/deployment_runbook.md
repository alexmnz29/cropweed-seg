# Jetson deployment runbook

Order of operations on the Orin Nano. The board state steps are not optional:
without them the FP16 and INT8 numbers are not comparable.

## 1. Fix the board state (before any measurement)

    sudo nvpmodel -m 0        # TODO: confirm the mode id for MAXN / 67 TOPS
    sudo jetson_clocks        # freeze clocks at their max for this mode

- nvpmodel sets the power/frequency ceiling. Mode must be the same for both engines.
- jetson_clocks stops the board scaling frequencies up and down mid-run.
- Record which mode was used. A latency number without the power mode is meaningless.

## 2. Build engines

    uv run scripts/build_engines.py --run focal_dice_s42

Builds fp16 and int8 from model.onnx. INT8 calibrates with TensorRT's entropy
calibrator (accuracy already validated off-device, see the calibration ADR).

## 3. Sanity check with trtexec (optional, fast)

    trtexec --loadEngine=runs/focal_dice_s42/model_fp16.engine
    # TODO: same for int8; read off the rough latency to confirm sane before the real run

## 4. Benchmark

    uv run scripts/benchmark_engines.py --run focal_dice_s42

Median + P95 latency, throughput, and INT8 mIoU validation.

## 5. Record with the numbers

    # TODO: fill this environment block next to the results table
    - JetPack version:
    - TensorRT version:
    - nvpmodel mode:
    - jetson_clocks: on
    - warmup / iters: 50 / 500
    - timed region: GPU inference only
    - board temperature (optional):
