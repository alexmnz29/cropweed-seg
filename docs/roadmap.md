# Roadmap

Current state and planned next steps. Kept separate from the README, which
covers scope at a higher level.

## Done

- Data pipeline: exploration, lossless mask conversion, crop-stratified split.
- Production modules: masks, transforms, dataset, metrics, model, engine, losses.
- Loss study: focal+dice champion, weed 0.670, reproducible across seeds.
- Negative results documented: DeepLabV3+ and higher resolution, no gain.
- Test evaluation (once): weed 0.670, mIoU 0.850, matches validation.
- Error analysis: failures concentrate on thin low-contrast weed.
- ONNX export, fidelity validated three ways.
- INT8 quantization: 3.98x smaller, near-zero per-class accuracy drop.

## Next

### Jetson deployment (pending hardware decision)
Convert ONNX to TensorRT on a Jetson Orin Nano, run INT8 there, and measure
latency. Accuracy and size are already measured and transfer; latency is the
only piece that needs the device. Hardware purchase is deliberately paused.

### Learning-curve experiment
Retrain the champion on 25/50/75/100% of the train set to test whether the
ceiling is set by data volume. Turns the data-bottleneck reading from inference
into evidence. A natural place to use MLflow for tracking and Optuna for any
hyperparameter search, where the run count actually justifies them.

## Future work (documented, not scheduled)

- Cross-dataset generalization on Bonn sugar beets. Needs label-scheme and
  spectral remapping; a weed-only framing would isolate generalization from
  class mismatch.
- Edge-guided segmentation (boundary loss or edge branch) to target the
  fine-structure errors. Faces the same data ceiling.

## Not planned

More backbones, losses, or hyperparameter tuning. Three levers showed the
ceiling; further model tuning has diminishing returns.
