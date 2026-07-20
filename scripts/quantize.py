"""Quantize an ONNX model to INT8 and measure size and per-class accuracy drop.

Static INT8 quantization (QDQ, MinMax calibration) using train images for
calibration, never val/test, to avoid leaking evaluation data. Reports size
reduction and per-class IoU drop on the test split. Latency benefit is not
measured here; it requires TensorRT on Jetson hardware. The accuracy drop,
however, is hardware-independent and transfers. Run with:
    uv run scripts/quantize.py --run focal_dice_s42
"""

import argparse
from pathlib import Path

import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)

from cropweed_seg.dataset import PeanutDataset
from cropweed_seg.metrics import ConfusionMatrixMetric
from cropweed_seg.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent
N_CALIBRATION = 50


class CalibrationReader(CalibrationDataReader):
    """Feeds full-frame train images to the quantizer for range calibration.

    Train split only (no val/test leak). Full-frame transform to match the
    model's fixed 720x960 input shape.
    """

    def __init__(self, root: Path, n_samples: int) -> None:
        ds = PeanutDataset(root, "train", build_transforms("val"))
        self.samples = [ds[i][0].unsqueeze(0).numpy() for i in range(n_samples)]
        self.idx = 0

    def get_next(self):
        if self.idx >= len(self.samples):
            return None
        sample = {"input": self.samples[self.idx]}
        self.idx += 1
        return sample


def evaluate_onnx(model_path: Path, test_ds: PeanutDataset) -> dict:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    metric = ConfusionMatrixMetric()
    for idx in range(len(test_ds)):
        image, mask = test_ds[idx]
        logits = session.run(["logits"], {"input": image.unsqueeze(0).numpy()})[0]
        metric.update(torch.from_numpy(logits).argmax(dim=1), mask.unsqueeze(0))
    return metric.compute()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run dir under runs/")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    fp32_path = run_dir / "model.onnx"
    int8_path = run_dir / "model_int8.onnx"
    if not fp32_path.exists():
        raise FileNotFoundError(f"{fp32_path} not found. Run export_onnx.py first.")

    print(f"calibrating with {N_CALIBRATION} train images...")
    reader = CalibrationReader(ROOT, N_CALIBRATION)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QInt8,
    )

    fp32_mb = fp32_path.stat().st_size / 1e6
    int8_mb = int8_path.stat().st_size / 1e6
    print(
        f"\nfp32: {fp32_mb:.1f} MB | int8: {int8_mb:.1f} MB | reduction: {fp32_mb / int8_mb:.2f}x"
    )

    print("\nper-class IoU on test (fp32 vs int8):")
    test_ds = PeanutDataset(ROOT, "test", build_transforms("test"))
    fp32_iou = evaluate_onnx(fp32_path, test_ds)
    int8_iou = evaluate_onnx(int8_path, test_ds)
    print(f"  {'class':<16} {'fp32':>8} {'int8':>8} {'drop':>8}")
    for k in fp32_iou:
        drop = fp32_iou[k] - int8_iou[k]
        print(f"  {k:<16} {fp32_iou[k]:>8.4f} {int8_iou[k]:>8.4f} {drop:>+8.4f}")


if __name__ == "__main__":
    main()
