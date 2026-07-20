"""Export a trained checkpoint to ONNX and validate fidelity.

Exports the champion to ONNX (fixed 1x3x720x960 input, full-frame inference) and
verifies the export is faithful three ways: numerical equivalence on a random
input, exact prediction agreement, and matching per-class IoU on the test split.
Classic TorchScript exporter is used over dynamo for maturity of the ONNX to
TensorRT path. Run with: uv run scripts/export_onnx.py --run focal_dice_s42
"""

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from cropweed_seg.dataset import PeanutDataset
from cropweed_seg.metrics import ConfusionMatrixMetric
from cropweed_seg.model import build_model
from cropweed_seg.transforms import build_transforms

ROOT = Path(__file__).resolve().parent.parent
INPUT_H, INPUT_W = 720, 960
OPSET = 17


def export(model: torch.nn.Module, out_path: Path) -> None:
    dummy = torch.randn(1, 3, INPUT_H, INPUT_W)
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=OPSET,
        dynamo=False,
        dynamic_axes=None,
    )
    print(f"exported to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


def check_equivalence(model: torch.nn.Module, onnx_path: Path) -> None:
    """Numerical and prediction equivalence on a random input."""
    x = torch.randn(1, 3, INPUT_H, INPUT_W)
    with torch.no_grad():
        torch_out = model(x).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = session.run(["logits"], {"input": x.numpy()})[0]

    max_diff = np.abs(torch_out - onnx_out).max()
    agreement = (torch_out.argmax(1) == onnx_out.argmax(1)).mean()
    print(f"max abs diff: {max_diff:.2e}")
    print(f"prediction agreement: {agreement:.4%}")
    if agreement < 1.0:
        print("WARNING: predictions do not fully agree")


def check_test_iou(onnx_path: Path) -> None:
    """Per-class IoU of the ONNX model on the test split."""
    test_ds = PeanutDataset(ROOT, "test", build_transforms("test"))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    metric = ConfusionMatrixMetric()
    for idx in range(len(test_ds)):
        image, mask = test_ds[idx]
        logits = session.run(["logits"], {"input": image.unsqueeze(0).numpy()})[0]
        metric.update(torch.from_numpy(logits).argmax(dim=1), mask.unsqueeze(0))
    iou = metric.compute()
    print("ONNX model on test split:")
    for k, v in iou.items():
        print(f"  {k:<16} {v:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="run dir under runs/")
    args = parser.parse_args()

    run_dir = ROOT / "runs" / args.run
    ckpt = run_dir / "model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} not found")

    config_path = run_dir / "config.json"
    if config_path.exists():
        cfg = json.loads(config_path.read_text())
        arch, encoder = cfg["architecture"], cfg.get("encoder", "resnet34")
    else:
        arch, encoder = "unet", "resnet34"

    model = build_model(architecture=arch, encoder_name=encoder)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    onnx_path = run_dir / "model.onnx"
    export(model, onnx_path)
    print()
    check_equivalence(model, onnx_path)
    print()
    check_test_iou(onnx_path)


if __name__ == "__main__":
    main()
