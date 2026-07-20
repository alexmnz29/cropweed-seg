"""Generate demo overlay frames with the INT8 engine, on device.

Runs the INT8 TensorRT engine over the validation split and saves one overlay
frame per image: crop in green, weed in magenta, background transparent
(alpha keeps the field visible under the mask). Honest framing for the article:
inference on a Jetson Orin Nano over the validation set, pre-recorded source.

No timing happens here. Latency numbers come from benchmark_engines.py; this
pass only produces frames. Each filename carries the per-image weed IoU so
frames can be sorted for curation (best sweep + hard cases) without opening
them.

Run on the Orin from ~/cropweed:
    python3 scripts/make_demo_frames.py
Then pull frames to the Mac:
    scp -r alex@<jetson-ip>:~/cropweed/results/demo_frames .
"""

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from cropweed_seg.dataset import PeanutDataset  # noqa: E402
from cropweed_seg.metrics import ConfusionMatrixMetric  # noqa: E402
from cropweed_seg.transforms import build_transforms  # noqa: E402

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Class indices as used throughout the project (metrics print in this order).
# If overlay colors land on the wrong plants, these two are the knob.
CLASS_CROP = 1
CLASS_WEED = 2

# Overlay design, as decided: crop green, weed magenta (red/green-safe pair),
# background untouched, alpha 0.45 so real field texture shows through.
COLOR_CROP = (70, 170, 70)
COLOR_WEED = (225, 50, 160)
ALPHA = 0.45

# ImageNet normalization, inverted to recover the display image from the
# dataset tensor. Denormalizing guarantees pixel-perfect alignment between
# what the model saw and what the overlay is drawn on.
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def check(err):
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


class EngineRunner:
    """Same pattern as benchmark_engines.py: allocate once, copy per use."""

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()
        err, self.stream = cudart.cudaStreamCreate()
        check(err)

        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * 4
            err, ptr = cudart.cudaMalloc(nbytes)
            check(err)
            self.context.set_tensor_address(name, ptr)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            self.io[name] = {
                "ptr": ptr,
                "shape": shape,
                "nbytes": nbytes,
                "is_input": is_input,
            }

        self.input = next(v for v in self.io.values() if v["is_input"])
        self.output = next(v for v in self.io.values() if not v["is_input"])

    def predict(self, array: np.ndarray) -> np.ndarray:
        array = np.ascontiguousarray(array, dtype=np.float32)
        (err,) = cudart.cudaMemcpy(
            self.input["ptr"],
            array.ctypes.data,
            array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check(err)
        self.context.execute_async_v3(self.stream)
        (err,) = cudart.cudaStreamSynchronize(self.stream)
        check(err)
        out = np.empty(self.output["shape"], dtype=np.float32)
        (err,) = cudart.cudaMemcpy(
            out.ctypes.data,
            self.output["ptr"],
            self.output["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        check(err)
        return out


def denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Dataset tensor (3,H,W, ImageNet-normalized) -> uint8 RGB (H,W,3)."""
    img = tensor.numpy().transpose(1, 2, 0)  # HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)


def compose_overlay(image_rgb: np.ndarray, pred: np.ndarray) -> Image.Image:
    """Paint crop and weed with ALPHA over the image; background untouched."""
    out = image_rgb.astype(np.float32)
    for cls, color in ((CLASS_CROP, COLOR_CROP), (CLASS_WEED, COLOR_WEED)):
        mask = pred == cls
        out[mask] = (1.0 - ALPHA) * out[mask] + ALPHA * np.array(color)
    return Image.fromarray(out.astype(np.uint8))


def per_image_weed_iou(pred: torch.Tensor, mask: torch.Tensor) -> float:
    metric = ConfusionMatrixMetric()
    metric.update(pred, mask.unsqueeze(0))
    return float(metric.compute()["iou_weed"])


def main():
    engine_path = ROOT / "models" / "model_int8.engine"
    out_dir = ROOT / "results" / "demo_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = EngineRunner(engine_path)
    val_ds = PeanutDataset(ROOT, "val", build_transforms("val"))
    print(f"generating {len(val_ds)} overlay frames with the INT8 engine...")

    for idx in range(len(val_ds)):
        image_t, mask = val_ds[idx]
        logits = runner.predict(image_t.unsqueeze(0).numpy())
        pred_t = torch.from_numpy(logits).argmax(dim=1)  # (1,H,W)
        pred = pred_t[0].numpy()

        wiou = per_image_weed_iou(pred_t, mask)
        frame = compose_overlay(denormalize(image_t), pred)
        name = f"frame_{idx:03d}_wiou{wiou:.2f}.png"
        frame.save(out_dir / name)
        print(f"  {name}")

    print(f"\ndone -> {out_dir}")
    print("high wiou = clean sweep material, low wiou = hard-case closers")


if __name__ == "__main__":
    main()
