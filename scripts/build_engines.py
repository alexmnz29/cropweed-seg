"""Build FP16 and INT8 TensorRT engines from the ONNX model, on device.

Engines are hardware-specific: they must be built on the Jetson, not the Mac.
INT8 uses TensorRT's own entropy calibrator (IInt8EntropyCalibrator2) over the
same 50 train images used for the ORT quantization (train split, val transform,
first 50, no val/test leak). Accuracy equivalence of calibration methods was
validated off-device (see the calibration ADR); on-device mIoU is a sanity
check, not a method comparison.

Run on the Orin from ~/cropweed:
    python3 scripts/build_engines.py
"""

import sys
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

ROOT = Path(__file__).resolve().parent.parent  # ~/cropweed on the board
sys.path.insert(0, str(ROOT / "src"))  # make cropweed_seg importable

from cropweed_seg.dataset import PeanutDataset  # noqa: E402
from cropweed_seg.transforms import build_transforms  # noqa: E402

N_CALIBRATION = 50
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def check(err):
    """cuda-python returns (err, ...) tuples; fail loudly on any CUDA error."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    """Feeds full-frame train images to TensorRT for INT8 calibration.

    Same recipe as the ORT PeanutCalibrationReader: train split, val transform,
    first N_CALIBRATION images. Caches the calibration table so repeat builds
    skip recalibration (delete calib.cache to force a fresh one).
    """

    def __init__(self, root: Path, cache_path: Path, n_samples: int = N_CALIBRATION):
        super().__init__()
        self.cache_path = cache_path

        ds = PeanutDataset(root, "train", build_transforms("val"))
        self.samples = [
            np.ascontiguousarray(ds[i][0].unsqueeze(0).numpy(), dtype=np.float32)
            for i in range(n_samples)
        ]
        self.idx = 0

        # One device buffer, sized for a single sample, reused for every image.
        nbytes = self.samples[0].nbytes  # 1*3*720*960*4 bytes
        err, self.device_input = cudart.cudaMalloc(nbytes)
        check(err)

    def get_batch_size(self):
        return 1

    def get_batch(self, names):
        if self.idx >= len(self.samples):
            return None  # tells TensorRT calibration data is exhausted
        sample = self.samples[self.idx]
        # host (numpy, CPU) -> device (GPU) copy
        (err,) = cudart.cudaMemcpy(
            self.device_input,
            sample.ctypes.data,
            sample.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check(err)
        self.idx += 1
        # TensorRT wants the raw device address of the input, as an int
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if self.cache_path.exists():
            print(f"using cached calibration table: {self.cache_path.name}")
            return self.cache_path.read_bytes()
        return None

    def write_calibration_cache(self, cache):
        self.cache_path.write_bytes(cache)
        print(f"calibration table cached: {self.cache_path.name}")


def build_engine(onnx_path: Path, out_path: Path, precision: str, calib=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    # 2 GB workspace: room for tactic timing without starving the 8 GB board
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        # FP16 fallback: layers without INT8 support drop to fp16, not fp32
        config.set_flag(trt.BuilderFlag.FP16)
        config.int8_calibrator = calib
    else:
        raise ValueError(precision)

    print(f"building {precision} engine (this takes a while on the Orin)...")
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError(f"engine build failed for {precision}")
    out_path.write_bytes(engine)
    print(f"{precision:<5} -> {out_path.name} ({out_path.stat().st_size / 1e6:.1f} MB)")


def main():
    models_dir = ROOT / "models"
    onnx_path = models_dir / "model.onnx"  # the fp32 ONNX; TensorRT starts here
    if not onnx_path.exists():
        raise FileNotFoundError(f"{onnx_path} not found")

    build_engine(onnx_path, models_dir / "model_fp16.engine", "fp16")

    calib = EntropyCalibrator(
        root=ROOT,
        cache_path=models_dir / "calib.cache",
    )
    build_engine(onnx_path, models_dir / "model_int8.engine", "int8", calib=calib)

    print("\ndone. engines in", models_dir)


if __name__ == "__main__":
    main()
