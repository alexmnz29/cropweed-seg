"""Measure FP16 vs INT8 engine latency and validate per-class IoU, on device.

Reports median and P95 latency plus throughput, timing GPU inference only
(copies and argmax stay outside the timed region). Both engines are also
validated for per-class IoU on the test split, the on-device counterpart of
the ORT accuracy check (separate artifacts, separate tables).

Reproducibility depends on a fixed board state (see deployment_runbook.md):
    sudo nvpmodel -m 2      # MAXN_SUPER
    sudo jetson_clocks      # freeze clocks (does not survive reboot)

Run on the Orin from ~/cropweed:
    python3 scripts/benchmark_engines.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorrt as trt
from cuda import cudart

ROOT = Path(__file__).resolve().parent.parent  # ~/cropweed on the board
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402  (CPU-only: dataset pipeline and metric)

from cropweed_seg.dataset import PeanutDataset  # noqa: E402
from cropweed_seg.metrics import ConfusionMatrixMetric  # noqa: E402
from cropweed_seg.transforms import build_transforms  # noqa: E402

WARMUP = 50
ITERS = 500
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def check(err):
    """cuda-python returns (err, ...) tuples; fail loudly on any CUDA error."""
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA error: {err}")


class EngineRunner:
    """Owns an engine, its execution context, stream, and device buffers.

    Same buffer pattern as the calibrator, extended to input + output:
    allocate once, copy per inference, reuse.
    """

    def __init__(self, engine_path: Path):
        runtime = trt.Runtime(TRT_LOGGER)
        self.engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
        self.context = self.engine.create_execution_context()

        err, self.stream = cudart.cudaStreamCreate()
        check(err)

        # Discover IO tensors (TensorRT 10 named-tensor API)
        self.io = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            nbytes = int(np.prod(shape)) * 4  # IO stays float32
            err, ptr = cudart.cudaMalloc(nbytes)
            check(err)
            self.context.set_tensor_address(name, ptr)
            mode = self.engine.get_tensor_mode(name)
            self.io[name] = {
                "ptr": ptr,
                "shape": shape,
                "nbytes": nbytes,
                "is_input": mode == trt.TensorIOMode.INPUT,
            }

        self.input = next(v for v in self.io.values() if v["is_input"])
        self.output = next(v for v in self.io.values() if not v["is_input"])

    def load_input(self, array: np.ndarray):
        """host -> device, outside any timed region."""
        array = np.ascontiguousarray(array, dtype=np.float32)
        (err,) = cudart.cudaMemcpy(
            self.input["ptr"],
            array.ctypes.data,
            array.nbytes,
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
        )
        check(err)

    def infer(self):
        """One inference. Synchronize stays INSIDE any timing around this:
        the launch is async; without sync you time the launch, not the GPU."""
        self.context.execute_async_v3(self.stream)
        (err,) = cudart.cudaStreamSynchronize(self.stream)
        check(err)

    def read_output(self) -> np.ndarray:
        """device -> host, outside any timed region."""
        out = np.empty(self.output["shape"], dtype=np.float32)
        (err,) = cudart.cudaMemcpy(
            out.ctypes.data,
            self.output["ptr"],
            self.output["nbytes"],
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        check(err)
        return out


def benchmark(runner: EngineRunner, sample: np.ndarray) -> dict:
    runner.load_input(sample)  # one representative frame, loaded once

    for _ in range(WARMUP):  # not timed: clocks, caches, allocator settling
        runner.infer()

    latencies_ms = np.empty(ITERS)
    for i in range(ITERS):
        start = time.perf_counter()
        runner.infer()  # sync inside -> this measures GPU compute
        latencies_ms[i] = (time.perf_counter() - start) * 1000.0

    median = float(np.median(latencies_ms))
    return {
        "median_ms": round(median, 2),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 2),
        "throughput_fps": round(1000.0 / median, 1),
    }


def validate_iou(runner: EngineRunner, test_ds) -> dict:
    metric = ConfusionMatrixMetric()
    for idx in range(len(test_ds)):
        image, mask = test_ds[idx]
        runner.load_input(image.unsqueeze(0).numpy())
        runner.infer()
        logits = runner.read_output()
        pred = torch.from_numpy(logits).argmax(dim=1)
        metric.update(pred, mask.unsqueeze(0))
    return {k: round(float(v), 4) for k, v in metric.compute().items()}


def main():
    models_dir = ROOT / "models"
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)

    test_ds = PeanutDataset(ROOT, "test", build_transforms("test"))
    image, _ = test_ds[0]
    sample = image.unsqueeze(0).numpy()  # representative real frame

    results = {"tensorrt_version": trt.__version__, "warmup": WARMUP, "iters": ITERS}

    for name in ["model_fp16.engine", "model_int8.engine"]:
        path = models_dir / name
        if not path.exists():
            raise FileNotFoundError(f"{path} not found. Run build_engines.py first.")
        runner = EngineRunner(path)

        stats = benchmark(runner, sample)
        iou = validate_iou(runner, test_ds)
        size_mb = round(path.stat().st_size / 1e6, 1)

        results[name] = {"size_mb": size_mb, **stats, "iou": iou}
        print(f"\n{name}  ({size_mb} MB)")
        print(
            f"  median {stats['median_ms']} ms | P95 {stats['p95_ms']} ms"
            f" | {stats['throughput_fps']} fps"
        )
        for k, v in iou.items():
            print(f"  {k:<16} {v:.4f}")

    out_path = results_dir / "benchmark.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved -> {out_path}")
    print(
        "record alongside: JetPack version, nvpmodel mode (MAXN_SUPER),"
        " jetson_clocks on, timed region = GPU inference only"
    )


if __name__ == "__main__":
    main()
