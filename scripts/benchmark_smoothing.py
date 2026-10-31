"""Compare the prior dense convolution with channel-wise Metal smoothing.

Run on an idle Apple Silicon Mac. All weights and inputs are synthetic; the
model case uses the public predict_numpy boundary, including all three networks
and host copies, but excludes image preprocessing, postprocessing and file I/O.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections.abc import Callable
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
from mlx.core import array as NativeArray

from synthseg_mlx import SynthSegModel, get_flip_indices, load_label_data, predict_numpy
from synthseg_mlx._typing import FloatArray, NativeArrayApi

SEED = 20261009
ROOT = Path(__file__).resolve().parents[1]
type Output = NativeArray | tuple[FloatArray, FloatArray, FloatArray]


def dense_kernel(channels: int, sigma: float) -> NativeArray:
    """The baseline Gaussian construction as of repository commit 68744a9."""
    import math

    width = int(math.ceil(2.5 * sigma) / 2) * 2 + 1
    coordinates = mx.arange(width, dtype=mx.float32) - (width - 1) / 2
    squared = coordinates**2
    log_kernel = -(squared[:, None, None] + squared[None, :, None] + squared[None, None, :]) / (
        2 * sigma**2
    )
    kernel = mx.exp(log_kernel)
    kernel = kernel / kernel.sum()
    return mx.eye(channels, dtype=mx.float32)[:, None, None, None, :] * kernel[None, :, :, :, None]


class ComparedModel(SynthSegModel):
    """Switch only the smoothing operator; both runs share every network tensor."""

    def __init__(self, features: int, qc_shape: int) -> None:
        """Construct one model with shared weights and both smoothing implementations."""
        assets = load_label_data()
        labels, flip, unique = get_flip_indices(assets["segmentation"], 19)
        parcellation = np.unique(assets["parcellation"])
        super().__init__(
            labels,
            parcellation,
            assets["qc"][unique],
            flip,
            features=features,
            qc_shape=qc_shape,
        )
        self.reference = False
        self.dense_segmentation = dense_kernel(len(labels), 0.5)
        self.dense_parcellation = dense_kernel(len(parcellation), 0.5)
        cast(NativeArrayApi, mx).eval(self.parameters())

    def blur(self, x: NativeArray, *, parcellation: bool = False) -> NativeArray:
        """Run the selected dense reference or current smoothing operator."""
        if not self.reference:
            return super().blur(x, parcellation=parcellation)
        kernel = self.dense_parcellation if parcellation else self.dense_segmentation
        return mx.conv3d(x, kernel, padding=kernel.shape[1] // 2)


def completed(call: Callable[[], Output]) -> Output:
    """Evaluate an operation and synchronize before returning its result."""
    result = call()
    if isinstance(result, NativeArray):
        cast(NativeArrayApi, mx).eval(result)
    mx.synchronize()
    return result


def assert_parity(reference: Output, current: Output) -> list[float]:
    """Check reference and current outputs are finite and exactly equal."""
    left = (reference,) if isinstance(reference, NativeArray) else reference
    right = (current,) if isinstance(current, NativeArray) else current
    errors: list[float] = []
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert np.isfinite(a).all() and np.isfinite(b).all()
        np.testing.assert_array_equal(a, b)
        errors.append(float(np.max(np.abs(a - b))))
    return errors


def compare(calls: dict[str, Callable[[], Output]], repeats: int) -> dict[str, object]:
    """Measure synchronized reference and current calls after verifying parity."""
    reference_name, current_name = calls
    for call in calls.values():
        for _ in range(3):
            completed(call)
    errors = assert_parity(completed(calls[reference_name]), completed(calls[current_name]))

    samples: dict[str, list[float]] = {name: [] for name in calls}
    peaks: dict[str, int] = {}
    for name, call in calls.items():
        mx.clear_cache()
        mx.reset_peak_memory()
        completed(call)
        peaks[name] = mx.get_peak_memory()
    for iteration in range(repeats):
        order = list(calls) if iteration % 2 == 0 else list(reversed(calls))
        for name in order:
            mx.synchronize()
            started = time.perf_counter()
            output = completed(calls[name])
            samples[name].append((time.perf_counter() - started) * 1000)
            del output
    statistics = {
        name: {
            "samples_ms": values,
            "median_ms": float(np.median(values)),
            "p95_ms": float(np.percentile(values, 95)),
            "min_ms": min(values),
            "max_ms": max(values),
            "peak_mlx_bytes": peaks[name],
        }
        for name, values in samples.items()
    }
    baseline = cast(float, statistics[reference_name]["median_ms"])
    optimized = cast(float, statistics[current_name]["median_ms"])
    return {
        "statistics": statistics,
        "max_abs_errors": errors,
        "bitwise_equal": True,
        "speedup": baseline / optimized,
        "latency_reduction_percent": 100 * (1 - optimized / baseline),
        "status": "improved" if optimized < 0.97 * baseline else "flat_or_regressed",
    }


def main() -> None:
    """Compare smoothing implementations and write the requested measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--qc-shape", type=int, default=224)
    args = parser.parse_args()
    if args.repeats < 3 or args.volume_size < 16 or args.volume_size % 16:
        parser.error("use at least 3 repeats and a volume size divisible by 16")
    mx.random.seed(SEED)
    model = ComparedModel(args.features, args.qc_shape)
    cases: dict[str, object] = {}

    def operator(values: NativeArray, parcellation: bool, reference: bool) -> NativeArray:
        """Select a smoothing path and run the operator benchmark."""
        model.reference = reference
        return model.blur(values, parcellation=parcellation)

    for shape in [(1, 128, 128, 128, 33), (1, 64, 64, 64, 69)]:
        host = np.random.default_rng(SEED).uniform(size=shape).astype(np.float32)
        values = mx.array(host)
        cast(NativeArrayApi, mx).eval(values)
        parcellation = shape[-1] == 69

        result = compare(
            {
                "dense": partial(operator, values, parcellation, True),
                "metal": partial(operator, values, parcellation, False),
            },
            args.repeats,
        )
        result.update(shape=shape)
        cases[f"smoothing_{shape[1]}cubed_{shape[-1]}channels"] = result
        print(json.dumps({"case": shape, "result": result}), flush=True)
        del values, host
        mx.clear_cache()

    image = (
        np.random.default_rng(SEED)
        .uniform(size=(1, args.volume_size, args.volume_size, args.volume_size, 1))
        .astype(np.float32)
    )

    def inference(reference: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Select a smoothing path and run the complete inference benchmark."""
        model.reference = reference
        return predict_numpy(model, image)

    result = compare(
        {"dense": lambda: inference(True), "metal": lambda: inference(False)}, args.repeats
    )
    result.update(
        shape=image.shape,
        features=args.features,
        qc_shape=args.qc_shape,
    )
    cases["predict_numpy_standard"] = result
    report = {
        "seed": SEED,
        "runtime": {
            "python": platform.python_version(),
            "macos": platform.mac_ver()[0],
            "mlx": version("mlx"),
            "numpy": version("numpy"),
            "medmlx_core": version("medmlx-core"),
            "device": mx.device_info(),
            "dtype": "float32",
            "mx_compile": False,
            "warmups_per_path": 3,
            "repeats": args.repeats,
            "order": "alternating dense/metal",
            "synchronization": "mx.eval native outputs and mx.synchronize inside timing",
            "timing": "warm execution; construction and kernel compilation excluded",
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "weight_source": "seeded MLX initialization; no pretrained checkpoints",
        "noise_threshold_percent": 3,
        "scope": "smoothing and standard predict_numpy; no preprocessing, postprocessing or I/O",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": "predict_numpy_standard", "result": result}), flush=True)


if __name__ == "__main__":
    main()
