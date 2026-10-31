"""Compare original QC scoring with sparse convolutions and fused pooling.

Both paths share float32 network weights and the optimized Gaussian smoothing.
The full case measures predict_numpy, including all three networks and host
copies, and excludes preprocessing, postprocessing and file I/O.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from functools import partial
from importlib.metadata import version
from pathlib import Path
from typing import cast

import mlx.core as mx
import numpy as np
from benchmark_smoothing import ROOT, SEED, ComparedModel, assert_parity, compare
from mlx.core import array as NativeArray

from synthseg_mlx import predict_numpy
from synthseg_mlx._typing import FloatArray, NativeArrayApi
from synthseg_mlx.mlx_model import BatchNorm3d, QCNet, crop_for_qc, nn


class ComparedQC(QCNet):
    """Frozen QC forward as of repository commit 68744a9; no weight, crop or score changes."""

    def __init__(self, labels: int, features: int) -> None:
        """Construct one QC graph with a selectable reference execution path."""
        super().__init__(labels, features=features)
        self.labels = labels
        self.reference = False

    def __call__(self, x: NativeArray) -> NativeArray:
        """Run the reference or current QC path on one-hot inputs."""
        if not self.reference:
            return super().__call__(x)
        for level in range(4):
            residual = nn.relu(self.layer(f"qc_expand_down_merge_{level}")(x))
            x = nn.relu(self.layer(f"qc_conv_downarm_{level}_0")(x))
            x = self.layer(f"qc_conv_downarm_{level}_1")(x)
            x = self.layer(f"qc_bn_down_{level}")(nn.relu(x + residual))
            x = self.pool(x)
        x = nn.relu(self.layer("qc_final_conv_0")(x))
        x = nn.relu(self.layer("qc_final_conv_1")(x))
        return x.mean(axis=(1, 2, 3))

    def score_indices(self, indices: NativeArray) -> NativeArray:
        """Score class indices through the selected QC path."""
        if not self.reference:
            return super().score_indices(indices)
        one_hot = (indices[..., None] == mx.arange(self.labels, dtype=mx.int32)).astype(mx.float32)
        return self(one_hot)


def check_layouts() -> list[dict[str, object]]:
    """Verify batches, spatial strides, negative gamma and channel-block boundaries."""
    results: list[dict[str, object]] = []
    for features, labels in ((1, 9), (17, 9), (24, 9), (32, 9), (1, 16), (1, 17)):
        mx.random.seed(SEED + features)
        network = ComparedQC(labels, features)
        rng = np.random.default_rng(SEED)
        for level in range(4):
            norm = cast(BatchNorm3d, network.layer(f"qc_bn_down_{level}"))
            size = norm.weight.shape[0]
            norm.weight = mx.array(rng.normal(size=size).astype(np.float32))
            norm.bias = mx.array(rng.normal(size=size).astype(np.float32))
            norm.running_mean = mx.array(rng.normal(size=size).astype(np.float32))
            norm.running_var = mx.array(rng.uniform(0.5, 1.5, size=size).astype(np.float32))
        network.eval()
        cast(NativeArrayApi, mx).eval(network.parameters())
        indices = rng.integers(0, labels, (2, 17, 19, 21), dtype=np.int32)
        values = mx.flip(mx.array(np.eye(labels, dtype=np.float32)[indices]).swapaxes(1, 3), axis=2)
        network.reference = True
        reference = network(values)
        network.reference = False
        current = network(values)
        errors = assert_parity(reference, current)
        native_indices = mx.flip(mx.array(indices).swapaxes(1, 3), axis=2)
        encoding_errors = assert_parity(reference, network.score_indices(native_indices))
        results.append(
            dict(
                features=features,
                classes=labels,
                shape=values.shape,
                layout="flipped spatial transpose; signed gamma and nontrivial BN statistics",
                bitwise_equal=True,
                max_abs_errors=errors,
                encoded_max_abs_errors=encoding_errors,
            )
        )
    return results


def main() -> None:
    """Compare reference and current QC operators and write the requested measurements."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--volume-size", type=int, default=128)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--qc-shape", type=int, default=224)
    args = parser.parse_args()
    if (
        args.repeats < 3
        or args.volume_size < 16
        or args.volume_size % 16
        or args.qc_shape < 16
        or args.features < 1
    ):
        parser.error("use at least 3 repeats, positive features, and a volume divisible by 16")
    parity_cases = check_layouts()
    mx.clear_cache()
    mx.random.seed(SEED)
    model = ComparedModel(args.features, args.qc_shape)
    quality = ComparedQC(model.qc_classes, args.features)
    quality.update(model.qc.parameters())
    model.qc = quality
    model.eval()
    cast(NativeArrayApi, mx).eval(model.parameters())
    image = (
        np.random.default_rng(SEED)
        .uniform(size=(1, args.volume_size, args.volume_size, args.volume_size, 1))
        .astype(np.float32)
    )
    segmentation, conditioning = model.segmentation_predictions(mx.array(image))
    cropped = mx.stack(
        [crop_for_qc(item, args.qc_shape) for item in mx.argmax(segmentation, axis=-1)]
    )
    indices = model.qc_labels[cropped]
    cast(NativeArrayApi, mx).eval(indices)
    del segmentation, conditioning, cropped

    def qc(values: NativeArray, reference: bool) -> NativeArray:
        """Select a QC path and run it on the benchmark input."""
        quality.reference = reference
        return quality.score_indices(values)

    qc_result = compare(
        {"baseline": partial(qc, indices, True), "metal": partial(qc, indices, False)},
        args.repeats,
    )
    qc_result.update(shape=indices.shape, features=args.features, classes=model.qc_classes)
    del indices
    mx.clear_cache()
    print(json.dumps({"case": "qc", "result": qc_result}), flush=True)

    def inference(reference: bool) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Select a QC path and run the complete benchmark inference."""
        quality.reference = reference
        return predict_numpy(model, image)

    full_result = compare(
        {"baseline": lambda: inference(True), "metal": lambda: inference(False)},
        args.repeats,
    )
    full_result.update(shape=image.shape, features=args.features, qc_shape=args.qc_shape)
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
            "order": "alternating baseline/metal",
            "synchronization": "mx.eval native outputs and mx.synchronize inside timing",
            "timing": "warm execution; construction and kernel compilation excluded",
        },
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "weight_source": "seeded MLX initialization; no pretrained checkpoints",
        "baseline": "QCNet forward from 68744a9; optimized Gaussian smoothing in both paths",
        "noise_threshold_percent": 3,
        "scope": (
            "QC from class indices (including encoding) and standard predict_numpy; "
            "no preprocessing, postprocessing or I/O"
        ),
        "cases": {"qc": qc_result, "predict_numpy_standard": full_result},
        "layout_parity": parity_cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"case": "predict_numpy_standard", "result": full_result}), flush=True)


if __name__ == "__main__":
    main()
