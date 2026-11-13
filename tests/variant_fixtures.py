"""Read TensorFlow recordings and reconstruct their small native MLX weights."""

import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal, Protocol, cast

import mlx.core as mx
import numpy as np
from recording_types import Recording

from synthseg_mlx._typing import DoubleArray, IntArray, NativeArrayApi, RealArray


class SafeTensorWriter(Protocol):
    def save_safetensors(self, file: str, arrays: dict[str, mx.array]) -> None:
        """Write named native arrays as a safetensors file."""
        ...


FIXTURES = Path(__file__).parent / "fixtures" / "variants"


def load_recording(path: Path) -> Recording:
    """Load one recorded fixture archive without pickle."""
    with np.load(path, allow_pickle=False) as payload:
        fixture = dict(payload)
    if (path.parent.name == "processing" and path.stem.startswith("preprocess_")) or (
        path.parent.name == "variants"
    ):
        default = Path(__file__).resolve().parents[1] / "work/real-reference"
        directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
        if path.parent.name == "variants":
            directory /= "variants"
        with np.load(directory / path.name, allow_pickle=False) as payload:
            fixture.update(dict(payload))
    assert fixture["reference_host"].item() == "darwin-arm64"
    assert fixture["synthseg_commit"].item() == "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d"
    return cast(Recording, fixture)


def load_variant(name: str) -> Recording:
    """Load the fixture for a named SynthSeg variant."""
    return load_recording(FIXTURES / f"{name}.npz")


def load_standard_volume_reference(kind: Literal["graph", "runner"] = "graph") -> DoubleArray:
    """Read original-fork float64 reporting of real CT TF probabilities."""
    default = Path(__file__).resolve().parents[1] / "work/real-reference"
    directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
    with np.load(directory / "standard_reporting.npz", allow_pickle=False) as reference:
        assert reference["source_revision"].item() == "d6a64db19ec7b8b71c64edd59e7e485e0235c26c"
        return cast(DoubleArray, reference[f"{kind}_volumes"])


def assert_reporting_units(actual: object, expected: object, *, decimals: int) -> None:
    """Allow one output rounding unit, rejecting two units or nonfinite values.

    Upstream reports volumes to 0.001 mm³ and QC to 0.0001. A one-unit
    difference can exceed that decimal bound when represented as binary floats.
    Quantizing the reported numbers avoids that artifact without widening the bound.
    """

    left, right = (
        np.asarray(cast(RealArray, actual), dtype=np.float64),
        np.asarray(cast(RealArray, expected), dtype=np.float64),
    )
    assert left.shape == right.shape
    assert np.isfinite(left).all() and np.isfinite(right).all()
    units = 10**decimals
    difference = np.abs(np.rint(left * units) - np.rint(right * units))
    assert np.all(difference <= 1), f"Reporting-unit differences: {difference}"


def write_weights(directory: Path, fixture: Recording) -> list[Path]:
    """Write the fixture checkpoint files for the selected variant as Hub-style links."""
    segmentation = "synthseg_1.0.safetensors" if fixture["v1"] else "synthseg_2.0.safetensors"
    if fixture["robust"]:
        segmentation = "synthseg_robust_2.0.safetensors"
    paths = [
        directory / name
        for name in (
            segmentation,
            "synthseg_parc_2.0.safetensors",
            "synthseg_qc_2.0.safetensors",
        )
    ]
    fields = (
        "layers",
        "keys",
        "shapes",
        "seeds",
        "scales",
        "offsets",
        "boost_indices",
        "boost_values",
    )
    rows = cast(
        Iterable[
            tuple[
                np.str_, np.str_, IntArray, np.uint32, np.float32, np.float32, np.int32, np.float32
            ]
        ],
        zip(*(fixture[f"weight_{field}"] for field in fields), strict=True),
    )
    tensors: list[dict[str, mx.array]] = [{}, {}, {}]
    attributes = {
        "kernel:0": "weight",
        "bias:0": "bias",
        "gamma:0": "weight",
        "beta:0": "bias",
        "moving_mean:0": "running_mean",
        "moving_variance:0": "running_var",
    }
    for layer, key, padded_shape, seed, scale, offset, boost_index, boost in rows:
        shape = tuple(int(size) for size in padded_shape if size >= 0)
        values = np.random.default_rng(int(seed)).uniform(-1, 1, shape).astype(np.float32)
        values = values * scale + offset
        if boost_index >= 0:
            values[int(boost_index)] += boost
        if key == "kernel:0":
            values = np.ascontiguousarray(values.transpose(4, 0, 1, 2, 3))
        index = 1 if layer.startswith("unet_parc") else 2 if layer.startswith("qc") else 0
        prefix = ""
        if index == 0 and fixture["robust"]:
            prefix = (
                "denoiser."
                if layer.startswith("l2l")
                else ("fine." if layer.startswith("unet2") else "coarse.")
            )
        tensors[index][f"{prefix}{layer}.{attributes[str(key)]}"] = cast(NativeArrayApi, mx).array(
            values
        )
    for path, weights in zip(paths, tensors, strict=True):
        # Hub snapshots link the safetensors names to extensionless blobs; load through one.
        blob = directory / f"{path.stem}-blob"
        cast(SafeTensorWriter, mx).save_safetensors(str(path), weights)
        path.rename(blob)
        path.symlink_to(blob)
    return paths
