"""Replay the pinned standard fork's float64 reporting over retained TF outputs.

This recorder executes original host definitions at the pinned fork revision, never the MLX port or
TensorFlow inference. It leaves the original float32 reference archive unchanged.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
from reference_types import FloatArray, HostArray, IndexArray, ReferenceNamespace
from scipy.ndimage import find_objects, label

REVISION = "d6a64db19ec7b8b71c64edd59e7e485e0235c26c"
REFERENCE_REVISION = "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d"


def array_fact(array: HostArray) -> dict[str, object]:
    """Summarize one recorded numeric array for source comparison."""
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def definitions(
    root: Path,
    path: str,
    names: set[str],
    namespace: dict[str, object],
    provenance: dict[str, object],
) -> SimpleNamespace:
    """Read source reporting definitions from the pinned Git revision."""
    raw = subprocess.check_output(["git", "show", f"{REVISION}:{path}"], cwd=root)
    text = raw.decode("utf-8")
    tree = ast.parse(text)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise ValueError(f"Missing reporting definitions: {path}")
    provenance[path] = {"functions": sorted(names)}
    tree.body = list(nodes)
    exec(compile(tree, f"{REVISION}:{path}", "exec"), namespace)
    return SimpleNamespace(**namespace)


def record(root: Path, fixture: Path, output: Path) -> None:
    """Record source reporting values on the supported host."""
    if (platform.system(), platform.machine()) != ("Darwin", "arm64"):
        raise RuntimeError("Record references on macOS arm64")
    provenance: dict[str, object] = {}
    namespace: dict[str, object] = {"np": np, "scipy_label": label, "find_objects": find_objects}
    namespace["utils"] = definitions(
        root, "ext/lab2im/utils.py", {"get_dims", "reformat_to_list"}, dict(namespace), provenance
    )
    namespace["edit_volumes"] = definitions(
        root,
        "ext/lab2im/edit_volumes.py",
        {
            "crop_volume_with_idx",
            "get_ras_axes",
            "align_volume_to_ref",
            "get_largest_connected_component",
        },
        dict(namespace),
        provenance,
    )
    source = cast(
        ReferenceNamespace,
        definitions(
            root,
            "SynthSeg/predict_synthseg.py",
            {"postprocess", "largest_component", "argmax_channels"},
            dict(namespace),
            provenance,
        ),
    )
    with np.load(fixture, allow_pickle=False) as archive:
        values = {name: cast(HostArray, archive[name]) for name in archive.files}
    if (
        values["reference_host"].item() != "darwin-arm64"
        or values["synthseg_commit"].item() != REFERENCE_REVISION
        or any(values[flag].item() for flag in ("robust", "fast", "v1"))
    ):
        raise ValueError("Expected the retained standard TensorFlow recording")
    mask, posteriors, volumes = source.postprocess(
        cast(FloatArray, values["segmentation"]).copy(),
        cast(FloatArray, values["parcellation"]).copy(),
        labels_segmentation=cast(IndexArray, values["labels_segmentation"]),
        labels_parcellation=cast(IndexArray, values["labels_parcellation"]),
        fast=False,
        topology_classes=cast(IndexArray, values["topology_classes"]),
        v1=False,
        shape=[16] * 3,
        pad_idx=np.array([0, 0, 0, 16, 16, 16]),
        crop_idx=None,
        aff=np.eye(4),
        im_res=np.ones(3),
    )
    if not np.array_equal(mask, values["mask"]) or not np.array_equal(
        posteriors, values["posteriors"]
    ):
        raise ValueError("Fork replay differs from the retained source labels/posteriors")
    if volumes.dtype != np.float64:
        raise ValueError("Standard fork reporting requires float64 accumulation")
    result = {
        "source_revision": REVISION,
        "source_files": provenance,
        "reference_host": "darwin-arm64",
        "runtime": {name: version(name) for name in ("numpy", "scipy", "nibabel")},
        "input_fixture": fixture.name,
        "inputs": {
            name: array_fact(values[name])
            for name in (
                "segmentation",
                "parcellation",
                "labels_segmentation",
                "labels_parcellation",
                "topology_classes",
            )
        },
        "mask": array_fact(mask),
        "posteriors": array_fact(posteriors),
        "volumes": array_fact(volumes),
        "expected_volumes": volumes.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Recorded {output}: {volumes[0]} mm³, float64")


def main() -> None:
    """Record the upstream reporting fixture from the supplied source."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthseg-src", type=Path, required=True)
    parser.add_argument(
        "--fixture", type=Path, default=Path("tests/fixtures/variants/standard.npz")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("tests/fixtures/reporting/standard_float64.json")
    )
    args = parser.parse_args()
    record(args.synthseg_src, args.fixture, args.output)


if __name__ == "__main__":
    main()
