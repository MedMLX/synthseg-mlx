"""Record pinned SynthSeg preprocessing on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import ast
import os
import platform
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import nibabel as nib_runtime
import numpy as np
import scipy
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from synthseg_mlx._typing import HostArray, NiftiApi

REVISION = "2a2aa3bbfccb83f8253a51ca8b329b9938a2646d"
nib = cast(NiftiApi, nib_runtime)


class ArchiveWriter(Protocol):
    def __call__(self, file: Path, **arrays: HostArray) -> None:
        """Write named arrays to a compressed archive."""
        ...


save_archive = cast(ArchiveWriter, np.savez_compressed)


class NumpyAliases(ast.NodeTransformer):
    """Replace removed scalar aliases while retaining upstream numerical code."""

    def visit_Attribute(self, node: ast.Attribute) -> ast.expr:
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "np"
            and node.attr in {"int", "float"}
        ):
            return ast.copy_location(ast.Name(id=node.attr, ctx=node.ctx), node)
        return node


def definitions(path: Path, names: set[str], namespace: dict[str, object]) -> SimpleNamespace:
    """Load only named upstream functions and their supplied dependencies."""
    code = path.read_text()
    if path.name == "edit_volumes.py":
        old = "ras_axes_flo[i], ras_axes_flo[swapped_axis_idx]"
        if code.count(old) != 1:
            raise ValueError("Unexpected upstream orientation assignment")
        code = code.replace(old, old + ".item()")
    tree = NumpyAliases().visit(ast.parse(code, filename=str(path)))
    tree.body = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    if {node.name for node in tree.body} != names:
        raise ValueError(f"Missing upstream definitions in {path}")
    scope = dict(namespace)
    exec(compile(tree, str(path), "exec"), scope)
    return SimpleNamespace(**scope)


def main() -> None:
    """Regenerate real-image preprocessing outputs using source functions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != REVISION:
        raise ValueError(f"Expected upstream {REVISION}, got {revision}")
    subprocess.run(
        [
            "git",
            "-C",
            str(args.source),
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "SynthSeg/predict_synthseg.py",
            "ext/lab2im/utils.py",
            "ext/lab2im/edit_volumes.py",
        ],
        check=True,
        capture_output=True,
    )
    scope: dict[str, object] = {
        "np": np,
        "nib": nib,
        "os": os,
        "gaussian_filter": gaussian_filter,
        "RegularGridInterpolator": RegularGridInterpolator,
    }
    utils = definitions(
        args.source / "ext/lab2im/utils.py",
        {
            "load_volume",
            "save_volume",
            "get_volume_info",
            "load_array_if_path",
            "reformat_to_list",
            "reformat_to_n_channels_array",
            "mkdir",
            "get_dims",
            "add_axis",
            "find_closest_number_divisible_by_m",
        },
        scope,
    )
    scope["utils"] = utils
    edits = definitions(
        args.source / "ext/lab2im/edit_volumes.py",
        {
            "rescale_volume",
            "crop_volume",
            "pad_volume",
            "resample_volume",
            "get_ras_axes",
            "align_volume_to_ref",
        },
        scope,
    )
    scope["edit_volumes"] = edits
    predict = definitions(args.source / "SynthSeg/predict_synthseg.py", {"preprocess"}, scope)
    ct = np.asarray(nib.load(str(args.ct)).dataobj, dtype=np.float32)
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/processing"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="synthseg-preprocess-") as temporary:
        root = Path(temporary)
        for name in (
            "resample",
            "orientation",
            "crop_ct",
            "constant",
            "downsample",
            "multichannel",
        ):
            path = fixtures / f"preprocess_{name}.npz"
            with np.load(path, allow_pickle=False) as archive:
                original = dict(archive)
            shape = (
                tuple(original["input_shape"].tolist())
                if "input_shape" in original
                else original["input"].shape
            )
            spatial = shape[:3]
            origin = (0, 0, 0) if name == "constant" else (100, 210, 45)
            crop = ct[
                tuple(
                    slice(start, start + size) for start, size in zip(origin, spatial, strict=True)
                )
            ]
            if crop.shape != spatial:
                raise ValueError(f"CT crop {crop.shape} does not match {spatial}")
            if name == "constant" and np.unique(crop).size != 1:
                raise ValueError("The constant-intensity crop must contain one CT value")
            if name == "multichannel":
                crop = np.stack((crop, np.flip(crop, axis=0)), axis=-1)
            image_path, resampled_path = root / f"{name}.nii.gz", root / f"{name}-resampled.nii.gz"
            nib.save(nib.Nifti1Image(crop, original["affine"]), str(image_path))
            result = predict.preprocess(
                str(image_path),
                bool(original["ct"]),
                target_res=original["target_res"],
                n_levels=int(original["n_levels"]),
                crop=original["crop"] if original["crop"].size else None,
                min_pad=int(original["min_pad"]),
                path_resample=str(resampled_path),
            )
            image, affine, _, resolution, output_shape, pad_idx, crop_idx = result
            output = dict(
                reference_host=np.asarray(f"{platform.system().lower()}-{platform.machine()}"),
                synthseg_commit=np.asarray(revision),
                input=crop.astype(np.float64),
                expected_image=image,
                expected_affine=affine,
                expected_resolution=resolution,
                expected_shape=np.asarray(output_shape),
                expected_pad_idx=pad_idx,
                expected_crop_idx=np.asarray([] if crop_idx is None else crop_idx),
                numpy=np.asarray(np.__version__),
                scipy=np.asarray(scipy.__version__),
                nibabel=np.asarray(nib_runtime.__version__),
            )
            if resampled_path.exists():
                resampled = nib.load(str(resampled_path))
                output["resampled_image"] = np.asarray(resampled.dataobj)
                output["resampled_affine"] = resampled.affine
            save_archive(args.output / path.name, **output)
            retained = {
                key: value
                for key, value in original.items()
                if key != "input" and not key.startswith(("expected_", "resampled_"))
            }
            retained["input_shape"] = np.asarray(shape)
            save_archive(path, **retained)


if __name__ == "__main__":
    main()
