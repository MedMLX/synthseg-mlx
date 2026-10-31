"""Runner declaration, admission and NIfTI/CSV file boundaries without Metal."""

import csv
import inspect
import subprocess
import sys
import tomllib
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TypedDict, cast

import nibabel as nib_runtime
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from synthseg_mlx import labels as package_labels
from synthseg_mlx import medmlx
from synthseg_mlx._lab2im import utils as package_utils
from synthseg_mlx._typing import DoubleArray, NiftiApi, RealArray

nib = cast(NiftiApi, nib_runtime)


class RunPaths(TypedDict):
    image_path: Path
    weights_path: Path
    output_dir: Path


ROOT = Path(__file__).resolve().parents[1]


def test_entry_point_declares_the_runner() -> None:
    """Check the package entry point and model runner declaration."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["entry-points"]["medmlx.models"] == {"synthseg": "synthseg_mlx.medmlx:MODEL"}
    assert any(dependency.startswith("nibabel") for dependency in project["dependencies"])
    model = medmlx.MODEL
    assert set(model) == {"id", "label", "task", "run", "summary"}
    assert (model["id"], model["task"]) == ("synthseg", "segmentation")
    assert all(isinstance(value, str) and value for value in model.values())
    module, _, name = model["run"].partition(":")
    assert (module, name) == ("synthseg_mlx.medmlx", "run")
    parameters = inspect.signature(medmlx.run).parameters
    assert set(parameters) == {
        "image_path",
        "weights_path",
        "output_dir",
        "ct",
        "robust",
        "fast",
        "v1",
        "crop",
        "parc",
        "save_posteriors",
        "overwrite",
        "arithmetic_profile",
    }
    assert all(item.kind is inspect.Parameter.KEYWORD_ONLY for item in parameters.values())
    assert parameters["ct"].default is False
    assert parameters["crop"].default is None
    assert parameters["parc"].default is True
    assert parameters["save_posteriors"].default is False
    assert parameters["overwrite"].default is False


def write_image(path: Path, data: RealArray, affine: DoubleArray | None = None) -> Path:
    """Write a synthetic NIfTI image with the requested voxel data and affine."""
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(np.asarray(data, np.float32), np.eye(4) if affine is None else affine),
        str(path),
    )
    return path


@pytest.fixture
def placeholder_weights(tmp_path: Path) -> Path:
    """Files that pass the existence check but are not real checkpoints."""

    directory = tmp_path / "weights"
    directory.mkdir()
    for name in (
        "synthseg_2.0.safetensors",
        "synthseg_parc_2.0.safetensors",
        "synthseg_qc_2.0.safetensors",
    ):
        (directory / name).write_bytes(b"not a checkpoint")
    return directory


def valid_image(path: Path) -> Path:
    """Write a reproducible nonconstant test brain volume."""
    return write_image(path, np.random.default_rng(5).random((12, 13, 14)))


def test_omitted_weights_use_the_shared_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check omitted weights request the public SynthSeg bundle."""
    import medmlx_core.weights as managed_weights

    requests: list[str] = []
    image = tmp_path / "brain.nii.gz"
    image.write_bytes(b"not read by this test")

    def download(model_id: str) -> Path:
        """Record the requested model ID and return a fixture folder."""
        requests.append(model_id)
        return tmp_path / "native-weights"

    def stop(folder: Path, _variant: medmlx.SynthSegVariant) -> None:
        """Check the resolved model folder and stop before checkpoint loading."""
        assert folder == tmp_path / "native-weights"
        raise RuntimeError("weight loader reached")

    monkeypatch.setattr(managed_weights, "weights_dir", download)
    monkeypatch.setattr(medmlx, "_require_weight_files", stop)
    with pytest.raises(RuntimeError, match="weight loader reached"):
        medmlx.run(image_path=image, output_dir=tmp_path / "out")
    assert requests == ["synthseg"]


@pytest.mark.parametrize(
    ("data", "affine", "message"),
    [
        (np.random.default_rng(1).random((12, 13)), None, "one 3D brain volume"),
        (np.random.default_rng(1).random((12, 13, 14, 2)), None, "one 3D brain volume"),
        (np.random.default_rng(1).random((12, 2, 14)), None, "one 3D brain volume"),
        (np.full((12, 13, 14), 3.0), None, "finite, nonconstant"),
        (np.where(np.eye(12, 13)[..., None] > 0, np.nan, 1.0) * np.ones(14), None, "finite"),
        (
            np.random.default_rng(1).random((12, 13, 14)),
            np.diag([1e-3, 1e-3, 1e-3, 1.0]),
            "geometry",
        ),
    ],
    ids=["2d", "4d", "thin", "constant", "nan", "singular"],
)
def test_run_rejects_invalid_images(
    tmp_path: Path,
    placeholder_weights: Path,
    data: RealArray,
    affine: DoubleArray | None,
    message: str,
) -> None:
    """Reject unsupported dimensions, non-finite voxels, and invalid geometry."""
    image = write_image(tmp_path / "image.nii.gz", data, affine)
    with pytest.raises(InvalidInputError, match=message):
        medmlx.run(image_path=image, weights_path=placeholder_weights, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("units", [1, 3, 7], ids=["metres", "microns", "invalid"])
def test_run_rejects_non_mm_or_invalid_spatial_units(
    tmp_path: Path, placeholder_weights: Path, version: int, units: int
) -> None:
    """The resampling and volume calculations interpret voxel dimensions as mm."""
    data = np.random.default_rng(9).random((12, 13, 14)).astype(np.float32)
    image = (nib.Nifti1Image if version == 1 else nib.Nifti2Image)(data, np.eye(4))
    image.header["xyzt_units"] = units
    path = tmp_path / "image.nii.gz"
    nib.save(image, str(path))
    with pytest.raises(InvalidInputError, match="spatial units"):
        medmlx.run(image_path=path, weights_path=placeholder_weights, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_run_rejects_spacing_that_conflicts_with_affine(
    tmp_path: Path, placeholder_weights: Path, version: int
) -> None:
    """Header spacing and the selected affine must describe the same voxel sizes."""
    data = np.random.default_rng(9).random((12, 13, 14)).astype(np.float32)
    image = (nib.Nifti1Image if version == 1 else nib.Nifti2Image)(
        data, np.diag([2.0, 2.0, 2.0, 1.0])
    )
    image.header["xyzt_units"] = 2
    image.header.set_zooms((1.0, 1.0, 1.0))
    path = tmp_path / "image.nii.gz"
    nib.save(image, str(path))
    with pytest.raises(InvalidInputError, match=r"voxel spacing.*affine"):
        medmlx.run(image_path=path, weights_path=placeholder_weights, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", [1, 2])
def test_run_rejects_sheared_affine(
    tmp_path: Path, placeholder_weights: Path, version: int
) -> None:
    """The product of header spacings is not voxel volume on a sheared grid."""
    data = np.random.default_rng(9).random((12, 13, 14)).astype(np.float32)
    affine = np.eye(4)
    affine[0, 1] = 0.2
    image = (nib.Nifti1Image if version == 1 else nib.Nifti2Image)(data, affine)
    image.header["xyzt_units"] = 2
    path = tmp_path / "image.nii.gz"
    nib.save(image, str(path))
    with pytest.raises(InvalidInputError, match="sheared"):
        medmlx.run(image_path=path, weights_path=placeholder_weights, output_dir=tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_run_rejects_invalid_arguments_and_existing_outputs(
    tmp_path: Path, placeholder_weights: Path
) -> None:
    """Reject invalid runner options and refuse existing outputs."""
    image = valid_image(tmp_path / "image.nii.gz")
    arguments: RunPaths = {
        "image_path": image,
        "weights_path": placeholder_weights,
        "output_dir": tmp_path / "out",
    }
    with pytest.raises(InvalidInputError, match="does not exist"):
        medmlx.run(**cast(RunPaths, {**arguments, "image_path": tmp_path / "missing.nii.gz"}))
    mgz = tmp_path / "image.mgz"
    mgz.write_bytes(b"")
    with pytest.raises(InvalidInputError, match=r"\.nii or \.nii\.gz"):
        medmlx.run(**cast(RunPaths, {**arguments, "image_path": mgz}))
    with pytest.raises(InvalidInputError, match="ct must be true or false"):
        medmlx.run(**arguments, ct=cast(bool, "yes"))
    with pytest.raises(InvalidInputError, match="overwrite must be true or false"):
        medmlx.run(**arguments, overwrite=cast(bool, 1))
    (tmp_path / "file").write_text("")
    with pytest.raises(InvalidInputError, match="not a directory"):
        medmlx.run(**cast(RunPaths, {**arguments, "output_dir": tmp_path / "file"}))
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "mask.nii.gz").write_bytes(b"previous")
    with pytest.raises(InvalidInputError, match=r"already exist.*mask\.nii\.gz.*overwrite"):
        medmlx.run(**arguments)
    assert (tmp_path / "out" / "mask.nii.gz").read_bytes() == b"previous"


@pytest.mark.parametrize("alias_kind", ["direct", "symlink", "hardlink"])
def test_run_rejects_input_output_aliases_before_replacing_files(
    tmp_path: Path, placeholder_weights: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str
) -> None:
    """Reject output paths that would overwrite the source image."""
    output = tmp_path / "out"
    output.mkdir()
    alias = output / "resampled_input.nii.gz"
    source = valid_image(alias if alias_kind == "direct" else tmp_path / "image.nii.gz")
    if alias_kind == "symlink":
        alias.symlink_to(source)
    elif alias_kind == "hardlink":
        alias.hardlink_to(source)
    original = source.read_bytes()
    previous_mask = output / "mask.nii.gz"
    previous_mask.write_bytes(b"previous mask")
    unrelated = output / "notes.txt"
    unrelated.write_text("keep")

    def unexpected_inference(*_args: object, **_kwargs: object) -> None:
        """Fail if inference begins before an alias is rejected."""
        pytest.fail("An input/output alias must be rejected before inference")

    monkeypatch.setattr(medmlx, "_execute", unexpected_inference)
    with pytest.raises(InvalidInputError, match="input"):
        medmlx.run(
            image_path=source, weights_path=placeholder_weights, output_dir=output, overwrite=True
        )
    assert source.read_bytes() == original
    assert alias.samefile(source)
    assert previous_mask.read_bytes() == b"previous mask"
    assert unrelated.read_text() == "keep"


def test_run_requires_every_weight_file(tmp_path: Path, placeholder_weights: Path) -> None:
    """Report missing checkpoints required by the selected variant."""
    image = valid_image(tmp_path / "image.nii.gz")
    with pytest.raises(AssetNotReadyError, match="weights directory not found") as missing:
        medmlx.run(image_path=image, weights_path=tmp_path / "absent", output_dir=tmp_path / "out")
    assert missing.value.hint and "weights_path" in missing.value.hint
    (placeholder_weights / "synthseg_parc_2.0.safetensors").unlink()
    (placeholder_weights / "synthseg_qc_2.0.safetensors").unlink()
    with pytest.raises(
        AssetNotReadyError,
        match=r"missing synthseg_parc_2\.0\.safetensors, synthseg_qc_2\.0\.safetensors",
    ) as partial:
        medmlx.run(image_path=image, weights_path=placeholder_weights, output_dir=tmp_path / "out")
    assert partial.value.reason == (
        "missing synthseg_parc_2.0.safetensors, synthseg_qc_2.0.safetensors"
    )


def test_declaration_import_loads_no_mlx_or_reference_stack() -> None:
    """Check model discovery remains lazy and avoids reference frameworks."""
    code = (
        "import sys\n"
        "import synthseg_mlx.medmlx as m\n"
        "from synthseg_mlx import load_label_data, postprocess_arrays, qc_scores\n"
        "assert 14 in load_label_data(v1=True)['segmentation']\n"
        "assert m.MODEL['run'] == 'synthseg_mlx.medmlx:run'\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "forbidden = {'mlx', 'torch', 'monai', 'tensorflow', 'keras', 'tf_keras'}\n"
        "assert not loaded & forbidden, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@given(
    values=st.lists(st.integers(-32000, 32000), min_size=8, max_size=8),
    spacing=st.floats(0.125, 4, allow_nan=False, allow_infinity=False, width=32),
    compressed=st.booleans(),
)
def test_nifti_round_trip_preserves_values_and_geometry(
    values: list[int],
    spacing: float,
    compressed: bool,
) -> None:
    """Round-trip NIfTI voxels and physical geometry."""
    data = np.array(values, dtype=np.int16).reshape(2, 2, 2)
    affine = np.diag([spacing, spacing, spacing, 1.0])
    affine[:3, 3] = [3, -2, 7]
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / ("image.nii.gz" if compressed else "image.nii")
        package_utils.save_volume(data, affine, None, str(path), dtype="int16")
        loaded, loaded_affine, _ = package_utils.load_volume(str(path), im_only=False)
        np.testing.assert_array_equal(loaded, data)
        # NIfTI stores affine coefficients as float32.
        np.testing.assert_allclose(loaded_affine, affine, rtol=1e-7, atol=1e-7)
        if not compressed:
            copied = Path(temporary) / "copy.nii.gz"
            medmlx.copy_nifti(path, copied)
            round_trip = nib.load(str(copied))
            np.testing.assert_array_equal(cast(RealArray, np.asanyarray(round_trip.dataobj)), data)
            np.testing.assert_allclose(round_trip.affine, affine, rtol=1e-7, atol=1e-7)


def test_csv_reports_unique_labels_and_requested_column_order(tmp_path: Path) -> None:
    """Check exported volume columns follow unique labels and requested ordering."""
    path = tmp_path / "volumes.csv"
    labels = np.array([3, 0, 42, 3, 100])
    names = np.array(["left cortex", "background", "right cortex", "duplicate", "total"])
    package_labels.write_csv(str(path), None, True, labels, names, last_first=True)
    package_labels.write_csv(str(path), ["subject-a", "9.0", "3.0", "6.0"], True, labels, names)
    with path.open(newline="") as stream:
        assert list(csv.reader(stream)) == [
            ["subject", "total", "left cortex", "right cortex"],
            ["subject-a", "9.0", "3.0", "6.0"],
        ]
    with pytest.raises(ValueError, match="initialising a unique volume file"):
        package_labels.write_csv(str(path), None, False, labels, names)
