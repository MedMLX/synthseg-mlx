"""Variant admission, assets, pins and upstream postprocessing without MLX."""

import inspect
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from medmlx_core.errors import InvalidInputError
from test_medmlx import RunPaths
from variant_fixtures import assert_reporting_units, load_variant

from synthseg_mlx import medmlx, pinned_weights, postprocess_arrays
from synthseg_mlx._typing import IndexArray, RealArray
from synthseg_mlx.variants import SynthSegVariant, parse_variant


@given(robust=st.booleans(), fast=st.booleans(), v1=st.booleans())
def test_flags_parse_and_round_trip(robust: bool, fast: bool, v1: bool) -> None:
    """Check valid variant flags map to and from the canonical variant."""
    if robust and v1:
        with pytest.raises(InvalidInputError, match="v1 cannot be used with robust"):
            parse_variant(robust=robust, fast=fast, v1=v1)
        return
    variant = parse_variant(robust=robust, fast=fast, v1=v1)
    assert parse_variant(robust=variant.robust, fast=variant.fast, v1=variant.v1) is variant
    assert variant.fast is (fast or robust)


@pytest.mark.parametrize("flag", ["robust", "fast", "v1"])
@given(value=st.one_of(st.integers(), st.text(), st.none(), st.floats()))
def test_flags_reject_non_booleans(
    flag: Literal["robust", "fast", "v1"], value: int | str | float | None
) -> None:
    """Reject non-boolean variant flags."""
    with pytest.raises(InvalidInputError, match=f"{flag} must be true or false"):
        parse_variant(**{flag: cast(bool, value)})


@pytest.mark.parametrize("name", ["standard", "fast", "robust", "v1", "v1_fast"])
def test_upstream_labels_and_postprocessing(name: str) -> None:
    """Compare selected variant labels and postprocessing with upstream recordings."""
    fixture = load_variant(name)
    variant = SynthSegVariant(name)
    labels = medmlx.select_labels(variant)
    assert np.array_equal(labels.segmentation, fixture["labels_segmentation"])
    assert np.array_equal(labels.flip_indices, fixture["flip_indices"])
    assert np.array_equal(labels.qc, fixture["labels_qc"])
    assert np.array_equal(labels.topology_classes, fixture["topology_classes"])
    assert ("total intracranial" in labels.volume_names) is (not variant.v1)
    if variant.robust:
        assert np.array_equal(medmlx.denoiser_labels(), fixture["labels_denoiser"])
    actual = postprocess_arrays(
        fixture["segmentation"],
        fixture["parcellation"],
        shape=[16] * 3,
        pad_idx=np.array([0, 0, 0, 16, 16, 16]),
        crop_idx=None,
        labels_segmentation=fixture["labels_segmentation"],
        labels_parcellation=fixture["labels_parcellation"],
        affine=np.eye(4),
        image_resolution=np.ones(3),
        topology_classes=cast(IndexArray, fixture["topology_classes"]),
        robust=variant.robust,
        fast=variant.fast,
        v1=variant.v1,
    )
    for field, value in zip(("mask", "posteriors", "volumes"), actual, strict=True):
        assert cast(RealArray, value).dtype == fixture[field].dtype
        if field == "mask":
            np.testing.assert_array_equal(value, cast(RealArray, fixture[field]))
        elif field == "volumes":
            assert_reporting_units(value, cast(RealArray, fixture[field]), decimals=3)
        else:
            np.testing.assert_allclose(
                cast(RealArray, value), cast(RealArray, fixture[field]), rtol=1e-6, atol=1e-7
            )


@pytest.mark.parametrize("variant", list(SynthSegVariant))
def test_selected_pins_require_only_the_requested_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: SynthSegVariant
) -> None:
    """Check each variant requires its own checkpoint set."""
    contents = {name: name.encode() for name in variant.checkpoints}
    with pytest.raises(ValueError, match=f"missing {variant.segmentation_checkpoint}"):
        pinned_weights.checkpoint_paths(tmp_path, variant)
    for name, value in contents.items():
        (tmp_path / name).write_bytes(value)
    actual = pinned_weights.checkpoint_paths(tmp_path, variant)
    assert actual == {name: tmp_path / name for name in contents}


def test_variant_runner_admission_precedes_image_and_network_loading(tmp_path: Path) -> None:
    """Check variant admission fails before image and network loading."""
    args: RunPaths = {
        "image_path": tmp_path / "image.nii.gz",
        "weights_path": tmp_path,
        "output_dir": tmp_path,
    }
    with pytest.raises(InvalidInputError, match="v1 cannot be used with robust"):
        medmlx.run(**args, robust=True, v1=True)
    for flag in ("robust", "fast", "v1"):
        flag: Literal["robust", "fast", "v1"]
        invalid_options = {"robust": False, "fast": False, "v1": False}
        invalid_options[flag] = cast(bool, 1)
        with pytest.raises(InvalidInputError, match=f"{flag} must be true or false"):
            medmlx.run(
                **args,
                robust=invalid_options["robust"],
                fast=invalid_options["fast"],
                v1=invalid_options["v1"],
            )
        assert inspect.signature(medmlx.run).parameters[flag].default is False
