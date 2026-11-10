"""Variant labels and upstream postprocessing without MLX."""

from typing import cast

import numpy as np
import pytest
from variant_fixtures import assert_reporting_units, load_variant

from synthseg_mlx import medmlx, postprocess_arrays
from synthseg_mlx._typing import IndexArray, RealArray
from synthseg_mlx.variants import SynthSegVariant


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
    original_segmentation = fixture["segmentation"].copy()
    original_parcellation = fixture["parcellation"].copy()
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
    np.testing.assert_array_equal(fixture["segmentation"], original_segmentation)
    np.testing.assert_array_equal(fixture["parcellation"], original_parcellation)
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
