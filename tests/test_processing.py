"""Array processing against pinned upstream recordings made on macOS arm64."""

from typing import cast

import mlx.core as mx
import numpy as np
import pytest
from recording_types import Recording
from variant_fixtures import assert_reporting_units

from synthseg_mlx import load_label_data, postprocess, preprocess
from synthseg_mlx._typing import (
    DoubleArray,
    IndexArray,
    LabelArray,
    NameArray,
    RealArray,
)
from synthseg_mlx.label_data import LABEL_FILES
from synthseg_mlx.mlx_postprocess import postprocess_mlx

type Postprocessed = tuple[LabelArray, RealArray | None, RealArray]


@pytest.mark.parametrize(
    "name", ["resample", "orientation", "crop_ct", "constant", "downsample", "multichannel"]
)
def test_preprocess_matches_upstream(reference_fixtures: dict[str, Recording], name: str) -> None:
    """Compare preprocessing and geometry with recorded upstream cases."""
    fixture = reference_fixtures[f"preprocess_{name}"]
    captured: dict[str, RealArray] = {}

    def on_resample(image: RealArray, affine: DoubleArray) -> None:
        """Capture resampled data and affine from preprocessing."""
        captured.update(resampled_image=image.copy(), resampled_affine=affine.copy())

    actual = preprocess(
        fixture["input"],
        fixture["affine"],
        fixture["image_resolution"],
        bool(fixture["ct"]),
        target_res=fixture["target_res"],
        n_levels=int(fixture["n_levels"]),
        crop=cast(IndexArray, fixture["crop"]) if fixture["crop"].size else None,
        min_pad=int(fixture["min_pad"]),
        on_resample=on_resample,
    )
    for field, value in zip(
        ("image", "affine", "resolution", "shape", "pad_idx", "crop_idx"), actual, strict=True
    ):
        expected = cast(RealArray, fixture[f"expected_{field}"])
        if field == "crop_idx" and expected.size == 0:
            assert value is None
        elif field in ("shape", "pad_idx", "crop_idx"):
            np.testing.assert_array_equal(cast(RealArray, value), expected)
        else:
            np.testing.assert_allclose(
                cast(RealArray, value), expected, rtol=1e-6, atol=1e-6, err_msg=field
            )
    expected_resampled = {
        key: value for key, value in fixture.items() if key.startswith("resampled_")
    }
    assert captured.keys() == expected_resampled.keys()
    for field, value in captured.items():
        # The upstream NIfTI saves float32; the callback receives float64 samples.
        np.testing.assert_allclose(
            value, cast(RealArray, expected_resampled[field]), rtol=1e-6, atol=1e-5
        )


@pytest.mark.parametrize("name", ["standard", "crop", "fast", "no_topology", "v1_no_parc"])
@pytest.mark.parametrize("return_posteriors", [True, False])
@pytest.mark.parametrize("gpu", [False, True], ids=["numpy", "mlx"])
def test_postprocess_matches_upstream(
    reference_fixtures: dict[str, Recording], name: str, return_posteriors: bool, gpu: bool
) -> None:
    """Compare the NumPy and runner (MLX) postprocessing with recorded source outputs."""
    fixture = reference_fixtures[f"postprocess_{name}"]
    segmentation = fixture["segmentation"]
    has_parc = bool(fixture["has_parcellation"])
    parcellation = fixture["parcellation"] if has_parc else None
    crop_idx = cast(IndexArray, fixture["crop_idx"]) if fixture["crop_idx"].size else None
    classes = cast(IndexArray, fixture["topology_classes"])
    topology = classes if classes.size else None
    shape = cast(list[int], fixture["shape"].tolist())

    def run(gpu: bool, volume_dtype: type[np.float64] | None) -> Postprocessed:
        """Postprocess fresh copies with the NumPy reference or the runner's MLX path."""
        common = (fixture["labels_segmentation"], fixture["labels_parcellation"], fixture["aff"])
        flags = (fixture["im_res"], bool(fixture["fast"]), topology, bool(fixture["v1"]))
        if gpu:
            parc = None if parcellation is None else mx.array(parcellation)
            return postprocess_mlx(
                mx.array(segmentation), parc, shape, fixture["pad_idx"], crop_idx, *common,
                *flags, return_posteriors=return_posteriors,
            )
        parc_host = None if parcellation is None else parcellation.copy()
        return postprocess(
            segmentation.copy(), parc_host, shape, fixture["pad_idx"], crop_idx, *common,
            *flags, volume_dtype=volume_dtype, return_posteriors=return_posteriors,
        )

    mask, posteriors, volumes = run(gpu, None)
    # Upstream accumulates uncropped non-Standard volumes in float32. The MLX sums are
    # exact to float64 level, so they are held to the reference's float64 result.
    expected_volumes = run(False, np.float64)[2] if gpu else fixture["expected_volumes"]
    np.testing.assert_array_equal(mask, fixture["expected_mask"])
    if return_posteriors:
        np.testing.assert_allclose(
            cast(RealArray, posteriors), fixture["expected_posteriors"], rtol=1e-6, atol=1e-7
        )
    else:
        assert posteriors is None
    assert_reporting_units(volumes, expected_volumes, decimals=3)
    assert mask.dtype == np.int32
    assert mask[0, 0, 0] == 0  # The disconnected island is removed in every mode.
    island = (12, 9, 10) if name == "crop" else (10, 6, 6)
    assert (mask[island] == 0) == (name in ("standard", "crop", "v1_no_parc"))


@pytest.mark.parametrize(("v1", "robust"), [(False, False), (False, True), (True, False)])
def test_bundled_label_data_matches_upstream(
    reference_fixtures: dict[str, Recording],
    v1: bool,
    robust: bool,
) -> None:
    """Compare bundled label arrays with the upstream recording."""
    fixture = reference_fixtures["labels"]
    data = load_label_data(v1=v1, robust=robust)
    assert set(data) == set(LABEL_FILES) | ({"denoiser"} if robust else set())
    for field, array in data.items():
        name = "synthseg_denoiser_labels_2.0.npy" if field == "denoiser" else LABEL_FILES[field]
        if v1:
            name = name.replace("_2.0.npy", ".npy")
        np.testing.assert_array_equal(
            cast(RealArray | NameArray, array),
            cast(RealArray | NameArray, fixture[name.removesuffix(".npy")]),
        )


@pytest.mark.parametrize("v1", [False, True])
def test_declared_label_names_match_upstream(
    reference_fixtures: dict[str, Recording],
    v1: bool,
) -> None:
    """Compare declared label names with source values."""
    from synthseg_mlx.medmlx import label_names

    fixture, expected = reference_fixtures["labels"], {}
    expected: dict[int, str]
    for prefix in ("segmentation", "parcellation"):
        suffix = "_2.0" if prefix == "segmentation" and not v1 else ""
        values = cast(IndexArray, fixture[f"synthseg_{prefix}_labels{suffix}"])
        names = cast(NameArray, fixture[f"synthseg_{prefix}_names{suffix}"])
        expected.update({int(value): str(name) for value, name in zip(values, names, strict=True)})
    assert label_names(v1=v1) == expected
