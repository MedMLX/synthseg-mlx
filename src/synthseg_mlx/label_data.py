"""Bundled standard assets and upstream robust/v1 label arrays."""

from importlib.resources import files
from typing import Literal, cast, overload

import numpy as np

from synthseg_mlx._typing import HostArray, LabelData, RobustLabelData

LABEL_FILES = {
    "segmentation": "synthseg_segmentation_labels_2.0.npy",
    "segmentation_names": "synthseg_segmentation_names_2.0.npy",
    "topology_classes": "synthseg_topological_classes_2.0.npy",
    "parcellation": "synthseg_parcellation_labels.npy",
    "parcellation_names": "synthseg_parcellation_names.npy",
    "qc": "synthseg_qc_labels_2.0.npy",
    "qc_names": "synthseg_qc_names_2.0.npy",
}


@overload
def load_label_data(*, v1: bool = False, robust: Literal[True]) -> RobustLabelData:
    """Load the robust label assets, including denoiser labels."""
    ...


@overload
def load_label_data(*, v1: bool = False, robust: bool = False) -> LabelData | RobustLabelData:
    """Load label assets for the selected standard or robust variant."""
    ...


def load_label_data(*, v1: bool = False, robust: bool = False) -> LabelData | RobustLabelData:
    """Load source arrays in their original order, including duplicates.

    The default returns the original seven arrays. V1 selects its segmentation,
    name, topology and QC assets; robust adds denoiser labels. Selection, sorting
    and left/right correspondence remain with the caller's orchestration.

    Keys: ``segmentation`` (FreeSurfer label values; left/right-neutral labels first,
    then left, then right structures, with repeated values, so callers take
    ``np.unique(..., return_index=True)``), ``segmentation_names`` and
    ``topology_classes`` (one entry per ``segmentation`` entry), ``parcellation`` and
    ``parcellation_names`` (Desikan-Killiany codes 1001-1035 and 2001-2035 after a
    leading background), ``qc`` and ``qc_names`` (QC class per ``segmentation`` entry),
    and ``denoiser`` (robust only: tissue group 0-4 per ``segmentation`` entry).

    Raises:
        InvalidInputError: If ``v1`` and ``robust`` are both set.
    """

    from synthseg_mlx.variants import parse_variant

    variant = parse_variant(v1=v1, robust=robust)
    selected = {
        key: name.replace("_2.0.npy", ".npy") if variant.v1 else name
        for key, name in LABEL_FILES.items()
    }
    if variant.robust:
        selected["denoiser"] = "synthseg_denoiser_labels_2.0.npy"
    root = files("synthseg_mlx").joinpath("data")
    arrays: dict[str, HostArray] = {}
    for key, name in selected.items():
        with root.joinpath(name).open("rb") as payload:
            arrays[key] = cast(HostArray, np.load(payload, allow_pickle=False))
    return cast(LabelData | RobustLabelData, arrays)
