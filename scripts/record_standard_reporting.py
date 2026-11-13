"""Record pinned original-fork volume reporting on real CT graph outputs."""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import cast

import numpy as np
from record_preprocessing_reference import definitions, save_archive
from scipy.ndimage import find_objects, label

from synthseg_mlx._typing import HostArray

REVISION = "d6a64db19ec7b8b71c64edd59e7e485e0235c26c"


def main() -> None:
    """Execute the fork's postprocessing against recorded TensorFlow outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Upstream Git repository")
    parser.add_argument("--references", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="synthseg-reporting-") as temporary:
        root = Path(temporary)
        for name in (
            "ext/lab2im/utils.py",
            "ext/lab2im/edit_volumes.py",
            "SynthSeg/predict_synthseg.py",
        ):
            contents = subprocess.run(
                ["git", "-C", str(args.source), "show", f"{REVISION}:{name}"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents)
        scope: dict[str, object] = {"np": np, "scipy_label": label, "find_objects": find_objects}
        utils = definitions(root / "ext/lab2im/utils.py", {"get_dims", "reformat_to_list"}, scope)
        scope["utils"] = utils
        edits = definitions(
            root / "ext/lab2im/edit_volumes.py",
            {
                "align_volume_to_ref",
                "crop_volume_with_idx",
                "get_largest_connected_component",
                "get_ras_axes",
            },
            scope,
        )
        scope["edit_volumes"] = edits
        predict = definitions(
            root / "SynthSeg/predict_synthseg.py",
            {"argmax_channels", "largest_component", "postprocess"},
            scope,
        )
        with np.load(args.references / "variants/standard.npz", allow_pickle=False) as data:
            outputs: dict[str, HostArray] = {}
            for kind, prefix in (("graph", ""), ("runner", "runner_")):
                _, _, volumes = predict.postprocess(
                    data[f"{prefix}segmentation"].copy(),
                    data[f"{prefix}parcellation"].copy(),
                    shape=[16] * 3,
                    pad_idx=np.array([0, 0, 0, 16, 16, 16]),
                    crop_idx=None,
                    labels_segmentation=data["labels_segmentation"],
                    labels_parcellation=data["labels_parcellation"],
                    aff=np.eye(4),
                    im_res=np.ones(3),
                    fast=False,
                    topology_classes=data["topology_classes"],
                    v1=False,
                )
                outputs[f"{kind}_volumes"] = cast(HostArray, volumes)
        save_archive(
            args.references / "standard_reporting.npz",
            source_revision=np.asarray(REVISION),
            **outputs,
        )


if __name__ == "__main__":
    main()
