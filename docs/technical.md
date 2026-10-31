# synthseg-mlx technical notes

Full details behind the [README](../README.md).

## Overview

Native float32 MLX implementation of [SynthSeg](https://github.com/BBillot/SynthSeg),
part of [MedMLX](https://github.com/MedMLX). Runs without Torch, MONAI,
TensorFlow, or Keras. [MedMLX's segment extra](https://github.com/MedMLX/medmlx#install) installs it
with the other segmentation models.

**Status:** The standard model retains the extracted original arithmetic. Robust,
fast, v1 and v1+fast implement the prediction branches in upstream SynthSeg
revision [`2a2aa3b`](https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/SynthSeg/predict_synthseg.py).
References for all five graphs, array processing and reduced runners are recorded
from upstream TensorFlow/Keras and NumPy on macOS arm64. Tests use numerical
bounds for probabilities and reported values, and exact comparisons for label maps.
Standard volumes use the fork's float64 accumulation policy. Their reporting
oracle replays the original fork over the unchanged recorded TensorFlow
probabilities; the upstream float32 volume recording remains available for the
other variants. The one-unit reporting bound is unchanged. Real checkpoints,
full-size imaging and clinical performance remain unqualified.

Supported environment: **Apple Silicon only — macOS arm64 with MLX on Metal**.
Metal must be available for inference.

## Install

Runtime dependencies are `medmlx-core`, MLX, NumPy, SciPy, and nibabel (NIfTI
I/O for the medmlx runner). Native safetensors weights need no conversion. Runtime
selection belongs to the shared core; this package supports Apple Silicon Metal.

## Use with medmlx

The package declares the model `synthseg` in the `medmlx.models` entry-point group
(`synthseg_mlx.medmlx:MODEL`), so [medmlx](https://github.com/MedMLX/medmlx)
serves it once both are installed in one environment. The runner is
`synthseg_mlx.medmlx.run`, an MLX implementation of the SynthSeg route: SynthSeg
2.0 with cortical parcellation and QC. Standard remains the default; variants are
explicit keyword arguments.

| Argument | Meaning |
|---|---|
| `image_path` | One 3D brain MRI `.nii`/`.nii.gz` of any contrast and resolution, or CT in Hounsfield units with `ct=true`. |
| `weights_path` | Optional model folder holding the three native safetensors files listed below. Omit for automatic download from `MedMLX/synthseg-mlx`. |
| `output_dir` | Output directory, created if missing. |
| `ct` | `false` (default) for MRI; `true` clips CT intensities to [0, 80] before normalization. |
| `robust` | `false` (default); `true` selects `synthseg_robust_2.0` and the S1 → denoiser → S2 cascade. Implies `fast=true`; cannot combine with `v1`. |
| `fast` | `false` (default); `true` skips left/right averaging and topological-class cleanup. Uses the standard checkpoint unless `robust` or `v1` is selected. |
| `v1` | `false` (default); `true` selects `synthseg_1.0`, 1.0 label/name/QC/topology arrays and 18 neutral labels. Compatible with `fast`; omits total intracranial volume. |
| `crop` | `null` (default); an `int`, a one-element sequence or three per-axis `int` values in RAS order, rounded up to multiples of 32, bounding the processed patch size. When set, it replaces the default 128-voxel minimum pad, matching upstream's `--crop` (see "fixed settings" below). |
| `parc` | `true` (default); `false` turns off cortical parcellation in the mask and `volumes.csv`, matching upstream's `--parc` (upstream defaults this off; this runner defaults it on for established behavior). The parcellation network still runs either way; only the written labels/volumes differ. QC uses the same segmentation output for both settings, while upstream's QC path can differ with `do_parcellation`. |
| `save_posteriors` | `false` (default); `true` additionally writes `posteriors.nii.gz`, matching upstream's `--post`. |
| `overwrite` | `false` (default) refuses to replace existing outputs. |
| `arithmetic_profile` | `"source-fp32"` (default) keeps each variant's arithmetic; `"compensated-fp32"` selects compensated Metal operators for the standard 2.0 segmentation/parcellation graphs and smooths each prediction before averaging. QC retains its usual arithmetic. Other variants reject this profile. |

The runner admits finite, nonconstant 3D images with at least three voxels per
axis. Current source requires NIfTI spatial units of mm or unspecified (assumed mm),
and header voxel spacing must match the selected affine's axis lengths within
relative tolerance `1e-5` plus absolute tolerance `1e-6` mm. This prevents the
header-based resampling and affine geometry from describing different physical
scales. Axes must also be orthogonal (rotation is supported): source volume reporting
multiplies the axis lengths, which is not voxel volume on a sheared grid. The source dimension heuristic rejects a final
axis of three voxels.
It warns for four through ten, and padding can fail if that axis remains ten or
smaller after preprocessing.

Outputs (roles in `outputs`, all absolute paths):

| Role | File | Content |
|---|---|---|
| `mask` | `mask.nii.gz` | int32 FreeSurfer labels; cortex replaced by Desikan-Killiany parcels (1001-1035, 2001-2035), or plain cortex labels (3, 42) when `parc=false`. |
| `resampled_input` | `resampled_input.nii.gz` | The input on the mask's grid; a copy of the input when no resampling was needed. |
| `volumes` | `volumes.csv` | Soft (posterior) volumes in mm³; total intracranial is the first volume column for 2.0, and only that column is omitted for v1. |
| `qc_scores` | `qc.csv` | Eight regional QC scores in [0, 1]. |
| `posteriors` | `posteriors.nii.gz` | float32 per-label posterior probabilities on the mask's grid, summing to 1 per voxel; only written when `save_posteriors=true`. |

An input whose voxel sizes differ from 1 mm by more than 0.05 mm is resampled
to 1 mm, and the mask shares its
shape and affine with `resampled_input.nii.gz`. Otherwise the mask keeps the input's
shape and affine. `details` holds `labels` (every value/name), `voxel_counts` (by
label name), `shape`, `dtype`, `affine`, `grid` (`input` or `synthseg_1mm`),
`volumes_mm3`, `qc_scores`, `ct`, `variant`, effective `robust`/`fast`/`v1`, `parc`,
`runtime` (MLX version and device, arithmetic profile, stage timings,
peak memory) and `duration_seconds`. Output requires human review and is not ground
truth.

Omitted weights download automatically from `MedMLX/synthseg-mlx`. To override
them, supply a local native safetensors folder as `weights_path`; see the
[README](../README.md#local-weights) for an explicit download.
For `robust` or `v1`, use the corresponding segmentation file instead.

| Variant | Segmentation checkpoint |
|---|---|
| Standard or fast | `synthseg_2.0` |
| Robust | `synthseg_robust_2.0` |
| V1 or v1+fast | `synthseg_1.0` |

Append `.safetensors` to the names above. Every variant also requires
`synthseg_parc_2.0` and `synthseg_qc_2.0` in the same format, even with
`parc=false`. The label/name/topology arrays are bundled with the package.
Loading validates the expected parameters, tensor shapes and finite float32 values.
Missing checkpoints or incompatible checkpoint contents raise
`AssetNotReadyError`. Input admission rejects invalid images, options, or existing
outputs with `InvalidInputError`; source processing can also raise other errors,
including the small-volume padding error described above.

```python
from synthseg_mlx.medmlx import run

result = run(
    image_path="brain.nii.gz", weights_path="SynthSeg_models", output_dir="out",
    robust=True,
)
```

The robust cascade uses five tissue groups: S1 has 24 base features and 3³
kernels; the denoiser has 16 base features, 5³ kernels and omits its final two
decoder skip concatenations; S2 takes the original image plus hard denoised group
channels and has 24 base features and 3³ kernels. Both transitions use
argmax → one-hot. Robust skips segmentation smoothing and left/right averaging.
Fast non-robust still smooths the cortex-conditioning prediction, while its
returned segmentation and QC use raw probabilities. Parcellation and QC run in
every variant; with both enabled upstream returns raw parcellation probabilities.
Fast retains the global largest-component filter, 0.2 posterior threshold and
normalization; it skips only the per-topological-class cleanup. Preprocessing is
the same in all variants. These choices follow upstream's prediction graph and
[CLI rules](https://github.com/BBillot/SynthSeg/blob/2a2aa3bbfccb83f8253a51ca8b329b9938a2646d/scripts/commands/SynthSeg_predict.py).

Compatibility decision: the default keeps the original blur-after-averaging arithmetic
and float64 volume/QC accumulation, preserving established behavior.
The other variants use upstream float32 accumulation/rounding; slow v1 and the
explicit `compensated-fp32` standard profile blur each prediction before averaging.
The default does not adopt upstream's operation order
because that would change the established numerical contract. V1 uses
32 unique segmentation labels rather than 33 and omits the CSF label 24; the
parcellation and QC checkpoint files remain the 2.0 files used by the upstream CLI.

The fixed settings are: inputs are padded to at least 128 voxels
per axis (unless `crop` is given, which sets that minimum instead, matching
upstream's `--crop`) and QC runs on a 224³ crop on Metal.

## Public API

Callers load the image and retain its header. Pass the loader's squeezed array,
original dtype, affine, and voxel sizes without recasting; the pinned source's
NIfTI loader produces float64 arrays. The example assumes `loaded_image`,
`affine`, `image_resolution`, and a `Path` named `models_dir` were supplied by the caller.

```python
import numpy as np
from synthseg_mlx import (
    SynthSegModel, SynthSegVariant, get_flip_indices, load_label_data, postprocess_arrays,
    predict_numpy, preprocess, qc_scores,
)

variant = SynthSegVariant.STANDARD  # FAST, ROBUST, V1 or V1_FAST are also available.
options = dict(robust=variant.robust, fast=variant.fast, v1=variant.v1)
assets = load_label_data(v1=variant.v1, robust=variant.robust)
if variant.fast:
    labels_segmentation, unique_idx = np.unique(assets["segmentation"], return_index=True)
    flip_indices = None
else:
    labels_segmentation, flip_indices, unique_idx = get_flip_indices(
        assets["segmentation"], variant.neutral_labels,
    )
labels_parcellation = np.unique(assets["parcellation"])
labels_qc = assets["qc"][unique_idx]
topology_classes = assets["topology_classes"][unique_idx]

model = SynthSegModel.from_safetensors(
    models_dir / variant.segmentation_checkpoint, labels_segmentation,
    models_dir / "synthseg_parc_2.0.safetensors", labels_parcellation,
    models_dir / "synthseg_qc_2.0.safetensors", labels_qc, flip_indices,
    labels_denoiser=np.unique(assets["denoiser"]) if variant.robust else None,
    **options,
)
image, affine, image_resolution, shape, pad_idx, crop_idx = preprocess(
    loaded_image, affine, image_resolution, ct=False, crop=None, min_pad=128,
)
segmentation, parcellation, raw_qc = predict_numpy(model, image)
label_map, posteriors, volumes = postprocess_arrays(
    segmentation, parcellation, shape=shape, pad_idx=pad_idx, crop_idx=crop_idx,
    labels_segmentation=labels_segmentation,
    labels_parcellation=labels_parcellation, affine=affine,
    image_resolution=image_resolution, topology_classes=topology_classes,
    **options,
)
quality = qc_scores(raw_qc, **options)
```

`preprocess(image, affine, image_resolution, ct, target_res=1., n_levels=5,
crop=None, min_pad=None, *, on_resample=None)` retains source resampling (including
its 0.05 mm threshold), RAS alignment, center cropping, CT clipping, percentile
normalization, padding, and batch/channel insertion. Supply `min_pad`, as the
medmlx runner does. It returns six values; the caller keeps the header separately.
The optional `on_resample(image, affine)` hook runs only when resampling occurs, before
alignment/normalization, so callers can save the resampled input. The hook must not
modify these arrays. These array functions read and write no image files; only the
medmlx runner does.

`predict_numpy` consumes a padded, normalized NDHWC single-channel volume. The
original five-level networks require spatial dimensions divisible by 16; source
preprocessing rounds to multiples of 32 by default. It preserves contiguous
float32 preparation, forward evaluation, and the explicit NumPy-copy boundary.
`model(mlx_image)` exposes the MLX forward interface. `qc_scores` excludes
background, clips to `[0, 1]`, and rounds to four places. Pass the same variant
flags to loading, model construction, postprocessing and QC formatting.

`postprocess_arrays` accepts `robust=False`, `fast=False`, `v1=False`, and calls
`postprocess` with `reuse_segmentation=True`, `return_posteriors=False`; these two
settings are not wrapper arguments.
No caller-supplied postprocessor is required. It copies the posterior inputs and
returns `(label_map, posteriors, volumes)` unchanged. The fork's NumPy branch
returns posteriors even when `return_posteriors=False`; that behavior is preserved.
Use `postprocess` from `synthseg_mlx` or `synthseg_mlx.processing` for the original
source parameter names (`post_patch_seg`, `post_patch_parc`, `aff`, `im_res`) and
other modes. Its NumPy parcellation path mutates its posterior input, as upstream
does; pass a copy if ownership must be retained. Framework tensor inputs are
unsupported.

`load_label_data()` returns `segmentation`, `segmentation_names`,
`topology_classes`, `parcellation`, `parcellation_names`, `qc`, and `qc_names` in
original source order. `v1=True` substitutes the five upstream 1.0 arrays;
`robust=True` adds `denoiser`. Names and topology/QC arrays must use the same selection
indices as the labels. `labels_qc` maps each segmentation channel to a QC class;
its unique class indices must be contiguous starting at zero. Other public names retain
the original spelling: `UNet`,
`QCNet`, `BatchNorm3d`, `CheckpointLayers`, `crop_for_qc`,
and `prepare_network_input`; `SynthSegVariant` exposes the canonical variant
choices.

## Extraction and integration

Paths below refer to the SynthSeg fork at revision `d6a64db` (see
`THIRD_PARTY_NOTICES.md`). Counts are full source file →
package file; preprocessing/helpers are extracted subsets, not full source-file
copies. These adapted subsets retain their original notices and numerical bodies;
they are not byte-identical upstream files. The graph and runner also include the
upstream variant branches described
above; `variants.py` owns their choices, and the six added denoiser/v1 assets come
from that upstream revision.

| Source path | Package path | Lines, source → package |
|---|---|---:|
| SynthSeg `SynthSeg/predict_synthseg.py` | `src/synthseg_mlx/processing.py` | 846 → 275 |
| SynthSeg `SynthSeg/labels.py` (`get_flip_indices`, `write_csv`) | `src/synthseg_mlx/labels.py` | 94 → 126 |
| SynthSeg `ext/lab2im/utils.py` (adds `load_volume`, `save_volume`, `get_volume_info`, `get_list_labels` without its folder branch, `mkdir`) | `src/synthseg_mlx/_lab2im/utils.py` | 1064 → 486 |
| SynthSeg `ext/lab2im/edit_volumes.py` | `src/synthseg_mlx/_lab2im/edit_volumes.py` | 2850 → 432 |
| SynthSeg `data/labels_classes_priors/synthseg_segmentation_labels_2.0.npy` | `src/synthseg_mlx/data/synthseg_segmentation_labels_2.0.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_segmentation_names_2.0.npy` | `src/synthseg_mlx/data/synthseg_segmentation_names_2.0.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_topological_classes_2.0.npy` | `src/synthseg_mlx/data/synthseg_topological_classes_2.0.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_parcellation_labels.npy` | `src/synthseg_mlx/data/synthseg_parcellation_labels.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_parcellation_names.npy` | `src/synthseg_mlx/data/synthseg_parcellation_names.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_qc_labels_2.0.npy` | `src/synthseg_mlx/data/synthseg_qc_labels_2.0.npy` | binary |
| SynthSeg `data/labels_classes_priors/synthseg_qc_names_2.0.npy` | `src/synthseg_mlx/data/synthseg_qc_names_2.0.npy` | binary |
| SynthSeg `LICENSE.txt` | `THIRD_PARTY_NOTICES.md` | 200 → 200 |

The native MLX model, checkpoint lookup and execution/runner code
(`mlx_model.py`, `pinned_weights.py`, `mlx_engine.py`, `medmlx.py`) implement
the upstream network structure, checkpoint layer names and operation order in MLX.
The numerical reference recorder executes the pinned TensorFlow/Keras network
builders from `ext/neuron/models.py` and coordinate helpers from
`ext/neuron/utils.py`. Both files retain GPLv3 headers at the reference revision
`2a2aa3b`, while that repository's top-level `LICENSE.txt` is Apache-2.0.
Historical Neurite commit `b8efac0` added an Apache-2.0 license, and `c516287`
changed both corresponding file notices to Apache-2.0 without changing the five
relevant function bodies, apart from documentation. The coordinate helpers
`volshape_to_meshgrid` and `meshgrid` match this explicitly Apache-headered source
exactly in body and signature. The three builders share construction code but
differ in input composition, dropout handling and decoder skip selection. The
Apache-headered SynthSeg prediction code separately declares the fixed standard,
robust, parcellation and QC graphs, including the denoiser's skip selection; its
`MakeShape` and lab2im's `gaussian_kernel` supply the crop and Gaussian rules.
The native model arrived through RadNN `cbaa1ac` in commit `333f57d`; the reference
recorders execute external source definitions without importing the MLX port.
The numerical reference revision remains `2a2aa3b`.
This mapping records code lineage and scientific behavior. It does not determine
whether the native graph copies or translates source expression or independently
implements the architecture; the framework change alone does not settle that
question. See the source-specific distinctions in `THIRD_PARTY_NOTICES.md`.
The standard graph retains its extracted arithmetic; errors and runtime selection
use `medmlx_core`. Variant branches add upstream cascade, decoder skip selection,
smoothing/flip order and label selection.
Its NDHWC/Keras-specific repeat, Gaussian and kernel-layout math remain local.
Preprocessing/postprocessing retain the pinned fork's numerical statements; image
I/O, imports and framework adapters are local. Variant sums explicitly select
upstream dtypes. NumPy 2 requires explicit `.item()` extraction for the pinned
fork's one-element integer assignments in orientation and label correspondence.
Those selections preserve the original integer values; the default's
floating-point operations, reductions and dtypes are unchanged. The upstream
reference recorder makes these isolated NumPy compatibility adaptations as well.

The medmlx runner implements MLX `execute`, `select_labels`, input admission, output
checks and label names, plus the NIfTI/CSV helpers they call. It writes
`runtime.json`'s facts into the returned `details` instead of files. Training,
TensorFlow/Keras graph construction and unrelated lab2im helpers remain in
SynthSeg. No SynthSeg or core files were modified.

## Smoothing performance

Gaussian smoothing uses a channel-wise Metal stencil instead of a dense
convolution with zero cross-channel weights. It retains the same float32 Gaussian
coefficients, spatial tap order and zero padding, with 64-bit buffer offsets.
There are no separable passes or dtype changes.

Measurements on Apple M1 Max,
MLX 0.32.3, nine alternating repetitions after three warmups per path:

| Workload | Dense median | Metal median | Speedup |
|---|---:|---:|---:|
| 128³ smoothing, 33 channels | 399.81 ms | 18.39 ms | 21.74× |
| 64³ smoothing, 69 channels | 175.73 ms | 4.63 ms | 37.98× |
| Standard `predict_numpy`, 128³, 24 features, 224³ QC | 5997.58 ms | 6003.22 ms | Flat/noisy |

All compared outputs are bitwise identical. Peak MLX memory for the 128³
smoothing case falls from 1239.11 to 710.37 MB, including resident model/input
arrays; the complete inference case falls from 11982.96 to 11857.13 MB.
The inference comparison uses seeded weights at the default network widths,
with all 33 segmentation and 69 parcellation channels. It includes all three
networks and host copies, excluding preprocessing, postprocessing and file I/O.
This comparison predates the QC optimization below. QC convolutions dominated
total runtime; these results establish a smoothing improvement, with no measured
improvement in complete inference latency.
Pretrained-checkpoint performance is unmeasured.

Reproduce on an idle Apple Silicon Mac:

```bash
uv run --locked --extra dev python scripts/benchmark_smoothing.py \
  --repeats 9 --output /tmp/synthseg-gaussian-metal.json
```

## QC performance

QC reads class indices directly for its first two convolutions, avoiding a large
one-hot volume and products with its zero entries. Coefficients are packed with
adjacent output channels together for contiguous Metal reads. Residual addition,
ReLU, frozen batch normalization and 2³ max pooling share one kernel, preserving
each float32 rounding step and handling negative normalization weights.

Measurements on Apple M1 Max,
MLX 0.32.3, nine alternating repetitions after three warmups per path:

| Workload | Original QC median | Optimized median | Speedup |
|---|---:|---:|---:|
| QC scoring from 224³ class indices, 24 features | 3203.04 ms | 1910.06 ms | 1.68× |
| Standard `predict_numpy`, 128³, 24 features, 224³ QC | 4661.79 ms | 4069.28 ms | 1.15× |

Complete prediction has **12.71% lower median latency** and peak MLX memory falls
from **11.86 to 9.02 GB** (23.94%). All segmentation, parcellation and QC outputs
are bitwise identical. Full-inference samples have overlapping ranges; the claim
describes the measured medians, with p95 improving from 4.77 to 4.33 seconds.
The baseline and optimized paths share the Gaussian optimization and seeded
weights, with all 33 segmentation and 69 parcellation channels. The full case
includes every network and host copies, excluding preprocessing, postprocessing
and file I/O. Pretrained-checkpoint performance remains unmeasured.

Six additional parity cases cover batches, strided spatial axes, nontrivial
normalization statistics, signed normalization weights and 9/16/17 QC classes.
Class counts above 16 retain native convolution accumulation order.

```bash
uv run --locked --extra dev python scripts/benchmark_qc.py \
  --repeats 9 --output /tmp/synthseg-qc-metal.json
```

## Verification

Use Python 3.13 on macOS arm64. Keep environments and caches outside the repo:

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
uv venv "$TMPDIR/venv-synthseg-mlx" --python 3.13
uv pip install --python "$TMPDIR/venv-synthseg-mlx/bin/python" -e ".[dev]"
"$TMPDIR/venv-synthseg-mlx/bin/ruff" check . --no-cache
PYTHONDONTWRITEBYTECODE=1 HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/synthseg-hypothesis" \
  "$TMPDIR/venv-synthseg-mlx/bin/python" -m pytest -q -p no:cacheprovider
```

The full suite requires Metal. There are no backend branches, environment-based
skips or external-checkpoint tests. Array processing, admission, reporting and
file boundaries can be checked without loading MLX:

```bash
PYTHONDONTWRITEBYTECODE=1 HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/synthseg-hypothesis" \
  "$TMPDIR/venv-synthseg-mlx/bin/python" -m pytest -q -p no:cacheprovider \
  tests/test_variants.py tests/test_processing.py tests/test_reporting.py tests/test_medmlx.py
```

Tests load the committed Mac references directly. TensorFlow, Keras and Torch
are not imported at test time.

## Regenerating reference fixtures

Both recorders compare the eight consumed upstream source files and label assets
byte-for-byte against the declared Git revision before executing definitions.
They use local source-checkout objects by default. Archive copies remain supported
with `--source-repository /path/to/SynthSeg` containing that revision and its blobs;
missing objects or changed bytes fail explicitly without fetching. This proves
the admitted source and label assets, not released checkpoint weights. Existing
fixture formats and seeded parameters are unchanged. They never import this port.
Run them on macOS arm64 in a
separate Python 3.12 reference environment; keep source and downloads outside the
repository:

```bash
export UV_CACHE_DIR="$TMPDIR/uv-cache"
uv venv "$TMPDIR/venv-synthseg-reference" --python 3.12
uv pip install --python "$TMPDIR/venv-synthseg-reference/bin/python" \
  tensorflow==2.20.0 tf-keras==2.20.1 numpy==2.3.5 scipy==1.15.3 nibabel==5.4.2
git clone https://github.com/BBillot/SynthSeg "$TMPDIR/synthseg-upstream"
git -C "$TMPDIR/synthseg-upstream" checkout 2a2aa3bbfccb83f8253a51ca8b329b9938a2646d
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$TMPDIR/synthseg-matplotlib" \
  "$TMPDIR/venv-synthseg-reference/bin/python" scripts/make_variant_fixtures.py \
  --synthseg-src "$TMPDIR/synthseg-upstream"
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR="$TMPDIR/synthseg-matplotlib" \
  "$TMPDIR/venv-synthseg-reference/bin/python" scripts/make_processing_fixtures.py \
  --synthseg-src "$TMPDIR/synthseg-upstream"
```

The five `tests/fixtures/variants/` recordings are below 755 KB each (`fast.npz` is
largest: it also carries a `crop_runner_*` recording exercising `--crop` on a
20³ raw image cropped to 16³); the fourteen
`tests/fixtures/processing/` recordings are below 213 KB each. All carry schema
version 2, `reference_host=darwin-arm64`, the upstream revision and executed
source paths. Network recordings also carry framework versions and seeded
weight metadata.

`tests/fixtures/reporting/standard_float64.json` separately records the standard
fork's volume policy. `scripts/make_reporting_fixture.py` executes the original
host functions at `d6a64db`, reads the unchanged standard TensorFlow probabilities,
and verifies identical labels and posteriors before recording float64 volumes.
It imports neither TensorFlow nor this port. Regenerate it in the reference
environment with a Git checkout containing that fork revision:

```bash
python scripts/make_reporting_fixture.py --synthseg-src "$TMPDIR/synthseg-fork"
```

The source reports 2369.274 mm³ in float64, versus 2369.273 in the retained
float32 recording. Native 2369.275 differs from the matching policy by one
reporting unit. Mixing the float64 default with the float32 oracle had combined
accumulation drift with the allowed native probability drift into two units.
The separate oracle preserves both policies and records the input archive,
original source functions and runtime.
This host reporting replay does not regenerate TensorFlow graphs.

TensorFlow runs on this Mac's CPU with one thread as the independent reference;
MLX inference tests require Metal. Graph input and QC crops are 16³; base features
are 2 for segmentation/parcellation/QC and 1 for the denoiser. All five levels,
kernels, skip choices, hard transitions, Gaussian operations and prediction
branches remain. Four foreground parcellation labels are used; segmentation
retains every upstream label. Seed 719 includes a recorded cortical-head bias to
exercise nonempty parcellation. The complete runner tests execute real
preprocessing, H5 loading, inference, postprocessing, NIfTI and CSV output for
every variant. A separate anisotropic, reoriented CT case checks resampling and
replacement of stale outputs.

Probability comparisons retain `rtol=1e-5, atol=1e-6`; label maps and QC crop
indices must match exactly. Volumes and QC may differ by one reporting unit
(0.001 mm³ and 0.0001 respectively), compared in integer decimal units so binary
representation of a decimal boundary cannot falsely fail. A two-unit difference
fails. Processing references cover orientation, resampling, downsampling,
CT clipping/cropping, constant and multichannel input, global component removal,
per-class topology cleanup and its fast-mode omission. NIfTI round trips also
have property-based checks of values and physical geometry.
