# Technical notes

## GPU postprocessing

The runner postprocesses the network outputs on the GPU (`synthseg_mlx.mlx_postprocess`)
and keeps the posteriors on the device until the outputs are written. Volumes are
accumulated in float64 for every variant. `postprocess_arrays` keeps the NumPy/SciPy
path used by the equivalence tests.

Checked on a real 1 mm T1-weighted brain MRI (192 x 256 x 256) with the released
weights, against the previous host postprocessing: label maps were identical
(0 differing voxels), the volume and QC tables were identical text, and posteriors
differed by at most 1.8e-7, for the standard, robust and v1+fast variants. Peak
resident memory of the process fell from 16.0 GB to 7.6 GB (standard with
parcellation) and from 9.1 GB to 7.6 GB (other variants).

## Peak MLX memory on Apple M1 Max

Measured against `f9e4678` (0.4.0) on an Apple M1 Max with 64 GB of unified
memory, macOS 27.2.1 and MLX 0.32.4. The input was a 160 × 192 × 192 crop of an
acquired 1 mm T1 MRI (`sub-01/ses-mri/anat/sub-01_ses-mri_T1w.nii.gz`,
voxel slices `16:176, 32:224, 32:224`). Runs used `source-fp32` and the
released weights at revision `b4fa99020571cd5921f2205834dd5028fa804046`.
[OpenNeuro ds000117](https://openneuro.org/datasets/ds000117)

At this shape, the baseline reached 9.03 GB on the M1 Max, 22% below the
published 11.6 GB. The original scan identifier was unavailable. The peak
occurred in the QC network, which scores a 224³ grid of segmentation class
indices. Each full-resolution, 24-channel float32 activation takes 1.08 GB.
The bounded predictor now evaluates the main convolution of the first QC
block before the residual branch, then evaluates the pooled result before the
lower-resolution blocks.

Stage profiling used separate synchronized intervals on the standard run with
parcellation. Each interval reset the MLX peak counter and evaluated the stage
outputs before reading it. Values are MLX allocator peaks in decimal GB,
including tensors carried into each stage. Postprocessing cleanup and volume
reductions execute in alternating slabs.

| Stage | 0.4.0 peak (GB) | Staged QC peak (GB) |
| :--- | ---: | ---: |
| Segmentation U-Net, maximum across both passes | 6.43 | 6.43 |
| Segmentation softmax and left/right averaging | 3.29 | 3.29 |
| Gaussian smoothing | 3.29 | 3.29 |
| QC crop and network | 9.03 | 6.27 |
| Parcellation U-Net | 6.62 | 6.62 |
| Parcellation softmax | 2.78 | 2.78 |
| Posterior cleanup and topology masks | 3.90 | 3.77 |
| Volume accumulation | 4.04 | 4.04 |

The parcellation U-Net sets the peak after the change. A second profile took
every other slice of the acquired MRI to exercise resampling from 1 × 1 × 2 mm
to 1 mm and ran with `crop=(160, 192, 192)`. Resampling used no MLX memory. Crop
restoration and reorientation ran on the host, with 3.34 GB of MLX allocations
still live from earlier stages in both versions. The restored 192 × 256 × 256
mask and both tables matched exactly.

Each row uses three paired runs of the public `run` path in fresh processes
with `save_posteriors=False`. It reports the maximum MLX peak through output
writing and the median runner wall time. The last column is the median of
the percentage changes in the individual pairs. The largest slowdown in
any pair was 3.43%.

| Variant | Parcellation | Peak before → after (GB) | Wall time before → after (s) | Median paired change |
| :--- | :--- | ---: | ---: | ---: |
| Standard | Yes | 9.031 → 6.622 | 11.727 → 11.980 | +0.47% |
| Standard | No | 8.978 → 6.379 | 9.163 → 9.265 | +1.04% |
| Robust | Yes | 9.193 → 6.784 | 12.819 → 12.176 | −5.02% |
| Robust | No | 9.140 → 6.383 | 10.591 → 9.919 | −6.06% |
| v1+fast | Yes | 9.763 → 7.354 | 9.711 → 9.598 | −1.74% |
| v1+fast | No | 8.955 → 6.198 | 6.800 → 6.861 | +0.89% |

For all six variant and parcellation combinations, the run with posterior
export had zero differing mask voxels, byte-identical volume and QC CSVs,
and bit-identical float32 posteriors.
