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
