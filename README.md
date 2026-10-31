# synthseg-mlx

Segment brain MRI or CT with SynthSeg in native MLX. Includes cortical parcellation, structure volumes and QC scores.

[Docs](https://github.com/MedMLX/synthseg-mlx/blob/main/docs/technical.md) · [Model weights](https://huggingface.co/MedMLX/synthseg-mlx)

| Input | Output |
| :--- | :--- |
| 3D brain MRI or CT · NIfTI | Brain labels, volumes and QC |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Quick start

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "synthseg-mlx>=0.3.2"
```

```python
from synthseg_mlx.medmlx import run

result = run(image_path="brain.nii.gz", output_dir="out")
print(result["outputs"]["mask"])
```

Omitted weights download automatically from `MedMLX/synthseg-mlx` into the shared
MedMLX cache that `medmlx setup synthseg` also fills. Downloads resume after
interruption; completed downloads work offline. An explicit `weights_path`
overrides this with one local folder. Run details include `weights_revision` for
managed downloads.

## Local weights

Download the converted [SynthSeg weights](https://huggingface.co/MedMLX/synthseg-mlx)
and run:

```python
from huggingface_hub import snapshot_download
from synthseg_mlx.medmlx import run

weights = snapshot_download(
    "MedMLX/synthseg-mlx",
    revision="b4fa99020571cd5921f2205834dd5028fa804046",
    allow_patterns=["synthseg_2.0.safetensors", "synthseg_parc_2.0.safetensors",
                    "synthseg_qc_2.0.safetensors"],
)
result = run(image_path="brain.nii.gz", weights_path=weights, output_dir="out")
print(result["outputs"]["mask"])
```

Inputs are resampled to 1 mm if any voxel size differs from 1 mm by more than
0.05 mm; otherwise the
input grid is kept. The mask shares its grid with `resampled_input.nii.gz`.

## License

Apache-2.0; license and third-party notices are included in the distribution. Model weights retain their publisher terms.

For research and education. Outputs require human review.
