# synthseg-mlx

Native MLX port of SynthSeg for brain MRI and CT segmentation, parcellation,
volumes and QC, using native safetensors checkpoints.

## Rules

- Allow `Any` at untyped third-party library boundaries; keep package interfaces typed.

- Model execution runs in MLX on Metal (macOS arm64). There is no PyTorch,
  MPS, NumPy or CPU fallback; host I/O and preprocessing may use CPU
  libraries.
- Unsupported checkpoints, options and inputs raise a clear error. Do not add
  compatibility shims or silent fallbacks.
- Never commit model weights, medical images, patient data or generated
  outputs.
- Keep upstream notices in THIRD_PARTY_NOTICES.md; model weights retain their
  publisher terms.
- Keep the established default operation order; upstream-order variants remain
  explicit options.

## Checks

- `make env` creates the locked development environment.
- `make qa` runs Ruff and Pyright; it does not run models.
- `make test TESTS=tests/test_name.py` runs selected tests; numerical changes
  need the affected comparison against the upstream reference.
- There is no hosted CI; local checks are authoritative.
