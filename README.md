# Reproducibility files: Idea 9

This package reproduces the paper's CPU spectral analysis, numerical tables, and statistical plots from the archived float32 gradient histories, update matrices, and final-batch GPU SVD diagnostics. It also checks the GPU factorization's finite-precision residual against CPU float64 SVD. The matrices are the two analyzed 256 x 512 blocks from the frozen Qwen2.5-0.5B checkpoint.

## Run

Python 3.10+, NumPy, and Matplotlib are required. From this directory:

```bash
python figures/reanalyze.py
python figures/make_tables.py
python figures/make_figures.py
```

All calculations use the archived arrays. No model download or GPU is needed. The six input arrays are approximately 36 MB compressed. Their SHA-256 values and shapes are recorded in `results/gradient_matrix_manifest.json`.

## Contents

- `figures/`: CPU analysis, table, and plotting scripts.
- `results/gradient_matrices/`: two 32-minibatch histories, two Adam-style updates, and two final-gradient/GPU-SVD diagnostics.
- `results/source_cpu_summary.json`: only the recorded damping sweep needed to regenerate its table.
- `results/gradient_matrix_manifest.json`: hashes and array descriptions.

The arrays and scripts reproduce the matrix-level results. They do not contain model weights, prompts, server paths, access credentials, or a script for rerunning model inference. Rerunning the frozen-model measurement requires the named public checkpoint and the paper's collection protocol.

No reuse license has been assigned in this package. Add the authors' chosen license before public release.
