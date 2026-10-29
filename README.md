# Reproducible PyTorch Training and Compute Benchmarking

This project trains two multilayer perceptrons on the UCI Covertype dataset and measures real training and inference behaviour. A verified quick CPU run on 20,000 stratified observations reached **0.5765 accuracy / 0.4651 macro-F1** for the small MLP and **0.5900 accuracy / 0.4599 macro-F1** for the medium MLP. It separates compute-only latency from end-to-end latency, records hardware and package versions, and never substitutes projected GPU performance for measured results.

> Independent portfolio project by Sebastián Astroza. This repository is not affiliated with or endorsed by CERN.

## What this project demonstrates

- Leakage-safe preparation of a large, openly licensed scientific dataset.
- Reproducible PyTorch model training with AdamW, class weighting, early stopping, and checkpoints.
- Comparison of two MLP capacities under the same data split and seed.
- Accuracy and macro-F1 evaluation on a held-out test set.
- CPU/GPU-ready benchmark code with warm-up, repeated measurements, CUDA synchronization, latency dispersion, throughput, RAM, and optional VRAM reporting.
- Automated tests, linting, and CPU-only continuous integration.

## Problem statement

Covertype is a seven-class classification problem: predict the dominant forest cover type of a 30 x 30 metre cell from 54 cartographic attributes. The dataset contains 581,012 observations and is sufficiently large to expose meaningful training and batching trade-offs on commodity hardware.

## Dataset and licensing

- Dataset: [Covertype, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/31/covertype)
- Creator: Jock Blackard
- DOI: [10.24432/C50K5N](https://doi.org/10.24432/C50K5N)
- Dataset license: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Dimensions: 581,012 rows, 54 features, 7 target classes

The download is cached under `data/cache/`, which is excluded from Git. See [DATASET_CARD.md](DATASET_CARD.md) for provenance, limitations, and leakage controls.

## Methodology

1. Download dataset 31 through `ucimlrepo` or reuse a validated local cache.
2. In quick mode, take a reproducible stratified sample; full mode uses all observations.
3. Create stratified train/validation/test splits (60/20/20).
4. Fit `StandardScaler` on the first 10 continuous variables using the training split only. Keep binary indicator columns unscaled.
5. Train `small` (64-32) and `medium` (128-64-32) ReLU MLPs with class-weighted cross-entropy and AdamW.
6. Select the checkpoint with the lowest validation loss and evaluate once on the held-out test set.
7. Benchmark inference after warm-up across batch sizes 1, 32, 256, and 1,024.

## Experimental results

The checked-in `quick-cpu-20261029` run used 12,000 training, 4,000 validation, and 4,000 test observations on an AMD Ryzen 5 3550H. Both models completed all eight configured epochs.

| Model | Test accuracy | Test macro-F1 | Median epoch time | Peak process RAM |
|---|---:|---:|---:|---:|
| Small MLP | 0.5765 | 0.4651 | 0.395 s | 391.44 MiB |
| Medium MLP | 0.5900 | 0.4599 | 0.444 s | 403.52 MiB |

At batch 1,024, compute-only median inference latency was 0.592 ms for the small network and 1.128 ms for the medium network. The medium network's slight accuracy gain did not improve macro-F1 and required more memory and inference time. See [RESULTS.md](RESULTS.md) and `reports/runs/quick-cpu-20261029/` for raw JSON/CSV, checkpoints, and plots.

No CUDA-compatible device was detected on the development machine. Local evidence is therefore CPU-only; the CUDA path is implemented but must be executed on compatible hardware before any GPU comparison is claimed. The run is a quick-mode validation, not a full-dataset benchmark.

## Quick start

### Windows PowerShell

```powershell
& 'C:\Users\sebas\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m mlops_gpu_benchmark_lab.cli --config configs/quick.json --device cpu --output-dir reports/runs/quick-cpu
```

### Linux

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -m mlops_gpu_benchmark_lab.cli --config configs/quick.json --device cpu --output-dir reports/runs/quick-cpu
```

Use `configs/full.json` for the full dataset. Existing non-empty output directories are protected unless `--overwrite` is supplied.

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
```

The test suite uses generated data and does not require a network connection, GPU, or full Covertype download.

## Repository structure

```text
configs/                         quick and full experiment profiles
reports/runs/                    measured JSON, CSV, plots, and checkpoints
scripts/run_experiment.py        thin executable entry point
src/mlops_gpu_benchmark_lab/     data, models, training, evaluation, benchmark, CLI
tests/                           offline unit and smoke tests
DATASET_CARD.md                  provenance, license, quality, and leakage notes
MODEL_CARD.md                    intended use and model limitations
RESULTS.md                       interpreted measured results
REPRODUCE.md                     exact reproduction and GPU-validation guidance
```

## Limitations

- Covertype represents one US forest region and should not be treated as a universal ecological model.
- Quick mode is a pipeline validation run, not a full benchmark.
- Process RSS is a coarse measure of CPU memory and includes the Python process, libraries, and allocator behaviour.
- The development host has AMD graphics but no supported PyTorch CUDA path; GPU performance is not locally validated.
- This is an experimental portfolio workload, not a production forecasting system.

## Lessons and next steps

The main engineering lesson is that device claims require measured, synchronized workloads and enough work per batch to amortize framework overhead. A future GPU run should reuse the same code revision, dataset split, seed, numerical precision, and batch configurations, then record the new environment rather than merging projected values into the CPU report.

## Author contribution

Sebastián Astroza defined the portfolio objective and project constraints. The repository records the implementation, execution environment, tests, and measurements required to discuss the work transparently in an engineering interview.
