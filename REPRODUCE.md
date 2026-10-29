# Reproduction Guide

## Supported local profile

The verified development profile is Windows 11 with Python 3.12 in a repository-local virtual environment and CPU-only PyTorch. Python 3.14 is intentionally not used because the active installation cannot import the reported user-site PyTorch package.

## Environment creation

For the exact dependency set used by the checked-in run:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

`requirements-lock.txt` records the packages used for the checked-in result. `pyproject.toml` defines supported dependency ranges and the exact PyTorch version.

## Quick experiment

```powershell
.\.venv\Scripts\python.exe -m mlops_gpu_benchmark_lab.cli `
  --config configs/quick.json `
  --device cpu `
  --output-dir reports/runs/quick-cpu
```

Quick mode uses 20,000 stratified observations and is suitable for validating the complete CPU pipeline. It must not be described as a full-dataset performance result.

## Full experiment

```powershell
.\.venv\Scripts\python.exe -m mlops_gpu_benchmark_lab.cli `
  --config configs/full.json `
  --device cpu `
  --output-dir reports/runs/full-cpu
```

Full mode uses all 581,012 observations and may take substantially longer on the audited 4-core CPU.

## GPU validation

On a machine where a CUDA-enabled PyTorch build reports `torch.cuda.is_available() == True`, use `--device cuda`. Preserve the generated `run_metadata.json`; do not combine results from different code, precision, dataset, or hardware configurations into a single speedup claim.

The benchmark performs warm-up and explicit CUDA synchronization around measured regions. It reports compute-only and end-to-end inference latency separately. A GPU result is valid only if the generated environment metadata identifies the device and CUDA runtime.

## Expected artifacts

- `run_metadata.json`: status, timestamp, environment, device, configuration, and split sizes
- `preprocessing.json`: feature order and training-fitted scaler parameters
- `metrics.json`: held-out metrics and training resource measurements
- `benchmark.json` and `benchmark.csv`: raw latency and throughput rows
- `history_*.csv`: epoch-level loss, duration, and training throughput
- `figures/*.png`: measured learning curves, confusion matrices, and benchmark plot
- `checkpoints/*.pt`: best validation checkpoints

## Verification

```powershell
.\.venv\Scripts\ruff.exe check .
.\.venv\Scripts\pytest.exe
git status --short
```
