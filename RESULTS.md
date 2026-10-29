# Experimental Results

## Run identity

| Field | Value |
|---|---|
| Status | Completed |
| Mode | Quick |
| Date | 2026-10-29 |
| Dataset | UCI Covertype, ID 31, DOI 10.24432/C50K5N |
| Observations | 20,000 stratified sample |
| Split | 12,000 train / 4,000 validation / 4,000 test |
| Seed | 42 |
| Device | CPU |
| CPU | AMD Ryzen 5 3550H, 4 cores / 8 logical processors |
| PyTorch | 2.9.1+cpu |
| CUDA | Unavailable; not executed |
| Epochs | 8 maximum, both models completed 8 |

The run artifacts are in `reports/runs/quick-cpu-20261029/`. The experiment ran before the repository's initial commit, so `run_metadata.json` has a null Git revision; the result is committed together with the exact source and configuration used.

## Held-out test metrics

| Model | Architecture | Accuracy | Macro-F1 | Best epoch |
|---|---|---:|---:|---:|
| Small | 54-64-32-7 | 0.5765 | 0.4651 | 8 |
| Medium | 54-128-64-32-7 | 0.5900 | 0.4599 | 8 |

The medium network improved accuracy by 0.0135 but reduced macro-F1 by 0.0052. That divergence matters because Covertype is imbalanced: the accuracy increase is not evidence of a uniformly better classifier. The checked-in confusion matrices and per-class classification reports should be used to inspect minority-class errors.

Validation loss was still decreasing at epoch 8 for both models. These quick-mode results validate the complete pipeline but do not establish converged full-dataset performance.

## Training measurements

| Model | Total recorded training time | Median epoch time | Epoch-time IQR | Peak process RAM |
|---|---:|---:|---:|---:|
| Small | 8.852 s | 0.395 s | 0.390 s | 391.44 MiB |
| Medium | 3.718 s | 0.444 s | 0.106 s | 403.52 MiB |

The small model experienced two clear host-side timing disturbances (1.643 s and 5.016 s epochs), so its total duration must not be used to claim that the larger network trains faster. Median epoch time is the more robust comparison for this run: the medium network was about 12% slower per median epoch and used about 12 MiB more process memory.

Process RSS includes the Python interpreter, native libraries, allocator behaviour, and both data and model state. It is not model-only memory.

## Inference benchmark

Each measurement used three warm-up iterations and seven recorded repetitions. Compute-only timing reuses a device tensor. End-to-end timing includes NumPy-to-tensor conversion, transfer to the selected device, forward pass, and output transfer to CPU.

| Model | Batch | Compute median | Compute p95 | Compute throughput | End-to-end median |
|---|---:|---:|---:|---:|---:|
| Small | 1 | 0.136 ms | 0.141 ms | 7,364 samples/s | 0.151 ms |
| Small | 32 | 0.182 ms | 0.186 ms | 175,631 samples/s | 0.188 ms |
| Small | 256 | 0.298 ms | 0.315 ms | 859,060 samples/s | 0.303 ms |
| Small | 1,024 | 0.592 ms | 0.623 ms | 1,729,437 samples/s | 0.612 ms |
| Medium | 1 | 0.158 ms | 0.172 ms | 6,325 samples/s | 0.167 ms |
| Medium | 32 | 0.230 ms | 0.299 ms | 139,373 samples/s | 0.238 ms |
| Medium | 256 | 0.452 ms | 0.537 ms | 566,622 samples/s | 0.462 ms |
| Medium | 1,024 | 1.128 ms | 1.300 ms | 907,801 samples/s | 1.124 ms |

Batching substantially increases throughput for both networks. The small network remains faster at every measured batch size; at batch 1,024 its compute throughput is about 1.9 times the medium network's. Sub-millisecond CPU measurements are sensitive to background load and framework threading, so the raw values should be compared only within this recorded environment and method.

## Verification evidence

- `ruff check .`: passed after two formatting corrections.
- `pytest`: 7 passed in 9.50 seconds on the final pre-experiment verification run.
- Tests cover schema validation, stratified leakage-safe splitting, reproducibility, model dimensions, CPU training, checkpoint save/load, inference, metrics, and benchmark output.
- Generated plots were visually inspected for readable labels, legends, and axes.

## Limitations and next steps

1. Run full mode on all 581,012 observations to measure converged accuracy and macro-F1.
2. Repeat training runs with controlled host load to estimate run-to-run variability; epoch variability is reported here, but only one training run per architecture was executed.
3. Validate the CUDA path on a compatible NVIDIA device or Colab instance using the same revision and configuration.
4. Consider a spatially aware holdout if the goal changes from pipeline benchmarking to geographic generalisation.
5. Do not compare the CPU throughput values with a future GPU run unless precision, batch size, data, warm-up, and measurement boundaries remain equivalent.
