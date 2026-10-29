# Model Card: Covertype MLP Classifiers

## Models

| Name | Hidden layers | Output | Optimizer |
|---|---|---|---|
| `small` | 64, 32 | 7 logits | AdamW |
| `medium` | 128, 64, 32 | 7 logits | AdamW |

Both networks use ReLU activation, dropout 0.1, class-weighted cross-entropy, early stopping on validation loss, and a fixed experiment seed.

## Intended use

The models are educational and experimental classifiers used to study reproducible training and compute performance on tabular data. They are not intended for operational land management, safety decisions, or ecological policy.

## Training and selection

- Training, validation, and test partitions are stratified.
- Continuous scaling is fitted on the training split only.
- Hyperparameters and model selection use training/validation data.
- The held-out test set is used for final accuracy and macro-F1 evaluation.
- The checkpoint with the lowest validation loss is saved.

## Evaluation

The verified quick CPU run on 2026-10-29 used 20,000 stratified observations and an independent 4,000-row test set.

| Model | Accuracy | Macro-F1 | Best epoch |
|---|---:|---:|---:|
| `small` | 0.5765 | 0.4651 | 8 |
| `medium` | 0.5900 | 0.4599 | 8 |

Both validation losses were still decreasing at epoch 8, so quick-mode results should not be interpreted as converged full-dataset performance. Per-class metrics, confusion matrices, resource measurements, and raw inference timings are stored under `reports/runs/quick-cpu-20261029/` and interpreted in `RESULTS.md`.

## Limitations and risks

- A random split does not measure geographic generalisation.
- Class weighting changes the optimisation objective and can trade majority-class accuracy for minority-class recall.
- CPU and GPU results are comparable only when dataset, split, model, precision, and measurement method remain fixed.
- The local host has no supported CUDA device, so CUDA code is not locally validated.
