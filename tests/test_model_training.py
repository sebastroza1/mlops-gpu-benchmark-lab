from __future__ import annotations

import numpy as np
import torch

from mlops_gpu_benchmark_lab.data import prepare_splits
from mlops_gpu_benchmark_lab.evaluation import evaluate
from mlops_gpu_benchmark_lab.model import create_model
from mlops_gpu_benchmark_lab.training import load_checkpoint, train_model


def test_model_output_shape() -> None:
    model = create_model("small")
    assert model(torch.zeros(8, 54)).shape == (8, 7)


def test_training_checkpoint_inference_and_metrics(tmp_path, synthetic_covertype) -> None:
    features, target = synthetic_covertype
    splits = prepare_splits(features, target, seed=42)
    checkpoint = tmp_path / "model.pt"
    result = train_model(
        splits=splits,
        architecture="small",
        device=torch.device("cpu"),
        epochs=2,
        batch_size=64,
        learning_rate=0.001,
        weight_decay=0.0001,
        patience=2,
        seed=42,
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert len(result.history) == 2
    assert all(np.isfinite(row["train_loss"]) for row in result.history)
    restored = load_checkpoint(checkpoint, torch.device("cpu"))
    metrics = evaluate(restored, splits.x_test, splits.y_test, torch.device("cpu"))
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert np.asarray(metrics["confusion_matrix"]).shape == (7, 7)
