from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn


def predict(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    predictions: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            predictions.append(model(batch).argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def evaluate(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, object]:
    predictions = predict(model, features, device)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix": confusion_matrix(labels, predictions).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=list(range(7)),
            output_dict=True,
            zero_division=0,
        ),
    }
