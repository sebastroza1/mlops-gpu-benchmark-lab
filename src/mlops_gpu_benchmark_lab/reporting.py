from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_json(path: str | Path, payload: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)


def plot_history(histories: dict[str, list[dict[str, object]]], path: str | Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, history in histories.items():
        frame = pd.DataFrame(history)
        axis.plot(frame["epoch"], frame["train_loss"], label=f"{model_name} train")
        axis.plot(
            frame["epoch"],
            frame["validation_loss"],
            linestyle="--",
            label=f"{model_name} validation",
        )
    axis.set(title="Training and validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion(matrix: list[list[int]], model_name: str, path: str | Path) -> None:
    values = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(values, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        title=f"{model_name} test confusion matrix",
        xlabel="Predicted class (zero-based)",
        ylabel="True class (zero-based)",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_benchmark(rows: list[dict[str, object]], path: str | Path) -> None:
    frame = pd.DataFrame(rows)
    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, group in frame.groupby("model"):
        axis.plot(
            group["batch_size"],
            group["compute_throughput_samples_per_second"],
            marker="o",
            label=model_name,
        )
    axis.set(
        title="Inference throughput by batch size",
        xlabel="Batch size",
        ylabel="Samples per second (compute only)",
        xscale="log",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
