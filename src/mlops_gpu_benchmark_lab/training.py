from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import DataSplits
from .model import create_model


@dataclass(frozen=True)
class TrainingResult:
    model: nn.Module
    history: list[dict[str, float | int]]
    best_epoch: int
    total_seconds: float
    peak_ram_mb: float
    peak_vram_mb: float | None
    checkpoint: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        pin_memory=pin_memory,
        num_workers=0,
    )


def train_model(
    splits: DataSplits,
    architecture: str,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    seed: int,
    checkpoint_path: str | Path,
) -> TrainingResult:
    set_seed(seed)
    model = create_model(architecture, splits.x_train.shape[1]).to(device)
    pin_memory = device.type == "cuda"
    train_loader = _loader(
        splits.x_train, splits.y_train, batch_size, True, seed, pin_memory
    )
    validation_loader = _loader(
        splits.x_validation, splits.y_validation, batch_size, False, seed, pin_memory
    )
    class_counts = np.bincount(splits.y_train, minlength=7)
    class_weights = len(splits.y_train) / (len(class_counts) * class_counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    process = psutil.Process()
    peak_ram = process.memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    history: list[dict[str, float | int]] = []
    best_loss = float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        training_loss = 0.0
        seen = 0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device, non_blocking=pin_memory)
            batch_labels = batch_labels.to(device, non_blocking=pin_memory)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(batch_features), batch_labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item() * len(batch_labels)
            seen += len(batch_labels)
            peak_ram = max(peak_ram, process.memory_info().rss)

        model.eval()
        validation_loss = 0.0
        validation_seen = 0
        with torch.inference_mode():
            for batch_features, batch_labels in validation_loader:
                batch_features = batch_features.to(device, non_blocking=pin_memory)
                batch_labels = batch_labels.to(device, non_blocking=pin_memory)
                loss = criterion(model(batch_features), batch_labels)
                validation_loss += loss.item() * len(batch_labels)
                validation_seen += len(batch_labels)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        train_loss = training_loss / seen
        val_loss = validation_loss / validation_seen
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "epoch_seconds": epoch_seconds,
                "training_samples_per_second": seen / epoch_seconds,
            }
        )

        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    total_seconds = time.perf_counter() - started
    model.load_state_dict(best_state)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": architecture,
            "input_features": splits.x_train.shape[1],
            "state_dict": best_state,
            "best_epoch": best_epoch,
            "seed": seed,
        },
        checkpoint,
    )
    peak_vram = None
    if device.type == "cuda":
        peak_vram = torch.cuda.max_memory_allocated(device) / (1024**2)

    return TrainingResult(
        model=model,
        history=history,
        best_epoch=best_epoch,
        total_seconds=total_seconds,
        peak_ram_mb=peak_ram / (1024**2),
        peak_vram_mb=peak_vram,
        checkpoint=str(checkpoint),
    )


def load_checkpoint(path: str | Path, device: torch.device) -> nn.Module:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = create_model(checkpoint["architecture"], checkpoint["input_features"])
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval()
