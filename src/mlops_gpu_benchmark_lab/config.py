from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str
    sample_size: int | None
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    benchmark_repeats: int
    benchmark_warmup: int
    seed: int

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        with Path(path).open(encoding="utf-8") as stream:
            values = json.load(stream)
        return cls(**values)
