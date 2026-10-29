from __future__ import annotations

import numpy as np
import torch

from mlops_gpu_benchmark_lab.benchmark import benchmark_inference
from mlops_gpu_benchmark_lab.model import create_model


def test_cpu_benchmark_reports_measured_latency() -> None:
    model = create_model("small").eval()
    features = np.zeros((64, 54), dtype=np.float32)
    rows = benchmark_inference(
        model,
        features,
        torch.device("cpu"),
        batch_sizes=(1, 32),
        warmup=1,
        repeats=3,
    )
    assert [row["batch_size"] for row in rows] == [1, 32]
    assert all(row["compute_latency_median_ms"] > 0 for row in rows)
    assert all(row["compute_throughput_samples_per_second"] > 0 for row in rows)
    assert all(row["peak_vram_mb"] is None for row in rows)
