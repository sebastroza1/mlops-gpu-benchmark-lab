from __future__ import annotations

import statistics
import time

import numpy as np
import psutil
import torch
from torch import nn


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summarize(milliseconds: list[float], samples: int) -> dict[str, float]:
    ordered = sorted(milliseconds)
    median = statistics.median(ordered)
    p95_index = min(len(ordered) - 1, max(0, int(np.ceil(0.95 * len(ordered))) - 1))
    return {
        "latency_median_ms": median,
        "latency_p95_ms": ordered[p95_index],
        "latency_iqr_ms": float(np.percentile(ordered, 75) - np.percentile(ordered, 25)),
        "throughput_samples_per_second": samples / (median / 1000.0),
    }


def benchmark_inference(
    model: nn.Module,
    features: np.ndarray,
    device: torch.device,
    batch_sizes: tuple[int, ...],
    warmup: int,
    repeats: int,
) -> list[dict[str, float | int | str | None]]:
    model.eval()
    process = psutil.Process()
    results: list[dict[str, float | int | str | None]] = []

    for requested_batch in batch_sizes:
        size = min(requested_batch, len(features))
        numpy_batch = np.ascontiguousarray(features[:size])
        device_batch = torch.from_numpy(numpy_batch).to(device)
        with torch.inference_mode():
            for _ in range(warmup):
                model(device_batch)
            _synchronize(device)

            compute_times: list[float] = []
            end_to_end_times: list[float] = []
            transfer_times: list[float] = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            for _ in range(repeats):
                started = time.perf_counter()
                model(device_batch)
                _synchronize(device)
                compute_times.append((time.perf_counter() - started) * 1000.0)

                started = time.perf_counter()
                transferred = torch.from_numpy(numpy_batch).to(device)
                _synchronize(device)
                transfer_times.append((time.perf_counter() - started) * 1000.0)

                started = time.perf_counter()
                transferred = torch.from_numpy(numpy_batch).to(device)
                output = model(transferred)
                output.cpu()
                _synchronize(device)
                end_to_end_times.append((time.perf_counter() - started) * 1000.0)

        compute = _summarize(compute_times, size)
        end_to_end = _summarize(end_to_end_times, size)
        results.append(
            {
                "device": str(device),
                "batch_size": size,
                "warmup_iterations": warmup,
                "repetitions": repeats,
                "compute_latency_median_ms": compute["latency_median_ms"],
                "compute_latency_p95_ms": compute["latency_p95_ms"],
                "compute_latency_iqr_ms": compute["latency_iqr_ms"],
                "compute_throughput_samples_per_second": compute[
                    "throughput_samples_per_second"
                ],
                "end_to_end_latency_median_ms": end_to_end["latency_median_ms"],
                "end_to_end_latency_p95_ms": end_to_end["latency_p95_ms"],
                "end_to_end_throughput_samples_per_second": end_to_end[
                    "throughput_samples_per_second"
                ],
                "transfer_latency_median_ms": statistics.median(transfer_times),
                "process_rss_mb": process.memory_info().rss / (1024**2),
                "peak_vram_mb": (
                    torch.cuda.max_memory_allocated(device) / (1024**2)
                    if device.type == "cuda"
                    else None
                ),
            }
        )
    return results
