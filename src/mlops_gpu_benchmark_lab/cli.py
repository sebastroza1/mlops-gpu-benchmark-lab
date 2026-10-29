from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .benchmark import benchmark_inference
from .config import ExperimentConfig
from .data import load_covertype, prepare_splits, stratified_sample
from .environment import collect_environment, resolve_device
from .evaluation import evaluate
from .reporting import plot_benchmark, plot_confusion, plot_history, write_json
from .training import train_model


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def run(args: argparse.Namespace) -> Path:
    config = ExperimentConfig.from_json(args.config)
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    checkpoints = output / "checkpoints"
    figures.mkdir(exist_ok=True)
    checkpoints.mkdir(exist_ok=True)

    environment = collect_environment()
    device = resolve_device(args.device)
    features, target, data_source = load_covertype(args.cache_dir)
    features, target = stratified_sample(features, target, config.sample_size, config.seed)
    splits = prepare_splits(features, target, config.seed)

    run_metadata = {
        "status": "completed",
        "started_at_utc": datetime.now(UTC).isoformat(),
        "mode": config.mode,
        "configuration": asdict(config),
        "environment": environment,
        "selected_device": str(device),
        "dataset": {
            "name": "Covertype",
            "uci_id": 31,
            "doi": "10.24432/C50K5N",
            "source": data_source,
            "observations_used": len(features),
            "training_observations": len(splits.x_train),
            "validation_observations": len(splits.x_validation),
            "test_observations": len(splits.x_test),
            "features": len(splits.feature_names),
        },
        "git_revision": _git_revision(),
    }
    write_json(output / "run_metadata.json", run_metadata)
    write_json(
        output / "preprocessing.json",
        {
            "continuous_feature_count": 10,
            "scaler_fit": "training split only",
            "scaler_mean": splits.scaler_mean,
            "scaler_scale": splits.scaler_scale,
            "feature_names": splits.feature_names,
        },
    )

    histories: dict[str, list[dict[str, object]]] = {}
    model_results: dict[str, object] = {}
    benchmark_rows: list[dict[str, object]] = []
    for architecture in ("small", "medium"):
        training = train_model(
            splits=splits,
            architecture=architecture,
            device=device,
            epochs=config.epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            patience=config.patience,
            seed=config.seed,
            checkpoint_path=checkpoints / f"{architecture}.pt",
        )
        metrics = evaluate(training.model, splits.x_test, splits.y_test, device)
        benchmark = benchmark_inference(
            training.model,
            splits.x_test,
            device,
            batch_sizes=(1, 32, 256, 1024),
            warmup=config.benchmark_warmup,
            repeats=config.benchmark_repeats,
        )
        for row in benchmark:
            row["model"] = architecture
            benchmark_rows.append(row)
        histories[architecture] = training.history
        model_results[architecture] = {
            "best_epoch": training.best_epoch,
            "training_total_seconds": training.total_seconds,
            "training_peak_ram_mb": training.peak_ram_mb,
            "training_peak_vram_mb": training.peak_vram_mb,
            "checkpoint": training.checkpoint,
            "metrics": metrics,
        }
        pd.DataFrame(training.history).to_csv(output / f"history_{architecture}.csv", index=False)
        plot_confusion(
            metrics["confusion_matrix"], architecture, figures / f"confusion_{architecture}.png"
        )

    write_json(output / "metrics.json", model_results)
    write_json(output / "benchmark.json", benchmark_rows)
    pd.DataFrame(benchmark_rows).to_csv(output / "benchmark.csv", index=False)
    plot_history(histories, figures / "learning_curves.png")
    plot_benchmark(benchmark_rows, figures / "inference_throughput.png")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quick.json")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--output-dir", default="reports/runs/quick-cpu")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = run(args)
    print(json.dumps({"status": "completed", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
