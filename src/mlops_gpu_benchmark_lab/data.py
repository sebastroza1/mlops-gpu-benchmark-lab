from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo

UCI_DATASET_ID = 31
EXPECTED_FEATURES = 54
CONTINUOUS_FEATURES = 10
EXPECTED_CLASSES = set(range(1, 8))


@dataclass(frozen=True)
class DataSplits:
    x_train: np.ndarray
    y_train: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler_mean: list[float]
    scaler_scale: list[float]


def validate_covertype(features: pd.DataFrame, target: pd.Series) -> None:
    if features.shape[1] != EXPECTED_FEATURES:
        raise ValueError(f"Expected {EXPECTED_FEATURES} features, found {features.shape[1]}.")
    if len(features) != len(target) or len(features) == 0:
        raise ValueError("Features and target must have the same non-zero row count.")
    if features.isna().any().any() or target.isna().any():
        raise ValueError("Covertype must not contain missing values.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in features.dtypes):
        raise ValueError("All Covertype features must be numeric.")
    classes = set(pd.to_numeric(target).astype(int).unique())
    if classes != EXPECTED_CLASSES:
        raise ValueError(f"Expected classes 1-7, found {sorted(classes)}.")


def load_covertype(cache_dir: str | Path) -> tuple[pd.DataFrame, pd.Series, str]:
    cache_path = Path(cache_dir) / "covertype.csv.gz"
    if cache_path.exists():
        frame = pd.read_csv(cache_path)
        if "Cover_Type" not in frame:
            raise ValueError(f"Cached dataset is missing Cover_Type: {cache_path}")
        features = frame.drop(columns="Cover_Type")
        target = frame["Cover_Type"]
        source = "local-cache"
    else:
        try:
            dataset = fetch_ucirepo(id=UCI_DATASET_ID)
        except Exception as exc:
            raise RuntimeError(
                "Could not download UCI Covertype (dataset 31). Check network access and retry."
            ) from exc
        features = dataset.data.features.copy()
        target_frame = dataset.data.targets.copy()
        target = target_frame.iloc[:, 0].rename("Cover_Type")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.concat([features, target], axis=1).to_csv(cache_path, index=False, compression="gzip")
        source = "uci-ml-repository"

    target = pd.to_numeric(target).astype(int)
    validate_covertype(features, target)
    return features, target, source


def stratified_sample(
    features: pd.DataFrame,
    target: pd.Series,
    sample_size: int | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.Series]:
    if sample_size is None or sample_size >= len(features):
        return features.reset_index(drop=True), target.reset_index(drop=True)
    if sample_size < len(EXPECTED_CLASSES) * 3:
        raise ValueError("Sample size is too small for stratified train/validation/test splits.")
    sampled_x, _, sampled_y, _ = train_test_split(
        features,
        target,
        train_size=sample_size,
        random_state=seed,
        stratify=target,
    )
    return sampled_x.reset_index(drop=True), sampled_y.reset_index(drop=True)


def prepare_splits(
    features: pd.DataFrame,
    target: pd.Series,
    seed: int,
    test_size: float = 0.2,
    validation_size: float = 0.2,
) -> DataSplits:
    validate_covertype(features, target)
    indices = np.arange(len(features))
    train_validation, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=target.to_numpy(),
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        train_validation,
        test_size=relative_validation,
        random_state=seed,
        stratify=target.iloc[train_validation].to_numpy(),
    )

    arrays = features.to_numpy(dtype=np.float32, copy=True)
    labels = target.to_numpy(dtype=np.int64, copy=True) - 1
    scaler = StandardScaler()
    arrays[train, :CONTINUOUS_FEATURES] = scaler.fit_transform(
        arrays[train, :CONTINUOUS_FEATURES]
    )
    arrays[validation, :CONTINUOUS_FEATURES] = scaler.transform(
        arrays[validation, :CONTINUOUS_FEATURES]
    )
    arrays[test, :CONTINUOUS_FEATURES] = scaler.transform(
        arrays[test, :CONTINUOUS_FEATURES]
    )

    return DataSplits(
        x_train=arrays[train],
        y_train=labels[train],
        x_validation=arrays[validation],
        y_validation=labels[validation],
        x_test=arrays[test],
        y_test=labels[test],
        feature_names=features.columns.astype(str).tolist(),
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
    )
