from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlops_gpu_benchmark_lab.data import (
    load_covertype,
    prepare_splits,
    stratified_sample,
    validate_covertype,
)


def test_split_is_stratified_scaled_and_reproducible(synthetic_covertype) -> None:
    features, target = synthetic_covertype
    first = prepare_splits(features, target, seed=42)
    second = prepare_splits(features, target, seed=42)

    assert (len(first.y_train), len(first.y_validation), len(first.y_test)) == (420, 140, 140)
    assert set(first.y_train) == set(range(7))
    assert np.allclose(first.x_train[:, :10].mean(axis=0), 0.0, atol=1e-6)
    assert np.array_equal(first.x_train, second.x_train)
    assert np.array_equal(first.y_test, second.y_test)


def test_stratified_sample_preserves_every_class(synthetic_covertype) -> None:
    features, target = synthetic_covertype
    sampled_x, sampled_y = stratified_sample(features, target, sample_size=350, seed=7)
    assert sampled_x.shape == (350, 54)
    assert set(sampled_y) == set(range(1, 8))


def test_cached_dataset_is_validated(tmp_path, synthetic_covertype) -> None:
    features, target = synthetic_covertype
    cache = tmp_path / "covertype.csv.gz"
    pd.concat((features, target), axis=1).to_csv(cache, index=False, compression="gzip")
    loaded_x, loaded_y, source = load_covertype(tmp_path)
    assert loaded_x.shape == features.shape
    assert loaded_y.tolist() == target.tolist()
    assert source == "local-cache"


def test_schema_rejects_missing_values(synthetic_covertype) -> None:
    features, target = synthetic_covertype
    features.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="missing"):
        validate_covertype(features, target)
