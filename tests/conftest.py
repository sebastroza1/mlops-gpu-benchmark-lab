from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_covertype() -> tuple[pd.DataFrame, pd.Series]:
    generator = np.random.default_rng(42)
    rows = 700
    continuous = generator.normal(size=(rows, 10)).astype(np.float32)
    categorical = generator.integers(0, 2, size=(rows, 44)).astype(np.float32)
    features = pd.DataFrame(
        np.concatenate((continuous, categorical), axis=1),
        columns=[f"feature_{index}" for index in range(54)],
    )
    target = pd.Series(np.tile(np.arange(1, 8), rows // 7), name="Cover_Type")
    return features, target
