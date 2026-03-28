# Task 5: Dataset Loaders

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/data/datasets.py`
- Create: `lilaw-poc/tests/test_datasets.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_datasets.py
import pytest
import torch

from lilaw_poc.data.datasets import load_dataset, DATASET_NAMES


class TestLoadDataset:
    """Test dataset loading and preprocessing for all three PoC datasets."""

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_returns_tensors(self, name: str) -> None:
        """All splits should be float32 tensors."""
        splits = load_dataset(name, seed=42)
        for key in ["X_train", "X_val", "X_test"]:
            assert splits[key].dtype == torch.float32
        for key in ["y_train", "y_val", "y_test"]:
            assert splits[key].dtype == torch.long

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_split_ratios(self, name: str) -> None:
        """Splits should approximate 55/15/30."""
        splits = load_dataset(name, seed=42)
        total = sum(splits[k].shape[0] for k in ["y_train", "y_val", "y_test"])
        train_frac = splits["y_train"].shape[0] / total
        val_frac = splits["y_val"].shape[0] / total
        test_frac = splits["y_test"].shape[0] / total
        assert abs(train_frac - 0.55) < 0.05
        assert abs(val_frac - 0.15) < 0.05
        assert abs(test_frac - 0.30) < 0.05

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_binary_labels(self, name: str) -> None:
        """Labels must be binary (0 or 1)."""
        splits = load_dataset(name, seed=42)
        for key in ["y_train", "y_val", "y_test"]:
            unique = splits[key].unique()
            assert set(unique.tolist()).issubset({0, 1})

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_features_standardized(self, name: str) -> None:
        """Training features should be approximately zero-mean, unit-variance."""
        splits = load_dataset(name, seed=42)
        mean = splits["X_train"].mean(dim=0)
        std = splits["X_train"].std(dim=0)
        assert mean.abs().mean() < 0.1
        # std won't be exactly 1 for all features but should be in a reasonable range
        assert std.mean() < 2.0

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_no_nans(self, name: str) -> None:
        """No NaN values in any split."""
        splits = load_dataset(name, seed=42)
        for key in splits:
            assert not torch.isnan(splits[key]).any(), f"NaN found in {key}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/data/datasets.py
"""Dataset loaders for the three PoC tabular datasets.

Each loader returns standardized float32 feature tensors and binary long label tensors
split into 55% train / 15% val / 30% test.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_NAMES: list[str] = ["breast_cancer", "adult", "pima"]


def load_dataset(name: str, seed: int = 42) -> dict[str, torch.Tensor]:
    """Load and preprocess a dataset by name.

    Args:
        name: One of "breast_cancer", "adult", "pima".
        seed: Random seed for reproducible splits.

    Returns:
        Dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    """
    loaders = {
        "breast_cancer": _load_breast_cancer,
        "adult": _load_adult,
        "pima": _load_pima,
    }
    if name not in loaders:
        msg = f"Unknown dataset: {name}. Choose from {DATASET_NAMES}"
        raise ValueError(msg)
    return loaders[name](seed)


def _split_and_standardize(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Split 55/15/30 and standardize features using training set stats."""
    # First split: 70% train+val, 30% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Second split: 55/15 from the 70% (≈ 78.6% / 21.4%)
    val_frac = 0.15 / 0.70
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
    }


def _load_breast_cancer(seed: int) -> dict[str, torch.Tensor]:
    """Load sklearn breast cancer dataset (30 features, binary)."""
    data = load_breast_cancer()
    return _split_and_standardize(data.data, data.target, seed)


def _load_adult(seed: int) -> dict[str, torch.Tensor]:
    """Load Adult Census Income dataset from OpenML (mixed features, binary)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame
    y = (df[data.target_names[0]] == ">50K").astype(int).values

    # One-hot encode categoricals, fill NaN
    X = df.drop(columns=data.target_names)
    X = X.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == "object" or col.dtype.name == "category" else col.fillna(col.median()))
    X = pd.get_dummies(X, drop_first=True).values.astype(np.float64)

    return _split_and_standardize(X, y, seed)


def _load_pima(seed: int) -> dict[str, torch.Tensor]:
    """Load Pima Indians Diabetes dataset from OpenML (8 numeric features, binary)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"] == "tested_positive").astype(int).values
    X = df.drop(columns=["class"]).values.astype(np.float64)

    return _split_and_standardize(X, y, seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datasets.py -v`
Expected: All 15 tests PASS (5 tests × 3 datasets). First run of adult/pima may download data.

- [ ] **Step 5: Lint, type check, commit**

```bash
ruff check src/lilaw_poc/data/datasets.py
git add src/lilaw_poc/data/datasets.py tests/test_datasets.py
git commit -m "feat(lilaw-poc): add dataset loaders for breast cancer, adult, pima"
```
