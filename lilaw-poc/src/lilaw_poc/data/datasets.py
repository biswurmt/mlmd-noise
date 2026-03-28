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
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Split 55/15/30 and standardize features using training set stats."""
    # First split: 70% train+val, 30% test
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Second split: 55/15 from the 70% (approx 78.6% / 21.4%)
    val_frac = 0.15 / 0.70
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=val_frac, random_state=seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)

    return {
        "X_train": torch.tensor(x_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(x_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "X_test": torch.tensor(x_test, dtype=torch.float32),
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
    x = df.drop(columns=data.target_names)
    x = x.apply(
        lambda col: col.fillna(col.mode()[0])
        if col.dtype == "object" or col.dtype.name == "category"
        else col.fillna(col.median())
    )
    x_arr = pd.get_dummies(x, drop_first=True).values.astype(np.float64)

    return _split_and_standardize(x_arr, y, seed)


def _load_pima(seed: int) -> dict[str, torch.Tensor]:
    """Load Pima Indians Diabetes dataset from OpenML (8 numeric features, binary)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"] == "tested_positive").astype(int).values
    x_arr = df.drop(columns=["class"]).values.astype(np.float64)

    return _split_and_standardize(x_arr, y, seed)
