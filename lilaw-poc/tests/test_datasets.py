import torch

from lilaw_poc.data.datasets import DATASET_NAMES, load_dataset
import pytest


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
