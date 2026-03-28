import pytest
import torch

from lilaw_poc.train import TrainResult, train_baseline, train_lilaw


class TestTrainBaseline:
    """Test vanilla BCE training loop."""

    def test_returns_train_result(self) -> None:
        """Should return a TrainResult with loss history and final model."""
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        x_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        result = train_baseline(
            x_train, y_train, x_val, y_val, input_dim=10, epochs=2, batch_size=32
        )
        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 2
        assert result.model is not None

    def test_loss_decreases(self) -> None:
        """Loss should generally decrease over epochs on separable data."""
        torch.manual_seed(42)
        feats = torch.randn(200, 5)
        y = (feats[:, 0] > 0).long()
        x_train, y_train = feats[:160], y[:160]
        x_val, y_val = feats[160:], y[160:]
        result = train_baseline(
            x_train, y_train, x_val, y_val, input_dim=5, epochs=20, batch_size=32
        )
        assert result.train_losses[-1] < result.train_losses[0]


class TestTrainLilaw:
    """Test LiLAW-weighted training loop."""

    def test_returns_train_result_with_meta_params(self) -> None:
        """Should return TrainResult with meta-parameter history."""
        x_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        x_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        result = train_lilaw(
            x_train, y_train, x_val, y_val, input_dim=10, epochs=3, batch_size=32
        )
        assert isinstance(result, TrainResult)
        assert len(result.meta_params) == 3
        assert "alpha" in result.meta_params[0]

    def test_meta_params_change_during_training(self) -> None:
        """Alpha, beta, delta should evolve from their initial values."""
        torch.manual_seed(42)
        feats = torch.randn(200, 5)
        y = (feats[:, 0] > 0).long()
        result = train_lilaw(
            feats[:160], y[:160], feats[160:], y[160:], input_dim=5, epochs=10, batch_size=32
        )
        first = result.meta_params[0]
        last = result.meta_params[-1]
        changed = any(
            abs(first[k] - last[k]) > 1e-4 for k in ["alpha", "beta", "delta"]
        )
        assert changed, "Meta-parameters should change during training"

    def test_warmup_epoch_uses_vanilla_bce(self) -> None:
        """During warmup (epoch 0), meta-params should not change."""
        torch.manual_seed(42)
        x_train = torch.randn(100, 5)
        y_train = torch.randint(0, 2, (100,))
        x_val = torch.randn(20, 5)
        y_val = torch.randint(0, 2, (20,))
        result = train_lilaw(
            x_train, y_train, x_val, y_val,
            input_dim=5, epochs=2, batch_size=32, warmup_epochs=1,
        )
        assert result.meta_params[0]["alpha"] == pytest.approx(10.0)
        assert result.meta_params[0]["beta"] == pytest.approx(2.0)
        assert result.meta_params[0]["delta"] == pytest.approx(6.0)
