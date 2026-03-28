import pytest
import torch

from lilaw_poc.lilaw import LiLAWWeighter, compute_lilaw_weights


class TestComputeLilawWeights:
    """Test the three weight functions W_alpha, W_beta, W_delta."""

    def test_combined_weight_is_sum_of_components(self) -> None:
        """W = W_alpha + W_beta + W_delta."""
        alpha, beta, delta = 10.0, 2.0, 6.0
        s_label = torch.tensor([0.8])
        s_max = torch.tensor([0.8])
        w_a, w_b, w_d, w_total = compute_lilaw_weights(
            s_label, s_max, alpha, beta, delta
        )
        assert torch.allclose(w_total, w_a + w_b + w_d, atol=1e-6)

    def test_easy_sample_high_w_alpha(self) -> None:
        """Easy sample (correct, confident): W_alpha should be high."""
        s_label = torch.tensor([0.95])
        s_max = torch.tensor([0.95])
        w_a, _, _, _ = compute_lilaw_weights(s_label, s_max, alpha=10.0, beta=2.0, delta=6.0)
        assert w_a.item() > 0.5

    def test_hard_sample_high_w_beta(self) -> None:
        """Hard sample (incorrect, confident): W_beta should be high."""
        s_label = torch.tensor([0.05])
        s_max = torch.tensor([0.95])
        w_a, w_b, _, _ = compute_lilaw_weights(s_label, s_max, alpha=10.0, beta=2.0, delta=6.0)
        assert w_b.item() > w_a.item()

    def test_weight_bounds_w_alpha(self) -> None:
        """W_alpha is sigmoid output, must be in (0, 1)."""
        s_label = torch.rand(100)
        s_max = torch.rand(100)
        w_a, _, _, _ = compute_lilaw_weights(s_label, s_max, alpha=10.0, beta=2.0, delta=6.0)
        assert (w_a > 0).all() and (w_a < 1).all()

    def test_weight_bounds_w_delta(self) -> None:
        """W_delta is exp(-x^2/2), must be in (0, 1]."""
        s_label = torch.rand(100)
        s_max = torch.rand(100)
        _, _, w_d, _ = compute_lilaw_weights(s_label, s_max, alpha=10.0, beta=2.0, delta=6.0)
        assert (w_d > 0).all() and (w_d <= 1.0 + 1e-6).all()


class TestLiLAWWeighter:
    """Test the meta-parameter container and gradient flow."""

    def test_default_initialization(self) -> None:
        """Check default alpha=10, beta=2, delta=6."""
        w = LiLAWWeighter()
        assert w.alpha.item() == pytest.approx(10.0)
        assert w.beta.item() == pytest.approx(2.0)
        assert w.delta.item() == pytest.approx(6.0)

    def test_gradient_flows_to_meta_params(self) -> None:
        """Backprop through weighted loss must produce gradients on alpha, beta, delta."""
        weighter = LiLAWWeighter()
        logits = torch.randn(8, requires_grad=True)
        labels = torch.randint(0, 2, (8,), dtype=torch.float)
        preds = torch.sigmoid(logits)

        # Binary adaptation: s_label = p when y=1, (1-p) when y=0
        s_label = preds * labels + (1 - preds) * (1 - labels)
        s_max = torch.max(preds, 1 - preds)

        w_a, w_b, w_d, w_total = weighter.compute_weights(s_label, s_max)
        bce = torch.nn.functional.binary_cross_entropy(preds, labels, reduction="none")
        loss = (w_total * bce).mean()
        loss.backward()

        assert weighter.alpha.grad is not None
        assert weighter.beta.grad is not None
        assert weighter.delta.grad is not None

    def test_alpha_gradient_non_negative(self) -> None:
        """Per the paper, dL/d(alpha) >= 0: alpha should decrease during training.

        The gradient of W_alpha w.r.t. alpha is sigmoid'(alpha*s_label - s_max) * s_label,
        which is always non-negative since sigmoid' >= 0 and s_label >= 0.
        """
        weighter = LiLAWWeighter()
        # Use detached s_label/s_max (constants) as in the meta-update step
        s_label = torch.tensor([0.3, 0.5, 0.7, 0.9])
        s_max = torch.tensor([0.7, 0.5, 0.7, 0.9])
        w_a, _, _, _ = weighter.compute_weights(s_label, s_max)
        # Sum of W_alpha as a scalar loss proxy
        w_a.sum().backward()
        assert weighter.alpha.grad is not None
        assert weighter.alpha.grad.item() >= 0.0, (
            f"alpha gradient should be non-negative, got {weighter.alpha.grad.item()}"
        )

    def test_beta_gradient_non_positive(self) -> None:
        """Per the paper, dL/d(beta) <= 0: beta should increase during training.

        The gradient of W_beta w.r.t. beta is -sigmoid'(-(beta*s_label - s_max)) * s_label,
        which is always non-positive since sigmoid' >= 0 and s_label >= 0.
        """
        weighter = LiLAWWeighter()
        s_label = torch.tensor([0.3, 0.5, 0.7, 0.9])
        s_max = torch.tensor([0.7, 0.5, 0.7, 0.9])
        _, w_b, _, _ = weighter.compute_weights(s_label, s_max)
        w_b.sum().backward()
        assert weighter.beta.grad is not None
        assert weighter.beta.grad.item() <= 0.0, (
            f"beta gradient should be non-positive, got {weighter.beta.grad.item()}"
        )

    def test_meta_update_changes_params(self) -> None:
        """A manual SGD step should change alpha, beta, delta."""
        weighter = LiLAWWeighter()
        old_alpha = weighter.alpha.item()

        # Fake gradients
        weighter.alpha.grad = torch.tensor(1.0)
        weighter.beta.grad = torch.tensor(-1.0)
        weighter.delta.grad = torch.tensor(0.5)

        weighter.meta_step(lr=0.005, wd=0.0001)
        assert weighter.alpha.item() != old_alpha
