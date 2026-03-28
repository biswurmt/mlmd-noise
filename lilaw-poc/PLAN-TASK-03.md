# Task 3: LiLAW Weight Functions & Meta-Parameters

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/lilaw.py`
- Create: `lilaw-poc/tests/test_lilaw.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_lilaw.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lilaw.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/lilaw.py
"""LiLAW: Lightweight Learnable Adaptive Weighting.

Implements the three weight functions (W_alpha, W_beta, W_delta) and the
meta-parameter container with manual SGD update for the bilevel optimization.
"""

import torch


def compute_lilaw_weights(
    s_label: torch.Tensor,
    s_max: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    delta: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LiLAW sample weights from softmax outputs.

    Args:
        s_label: Predicted probability for the observed label, shape (N,).
        s_max: Maximum predicted probability across classes, shape (N,).
        alpha: Easy-sample scaling parameter.
        beta: Hard-sample scaling parameter.
        delta: Moderate-sample scaling parameter.

    Returns:
        Tuple of (W_alpha, W_beta, W_delta, W_total).
    """
    diff_a = alpha * s_label - s_max
    diff_b = beta * s_label - s_max
    diff_d = delta * s_label - s_max

    w_alpha = torch.sigmoid(diff_a)
    w_beta = torch.sigmoid(-diff_b)
    w_delta = torch.exp(-diff_d.pow(2) / 2.0)

    w_total = w_alpha + w_beta + w_delta
    return w_alpha, w_beta, w_delta, w_total


class LiLAWWeighter:
    """Container for LiLAW meta-parameters with manual SGD update.

    Attributes:
        alpha: Learnable scalar for easy-sample weighting (init 10.0).
        beta: Learnable scalar for hard-sample weighting (init 2.0).
        delta: Learnable scalar for moderate-sample weighting (init 6.0).
    """

    def __init__(
        self,
        alpha_init: float = 10.0,
        beta_init: float = 2.0,
        delta_init: float = 6.0,
    ) -> None:
        self.alpha = torch.tensor(alpha_init, requires_grad=True)
        self.beta = torch.tensor(beta_init, requires_grad=True)
        self.delta = torch.tensor(delta_init, requires_grad=True)

    def compute_weights(
        self,
        s_label: torch.Tensor,
        s_max: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute weights using current meta-parameters."""
        return compute_lilaw_weights(s_label, s_max, self.alpha, self.beta, self.delta)

    def meta_step(self, lr: float = 0.005, wd: float = 0.0001) -> None:
        """Manual SGD update on meta-parameters with weight decay."""
        with torch.no_grad():
            for param in [self.alpha, self.beta, self.delta]:
                if param.grad is not None:
                    param -= lr * (param.grad + wd * param)
        self.zero_grad()

    def zero_grad(self) -> None:
        """Zero all meta-parameter gradients."""
        for param in [self.alpha, self.beta, self.delta]:
            if param.grad is not None:
                param.grad.zero_()

    def set_requires_grad(self, requires_grad: bool) -> None:
        """Toggle gradient tracking for all meta-parameters."""
        self.alpha.requires_grad_(requires_grad)
        self.beta.requires_grad_(requires_grad)
        self.delta.requires_grad_(requires_grad)

    def state_dict(self) -> dict[str, float]:
        """Return current meta-parameter values."""
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "delta": self.delta.item(),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lilaw.py -v`
Expected: All 8 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `ruff check src/lilaw_poc/lilaw.py && ty check src/lilaw_poc/lilaw.py`

- [ ] **Step 6: Commit**

```bash
git add src/lilaw_poc/lilaw.py tests/test_lilaw.py
git commit -m "feat(lilaw-poc): add LiLAW weight functions and meta-parameter container"
```
