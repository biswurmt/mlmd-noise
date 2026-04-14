"""LiLAW: Lightweight Learnable Adaptive Weighting.

Implements the three weight functions (W_alpha, W_beta, W_delta) and the
meta-parameter container with manual SGD update for the bilevel optimization.

Reference: "Lightweight Learnable Adaptive Weighting to Claim the Jungle of
Noisy Labels" (arXiv:2502.01981).
"""

import torch


def compute_lilaw_weights(
    s_label: torch.Tensor,
    s_max: torch.Tensor,
    alpha: float | torch.Tensor,
    beta: float | torch.Tensor,
    delta: float | torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute LiLAW per-sample weights from predicted probabilities.

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

    Meta-parameters are plain tensors (not nn.Parameters) so they live
    outside Lightning's optimizer. Device placement is handled explicitly
    via to().

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

    def to(self, device: torch.device | str) -> "LiLAWWeighter":
        """Move meta-parameters to device, preserving grad state."""
        self.alpha = self.alpha.detach().to(device).requires_grad_(True)
        self.beta = self.beta.detach().to(device).requires_grad_(True)
        self.delta = self.delta.detach().to(device).requires_grad_(True)
        return self

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

    def set_requires_grad(self, *, requires_grad: bool) -> None:
        """Toggle gradient tracking for all meta-parameters."""
        self.alpha.requires_grad_(requires_grad)
        self.beta.requires_grad_(requires_grad)
        self.delta.requires_grad_(requires_grad)

    def state_dict(self) -> dict[str, float]:
        """Return current meta-parameter values as plain floats."""
        return {
            "alpha": self.alpha.item(),
            "beta": self.beta.item(),
            "delta": self.delta.item(),
        }
