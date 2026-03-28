"""Asymmetric label noise injection for simulating unobserved external tests."""

import numpy as np
import torch


def inject_asymmetric_noise(
    labels: torch.Tensor,
    flip_rate: float,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flip a fraction of positive labels (1→0) to simulate missing external tests.

    Args:
        labels: Binary label tensor (0 or 1).
        flip_rate: Probability of flipping each positive label to negative.
        rng: NumPy random generator for reproducibility.

    Returns:
        Tuple of (noisy_labels, flip_mask) where flip_mask is True at flipped positions.
    """
    noisy = labels.clone()
    flip_mask = torch.zeros_like(labels, dtype=torch.bool)

    pos_indices = (labels == 1).nonzero(as_tuple=True)[0]
    if len(pos_indices) == 0 or flip_rate == 0.0:
        return noisy, flip_mask

    flip_decisions = rng.random(len(pos_indices)) < flip_rate
    flip_indices = pos_indices[torch.from_numpy(flip_decisions)]

    noisy[flip_indices] = 0
    flip_mask[flip_indices] = True

    return noisy, flip_mask
