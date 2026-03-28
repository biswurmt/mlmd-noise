"""Symmetric label noise injection for MedMNIST replication."""

import numpy as np
import torch


def inject_symmetric_noise(
    labels: torch.Tensor,
    noise_rate: float,
    num_classes: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace each label with a uniformly random label with probability noise_rate.

    The effective flip rate is noise_rate * (K-1)/K since the original label
    can be re-drawn.

    Args:
        labels: Integer class labels, shape (N,).
        noise_rate: Probability of replacing each label.
        num_classes: Total number of classes K.
        rng: NumPy RNG for reproducibility.

    Returns:
        Tuple of (noisy_labels, flip_mask) where flip_mask[i] is True if
        label[i] was changed to a different class.
    """
    noisy = labels.clone()
    n = len(labels)

    if noise_rate == 0.0 or num_classes < 2:
        return noisy, torch.zeros(n, dtype=torch.bool)

    flip_decisions = rng.random(n) < noise_rate
    new_labels = torch.from_numpy(rng.integers(0, num_classes, size=n)).to(labels.dtype)

    flip_indices = torch.from_numpy(np.where(flip_decisions)[0])
    noisy[flip_indices] = new_labels[flip_indices]

    flip_mask = noisy != labels
    return noisy, flip_mask
