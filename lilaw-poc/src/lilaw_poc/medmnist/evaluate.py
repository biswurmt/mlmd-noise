"""Multi-class evaluation metrics: Top-1 accuracy and AUROC."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score


def compute_accuracy(y_true: torch.Tensor, logits: torch.Tensor) -> float:
    """Compute Top-1 accuracy from logits.

    Args:
        y_true: Integer class labels, shape (N,).
        logits: Raw model outputs, shape (N, K).

    Returns:
        Top-1 accuracy as a float in [0, 1].
    """
    preds = logits.argmax(dim=1)
    return float((preds == y_true).float().mean().item())


def compute_auroc(
    y_true: torch.Tensor,
    probs: torch.Tensor,
    num_classes: int,
) -> float:
    """Compute macro-averaged AUROC (one-vs-rest).

    Args:
        y_true: Integer class labels, shape (N,).
        probs: Softmax probabilities, shape (N, K).
        num_classes: Number of classes K.

    Returns:
        Macro-averaged AUROC. Returns NaN if computation fails
        (e.g., a class is absent from y_true).
    """
    y_np = y_true.numpy()
    p_np = probs.numpy()

    # Check that at least 2 classes are present
    unique = np.unique(y_np)
    if len(unique) < 2:
        return float("nan")

    try:
        return float(
            roc_auc_score(y_np, p_np, multi_class="ovr", average="macro")
        )
    except ValueError:
        return float("nan")
