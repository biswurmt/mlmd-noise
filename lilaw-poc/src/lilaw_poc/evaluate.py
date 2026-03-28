"""Evaluation metrics: PR-AUC and Recall @ PPV >= threshold."""

import numpy as np
import torch
from sklearn.metrics import auc, precision_recall_curve


def compute_pr_auc(y_true: torch.Tensor, y_scores: torch.Tensor) -> float:
    """Compute area under the precision-recall curve.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted probabilities.

    Returns:
        PR-AUC as a float.
    """
    precision, recall, _ = precision_recall_curve(
        y_true.numpy(), y_scores.numpy()
    )
    return float(auc(recall, precision))


def compute_recall_at_ppv(
    y_true: torch.Tensor,
    y_scores: torch.Tensor,
    min_ppv: float = 0.80,
) -> float:
    """Compute maximum recall achievable at or above a PPV (precision) threshold.

    Args:
        y_true: Binary ground truth labels.
        y_scores: Predicted probabilities.
        min_ppv: Minimum precision (PPV) required.

    Returns:
        Maximum recall at PPV >= min_ppv. Returns 0.0 if threshold is never met.
    """
    precision, recall, _ = precision_recall_curve(
        y_true.numpy(), y_scores.numpy()
    )
    # Find all operating points where precision >= min_ppv
    valid = precision >= min_ppv
    if not valid.any():
        return 0.0
    return float(np.max(recall[valid]))
