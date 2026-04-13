import numpy as np
import pytest
import torch

from lilaw_poc.evaluate import compute_pr_auc, compute_recall_at_ppv


class TestComputePrAuc:
    """Test PR-AUC computation."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions should give PR-AUC close to 1.0."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_scores = torch.tensor([0.1, 0.2, 0.9, 0.95])
        pr_auc = compute_pr_auc(y_true, y_scores)
        assert pr_auc > 0.99

    def test_random_predictions(self) -> None:
        """Random predictions should give PR-AUC roughly equal to prevalence."""
        rng = np.random.default_rng(42)
        y_true = torch.tensor([0] * 900 + [1] * 100)
        y_scores = torch.tensor(rng.random(1000).astype(np.float32))
        pr_auc = compute_pr_auc(y_true, y_scores)
        assert 0.05 < pr_auc < 0.25  # roughly around prevalence (0.1)


class TestComputeRecallAtPpv:
    """Test Recall @ PPV >= threshold."""

    def test_perfect_predictions_at_ppv_80(self) -> None:
        """Perfect model should achieve recall=1.0 at PPV>=0.8."""
        y_true = torch.tensor([0, 0, 1, 1])
        y_scores = torch.tensor([0.1, 0.2, 0.9, 0.95])
        recall = compute_recall_at_ppv(y_true, y_scores, min_ppv=0.80)
        assert recall > 0.99

    def test_returns_zero_when_ppv_unachievable(self) -> None:
        """If PPV threshold can never be met, return 0.0."""
        # All high-scoring samples are negatives, so PPV at any threshold is low
        y_true = torch.tensor([0, 0, 0, 0, 1])
        y_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        recall = compute_recall_at_ppv(y_true, y_scores, min_ppv=0.99)
        assert recall == pytest.approx(0.0)
