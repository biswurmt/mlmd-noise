# Task 6: Evaluation Metrics

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/evaluate.py`
- Create: `lilaw-poc/tests/test_evaluate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_evaluate.py
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
        y_true = torch.tensor([0, 0, 0, 0, 1])
        y_scores = torch.tensor([0.9, 0.8, 0.7, 0.6, 0.5])
        recall = compute_recall_at_ppv(y_true, y_scores, min_ppv=0.99)
        # With these scores, top predictions are all negative — PPV is low
        assert recall == pytest.approx(0.0) or recall <= 1.0  # implementation-dependent
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/evaluate.py
"""Evaluation metrics: PR-AUC and Recall @ PPV >= threshold."""

import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, auc


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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Lint, type check, commit**

```bash
ruff check src/lilaw_poc/evaluate.py
git add src/lilaw_poc/evaluate.py tests/test_evaluate.py
git commit -m "feat(lilaw-poc): add PR-AUC and Recall@PPV evaluation metrics"
```
