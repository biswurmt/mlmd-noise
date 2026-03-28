# Task 2: Asymmetric Noise Injection

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/data/noise.py`
- Create: `lilaw-poc/tests/test_noise.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_noise.py
import numpy as np
import pytest
import torch

from lilaw_poc.data.noise import inject_asymmetric_noise


class TestInjectAsymmetricNoise:
    """Test asymmetric positive→negative label noise injection."""

    def test_only_positive_labels_flipped(self) -> None:
        """No negative label should ever become positive."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 0, 1, 1, 1, 0, 1, 0, 1, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.5, rng=rng)
        neg_indices = (labels == 0).nonzero(as_tuple=True)[0]
        assert (noisy[neg_indices] == 0).all(), "Negative labels must never be flipped"

    def test_flip_rate_zero_no_change(self) -> None:
        """0% flip rate should return identical labels."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.0, rng=rng)
        assert torch.equal(noisy, labels)
        assert mask.sum() == 0

    def test_flip_rate_one_flips_all_positives(self) -> None:
        """100% flip rate should flip every positive to negative."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=1.0, rng=rng)
        assert (noisy == 0).all()
        assert mask.sum() == 3  # 3 positives flipped

    def test_approximate_flip_rate(self) -> None:
        """Over many samples, actual flip rate should approximate target."""
        rng = np.random.default_rng(42)
        labels = torch.ones(10000, dtype=torch.long)
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=0.3, rng=rng)
        actual_rate = mask.sum().item() / labels.sum().item()
        assert abs(actual_rate - 0.3) < 0.03, f"Expected ~0.3, got {actual_rate}"

    def test_returns_flip_mask(self) -> None:
        """Mask should be True exactly where labels were flipped."""
        rng = np.random.default_rng(42)
        labels = torch.tensor([0, 1, 1, 0, 1, 1, 0])
        noisy, mask = inject_asymmetric_noise(labels, flip_rate=1.0, rng=rng)
        expected_mask = torch.tensor([False, True, True, False, True, True, False])
        assert torch.equal(mask, expected_mask)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_noise.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lilaw_poc.data.noise'`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/data/noise.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_noise.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Lint and type check**

Run: `ruff check src/lilaw_poc/data/noise.py && ty check src/lilaw_poc/data/noise.py`

- [ ] **Step 6: Commit**

```bash
git add src/lilaw_poc/data/noise.py tests/test_noise.py
git commit -m "feat(lilaw-poc): add asymmetric noise injection with tests"
```
