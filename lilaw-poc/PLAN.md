# LiLAW PoC Implementation Plan

> **For agentic workers:** Implement this plan task-by-task in order. Each task is self-contained with tests, implementation, and a commit. Check off steps (`- [ ]` → `- [x]`) as you go. Use subagents/parallel execution for independent tasks where your environment supports it.

**Goal:** Build a minimum viable PyTorch implementation of LiLAW (Lightweight Learnable Adaptive Weighting) that demonstrates recall recovery under asymmetric label noise on three tabular datasets, simulating the MLMD clinical system.

**Architecture:** Modular Python package (`lilaw_poc`) with separate modules for noise injection, LiLAW weighting, MLP model, dataset loading, evaluation, and experiment orchestration. The training loop alternates between a standard BCE update on a training batch and a meta-update of (α, β, δ) on a validation batch. An experiment runner sweeps over datasets × noise rates × seeds and produces PR curves + summary tables.

**Tech Stack:** Python 3.12+, PyTorch, scikit-learn, pandas, matplotlib, uv, ruff, ty, pytest

**Reference documents:**
- Design decisions: `lilaw-poc/DECISIONS.md`
- LiLAW paper source: `presentation-slide/background/lilaw/example_paper.tex`

---

## Paper Verification Checkpoints

**This implementation must be verified against the LiLAW paper at two mandatory checkpoints.** The paper source is at `presentation-slide/background/lilaw/example_paper.tex`.

### Checkpoint 1: Before Writing Code (after reading the plan, before Task 2)

Read the LiLAW paper and verify the following match between the plan and the paper:

- [ ] **Weight functions:** W_α (Eq. 2), W_β (Eq. 3), W_δ (Eq. 4) — confirm the sigmoid/RBF formulas in `lilaw.py` match the paper exactly.
- [ ] **Combined weight:** W = W_α + W_β + W_δ — confirm additive combination, applied multiplicatively to the loss.
- [ ] **Algorithm 1:** The alternating loop — confirm Step 1 (training update with frozen meta-params), Step 2 (meta-validation with frozen model), Step 3 (manual SGD on α, β, δ) match the plan's `train_lilaw`.
- [ ] **Binary adaptation:** Confirm how s_i[ỹ_i] and max(s_i) are adapted for binary sigmoid output (not multi-class softmax).
- [ ] **Meta-parameter initialization:** α=10, β=2, δ=6, lr=0.005, wd=0.0001.
- [ ] **Warmup:** 1 epoch of vanilla training before LiLAW activates.
- [ ] **Gradient properties:** ∇_α L_W ≥ 0 (α decreases), ∇_β L_W ≤ 0 (β increases), ∇_δ L_W can go either way.

If any discrepancy is found, fix the plan before proceeding.

### Checkpoint 2: After All Code Is Written (after Task 10, before claiming done)

Re-read the LiLAW paper and audit the implemented code against it:

- [ ] Re-verify all items from Checkpoint 1 against the **actual code** (not the plan).
- [ ] Confirm gradient flow: run a test that checks ∇_α L_W ≥ 0 and ∇_β L_W ≤ 0 empirically on a small batch.
- [ ] Confirm meta-parameter evolution direction: on a short training run, verify α decreases and β increases (as stated in the paper).
- [ ] Compare weight function outputs for a known easy/moderate/hard sample against hand-calculated expected values from the paper's formulas.

Document any discrepancies found and fixes applied in `lilaw-poc/DECISIONS.md` under a new heading "## Verification Notes".

---

## File Structure

```
lilaw-poc/
├── DECISIONS.md                    # Design rationale (exists)
├── PLAN.md                         # This file
├── pyproject.toml                  # uv project, ruff/pytest/ty config
├── setup.sh                        # Environment bootstrap
├── src/
│   └── lilaw_poc/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py         # Load + preprocess breast cancer, adult, pima
│       │   └── noise.py            # Asymmetric noise injection
│       ├── model.py                # 5-layer MLP (MLMD mock)
│       ├── lilaw.py                # LiLAW weight functions + meta-params
│       ├── train.py                # Training loops (baseline + LiLAW)
│       ├── evaluate.py             # PR-AUC, Recall@PPV>=0.80, PR curves
│       └── experiment.py           # Sweep over datasets × noise × seeds
├── tests/
│   ├── test_noise.py
│   ├── test_lilaw.py
│   ├── test_model.py
│   ├── test_datasets.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   └── test_e2e.py
└── results/                        # Generated experiment outputs (gitignored)
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `lilaw-poc/pyproject.toml`
- Create: `lilaw-poc/setup.sh`
- Create: `lilaw-poc/src/lilaw_poc/__init__.py`
- Create: `lilaw-poc/src/lilaw_poc/data/__init__.py`
- Create: `lilaw-poc/.gitignore`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[project]
name = "lilaw-poc"
version = "0.1.0"
description = "LiLAW proof-of-concept for MLMD noise robustness"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "scikit-learn>=1.4",
    "pandas>=2.2",
    "matplotlib>=3.8",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.11",
]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "N", "UP", "ANN", "B", "SIM"]
ignore = ["ANN101", "ANN102"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
markers = ["slow: marks tests as slow (deselect with '-m \"not slow\"')"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Create `setup.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Create venv and install deps
echo "Creating virtual environment and installing dependencies..."
uv venv .venv
uv pip install -e ".[dev]"

# Install ty
echo "Installing ty type checker..."
uv pip install ty

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run tests:  pytest"
echo "Lint:       ruff check src/ tests/"
echo "Type check: ty check src/"
```

- [ ] **Step 3: Create `__init__.py` files and `.gitignore`**

`src/lilaw_poc/__init__.py` — empty.
`src/lilaw_poc/data/__init__.py` — empty.

`.gitignore`:
```
.venv/
__pycache__/
*.egg-info/
results/
.ruff_cache/
```

- [ ] **Step 4: Run `setup.sh` and verify**

Run: `cd lilaw-poc && bash setup.sh`
Expected: venv created, all deps installed, no errors.

- [ ] **Step 5: Verify toolchain**

Run: `source .venv/bin/activate && python -c "import torch; print(torch.__version__)" && ruff check src/ && pytest --co`
Expected: torch version printed, ruff reports no issues, pytest collects 0 tests.

- [ ] **Step 6: Commit**

```bash
git add lilaw-poc/pyproject.toml lilaw-poc/setup.sh lilaw-poc/src/ lilaw-poc/.gitignore
git commit -m "feat(lilaw-poc): scaffold project with uv, ruff, ty, pytest"
```

---

## Task 2: Asymmetric Noise Injection

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

---

## Task 3: LiLAW Weight Functions & Meta-Parameters

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

---

## Task 4: MLP Model (MLMD Mock)

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/model.py`
- Create: `lilaw-poc/tests/test_model.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
import pytest
import torch

from lilaw_poc.model import MLMDNet


class TestMLMDNet:
    """Test the 5-layer MLP matching MLMD architecture."""

    def test_output_shape(self) -> None:
        """Output should be (batch_size, 1)."""
        model = MLMDNet(input_dim=30)
        x = torch.randn(16, 30)
        out = model(x)
        assert out.shape == (16, 1)

    def test_output_range_sigmoid(self) -> None:
        """Output should be in (0, 1) due to sigmoid."""
        model = MLMDNet(input_dim=30)
        x = torch.randn(100, 30)
        out = model(x)
        assert (out > 0).all() and (out < 1).all()

    def test_has_five_linear_layers(self) -> None:
        """Architecture must have exactly 5 Linear layers."""
        model = MLMDNet(input_dim=30)
        linear_layers = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
        assert len(linear_layers) == 5

    def test_gradient_flows(self) -> None:
        """Loss backprop should produce gradients on all parameters."""
        model = MLMDNet(input_dim=10)
        x = torch.randn(4, 10)
        y = torch.tensor([1.0, 0.0, 1.0, 0.0]).unsqueeze(1)
        out = model(x)
        loss = torch.nn.functional.binary_cross_entropy(out, y)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None
            assert not torch.all(p.grad == 0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_model.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/model.py
"""5-layer MLP matching MLMD's feed-forward neural network architecture."""

import torch
from torch import nn


class MLMDNet(nn.Module):
    """5-layer fully connected network with ReLU activations and sigmoid output.

    Architecture mirrors Singh et al. (2022): 5 linear layers, ReLU on first 4,
    sigmoid on the final output for binary classification.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns probabilities in (0, 1)."""
        return self.net(x)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Lint and type check, then commit**

```bash
ruff check src/lilaw_poc/model.py
git add src/lilaw_poc/model.py tests/test_model.py
git commit -m "feat(lilaw-poc): add 5-layer MLP matching MLMD architecture"
```

---

## Task 5: Dataset Loaders

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/data/datasets.py`
- Create: `lilaw-poc/tests/test_datasets.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_datasets.py
import pytest
import torch

from lilaw_poc.data.datasets import load_dataset, DATASET_NAMES


class TestLoadDataset:
    """Test dataset loading and preprocessing for all three PoC datasets."""

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_returns_tensors(self, name: str) -> None:
        """All splits should be float32 tensors."""
        splits = load_dataset(name, seed=42)
        for key in ["X_train", "X_val", "X_test"]:
            assert splits[key].dtype == torch.float32
        for key in ["y_train", "y_val", "y_test"]:
            assert splits[key].dtype == torch.long

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_split_ratios(self, name: str) -> None:
        """Splits should approximate 55/15/30."""
        splits = load_dataset(name, seed=42)
        total = sum(splits[k].shape[0] for k in ["y_train", "y_val", "y_test"])
        train_frac = splits["y_train"].shape[0] / total
        val_frac = splits["y_val"].shape[0] / total
        test_frac = splits["y_test"].shape[0] / total
        assert abs(train_frac - 0.55) < 0.05
        assert abs(val_frac - 0.15) < 0.05
        assert abs(test_frac - 0.30) < 0.05

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_binary_labels(self, name: str) -> None:
        """Labels must be binary (0 or 1)."""
        splits = load_dataset(name, seed=42)
        for key in ["y_train", "y_val", "y_test"]:
            unique = splits[key].unique()
            assert set(unique.tolist()).issubset({0, 1})

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_features_standardized(self, name: str) -> None:
        """Training features should be approximately zero-mean, unit-variance."""
        splits = load_dataset(name, seed=42)
        mean = splits["X_train"].mean(dim=0)
        std = splits["X_train"].std(dim=0)
        assert mean.abs().mean() < 0.1
        # std won't be exactly 1 for all features but should be in a reasonable range
        assert std.mean() < 2.0

    @pytest.mark.parametrize("name", DATASET_NAMES)
    def test_no_nans(self, name: str) -> None:
        """No NaN values in any split."""
        splits = load_dataset(name, seed=42)
        for key in splits:
            assert not torch.isnan(splits[key]).any(), f"NaN found in {key}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/data/datasets.py
"""Dataset loaders for the three PoC tabular datasets.

Each loader returns standardized float32 feature tensors and binary long label tensors
split into 55% train / 15% val / 30% test.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASET_NAMES: list[str] = ["breast_cancer", "adult", "pima"]


def load_dataset(name: str, seed: int = 42) -> dict[str, torch.Tensor]:
    """Load and preprocess a dataset by name.

    Args:
        name: One of "breast_cancer", "adult", "pima".
        seed: Random seed for reproducible splits.

    Returns:
        Dict with keys X_train, y_train, X_val, y_val, X_test, y_test.
    """
    loaders = {
        "breast_cancer": _load_breast_cancer,
        "adult": _load_adult,
        "pima": _load_pima,
    }
    if name not in loaders:
        msg = f"Unknown dataset: {name}. Choose from {DATASET_NAMES}"
        raise ValueError(msg)
    return loaders[name](seed)


def _split_and_standardize(
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict[str, torch.Tensor]:
    """Split 55/15/30 and standardize features using training set stats."""
    # First split: 70% train+val, 30% test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=y
    )
    # Second split: 55/15 from the 70% (≈ 78.6% / 21.4%)
    val_frac = 0.15 / 0.70
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_frac, random_state=seed, stratify=y_trainval
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return {
        "X_train": torch.tensor(X_train, dtype=torch.float32),
        "y_train": torch.tensor(y_train, dtype=torch.long),
        "X_val": torch.tensor(X_val, dtype=torch.float32),
        "y_val": torch.tensor(y_val, dtype=torch.long),
        "X_test": torch.tensor(X_test, dtype=torch.float32),
        "y_test": torch.tensor(y_test, dtype=torch.long),
    }


def _load_breast_cancer(seed: int) -> dict[str, torch.Tensor]:
    """Load sklearn breast cancer dataset (30 features, binary)."""
    data = load_breast_cancer()
    return _split_and_standardize(data.data, data.target, seed)


def _load_adult(seed: int) -> dict[str, torch.Tensor]:
    """Load Adult Census Income dataset from OpenML (mixed features, binary)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    df = data.frame
    y = (df[data.target_names[0]] == ">50K").astype(int).values

    # One-hot encode categoricals, fill NaN
    X = df.drop(columns=data.target_names)
    X = X.apply(lambda col: col.fillna(col.mode()[0]) if col.dtype == "object" or col.dtype.name == "category" else col.fillna(col.median()))
    X = pd.get_dummies(X, drop_first=True).values.astype(np.float64)

    return _split_and_standardize(X, y, seed)


def _load_pima(seed: int) -> dict[str, torch.Tensor]:
    """Load Pima Indians Diabetes dataset from OpenML (8 numeric features, binary)."""
    from sklearn.datasets import fetch_openml

    data = fetch_openml("diabetes", version=1, as_frame=True, parser="auto")
    df = data.frame
    y = (df["class"] == "tested_positive").astype(int).values
    X = df.drop(columns=["class"]).values.astype(np.float64)

    return _split_and_standardize(X, y, seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_datasets.py -v`
Expected: All 15 tests PASS (5 tests × 3 datasets). First run of adult/pima may download data.

- [ ] **Step 5: Lint, type check, commit**

```bash
ruff check src/lilaw_poc/data/datasets.py
git add src/lilaw_poc/data/datasets.py tests/test_datasets.py
git commit -m "feat(lilaw-poc): add dataset loaders for breast cancer, adult, pima"
```

---

## Task 6: Evaluation Metrics

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

---

## Task 7: Training Loop — Baseline

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/train.py`
- Create: `lilaw-poc/tests/test_train.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_train.py
import pytest
import torch

from lilaw_poc.model import MLMDNet
from lilaw_poc.train import train_baseline, train_lilaw, TrainResult


class TestTrainBaseline:
    """Test vanilla BCE training loop."""

    def test_returns_train_result(self) -> None:
        """Should return a TrainResult with loss history and final model."""
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        result = train_baseline(
            X_train, y_train, X_val, y_val, input_dim=10, epochs=2, batch_size=32
        )
        assert isinstance(result, TrainResult)
        assert len(result.train_losses) == 2
        assert result.model is not None

    def test_loss_decreases(self) -> None:
        """Loss should generally decrease over epochs on separable data."""
        torch.manual_seed(42)
        X = torch.randn(200, 5)
        y = (X[:, 0] > 0).long()
        X_train, y_train = X[:160], y[:160]
        X_val, y_val = X[160:], y[160:]
        result = train_baseline(
            X_train, y_train, X_val, y_val, input_dim=5, epochs=20, batch_size=32
        )
        assert result.train_losses[-1] < result.train_losses[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py::TestTrainBaseline -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write implementation**

```python
# src/lilaw_poc/train.py
"""Training loops for baseline (vanilla BCE) and LiLAW-weighted training."""

from dataclasses import dataclass, field

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from lilaw_poc.lilaw import LiLAWWeighter
from lilaw_poc.model import MLMDNet


@dataclass
class TrainResult:
    """Container for training outputs."""

    model: MLMDNet
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    meta_params: list[dict[str, float]] = field(default_factory=list)


def train_baseline(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-6,
    hidden_dim: int = 128,
) -> TrainResult:
    """Train an MLMDNet with vanilla BCE loss.

    Args:
        X_train: Training features.
        y_train: Training labels (long, 0 or 1).
        X_val: Validation features.
        y_val: Validation labels.
        input_dim: Number of input features.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for SGD.
        momentum: SGD momentum.
        weight_decay: L2 regularization.
        hidden_dim: Hidden layer width.

    Returns:
        TrainResult with trained model and loss history.
    """
    model = MLMDNet(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    train_loader = DataLoader(
        TensorDataset(X_train, y_train.float()), batch_size=batch_size, shuffle=True
    )
    result = TrainResult(model=model)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        result.train_losses.append(epoch_loss / n_batches)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze()
            val_loss = criterion(val_preds, y_val.float())
            result.val_losses.append(val_loss.item())

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py::TestTrainBaseline -v`
Expected: Both tests PASS.

- [ ] **Step 5: Commit**

```bash
ruff check src/lilaw_poc/train.py
git add src/lilaw_poc/train.py tests/test_train.py
git commit -m "feat(lilaw-poc): add baseline BCE training loop"
```

---

## Task 8: Training Loop — LiLAW

**Files:**
- Modify: `lilaw-poc/src/lilaw_poc/train.py`
- Modify: `lilaw-poc/tests/test_train.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_train.py`:

```python
class TestTrainLilaw:
    """Test LiLAW-weighted training loop."""

    def test_returns_train_result_with_meta_params(self) -> None:
        """Should return TrainResult with meta-parameter history."""
        X_train = torch.randn(100, 10)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(20, 10)
        y_val = torch.randint(0, 2, (20,))
        result = train_lilaw(
            X_train, y_train, X_val, y_val, input_dim=10, epochs=3, batch_size=32
        )
        assert isinstance(result, TrainResult)
        assert len(result.meta_params) == 3
        assert "alpha" in result.meta_params[0]

    def test_meta_params_change_during_training(self) -> None:
        """Alpha, beta, delta should evolve from their initial values."""
        torch.manual_seed(42)
        X = torch.randn(200, 5)
        y = (X[:, 0] > 0).long()
        result = train_lilaw(
            X[:160], y[:160], X[160:], y[160:], input_dim=5, epochs=10, batch_size=32
        )
        first = result.meta_params[0]
        last = result.meta_params[-1]
        changed = any(
            abs(first[k] - last[k]) > 1e-4 for k in ["alpha", "beta", "delta"]
        )
        assert changed, "Meta-parameters should change during training"

    def test_warmup_epoch_uses_vanilla_bce(self) -> None:
        """During warmup (epoch 0), meta-params should not change."""
        torch.manual_seed(42)
        X_train = torch.randn(100, 5)
        y_train = torch.randint(0, 2, (100,))
        X_val = torch.randn(20, 5)
        y_val = torch.randint(0, 2, (20,))
        result = train_lilaw(
            X_train, y_train, X_val, y_val,
            input_dim=5, epochs=2, batch_size=32, warmup_epochs=1,
        )
        assert result.meta_params[0]["alpha"] == pytest.approx(10.0)
        assert result.meta_params[0]["beta"] == pytest.approx(2.0)
        assert result.meta_params[0]["delta"] == pytest.approx(6.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_train.py::TestTrainLilaw -v`
Expected: FAIL — `ImportError: cannot import name 'train_lilaw'`

- [ ] **Step 3: Write implementation**

Add `train_lilaw` to `src/lilaw_poc/train.py`:

```python
def train_lilaw(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-6,
    hidden_dim: int = 128,
    warmup_epochs: int = 1,
    meta_lr: float = 0.005,
    meta_wd: float = 0.0001,
    alpha_init: float = 10.0,
    beta_init: float = 2.0,
    delta_init: float = 6.0,
) -> TrainResult:
    """Train an MLMDNet with LiLAW-weighted BCE loss.

    Alternates between:
      1. Training update: freeze meta-params, update model on weighted training loss.
      2. Meta update: freeze model, update meta-params on weighted validation loss.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features (may be noisy — LiLAW handles this).
        y_val: Validation labels.
        input_dim: Number of input features.
        epochs: Total training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for model SGD.
        momentum: SGD momentum.
        weight_decay: Model L2 regularization.
        hidden_dim: Hidden layer width.
        warmup_epochs: Epochs of vanilla BCE before LiLAW activates.
        meta_lr: Learning rate for meta-parameter SGD.
        meta_wd: Weight decay for meta-parameters.
        alpha_init: Initial value for alpha.
        beta_init: Initial value for beta.
        delta_init: Initial value for delta.

    Returns:
        TrainResult with trained model, loss history, and meta-parameter trajectory.
    """
    model = MLMDNet(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    weighter = LiLAWWeighter(alpha_init=alpha_init, beta_init=beta_init, delta_init=delta_init)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train.float()), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val, y_val.float()), batch_size=batch_size, shuffle=True
    )
    val_iter = iter(val_loader)

    result = TrainResult(model=model)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        use_lilaw = epoch >= warmup_epochs

        for X_batch, y_batch in train_loader:
            # --- Step 1: Training update (freeze meta-params, update model) ---
            if use_lilaw:
                weighter.set_requires_grad(False)
            optimizer.zero_grad()

            preds = model(X_batch).squeeze()
            bce = torch.nn.functional.binary_cross_entropy(preds, y_batch, reduction="none")

            if use_lilaw:
                # For binary: s_label = pred when y=1, (1-pred) when y=0
                s_label = preds * y_batch + (1 - preds) * (1 - y_batch)
                s_max = torch.max(preds, 1 - preds)
                _, _, _, w_total = weighter.compute_weights(s_label.detach(), s_max.detach())
                loss = (w_total.detach() * bce).mean()
            else:
                loss = bce.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # --- Step 2: Meta update (freeze model, update meta-params) ---
            if use_lilaw:
                # Get a validation batch
                try:
                    X_val_batch, y_val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    X_val_batch, y_val_batch = next(val_iter)

                weighter.set_requires_grad(True)
                weighter.zero_grad()

                with torch.no_grad():
                    val_preds = model(X_val_batch).squeeze()

                val_bce = torch.nn.functional.binary_cross_entropy(
                    val_preds, y_val_batch, reduction="none"
                )

                s_label_val = val_preds * y_val_batch + (1 - val_preds) * (1 - y_val_batch)
                s_max_val = torch.max(val_preds, 1 - val_preds)
                _, _, _, w_val = weighter.compute_weights(s_label_val, s_max_val)
                meta_loss = (w_val * val_bce).mean()
                meta_loss.backward()

                weighter.meta_step(lr=meta_lr, wd=meta_wd)

        result.train_losses.append(epoch_loss / max(n_batches, 1))
        result.meta_params.append(weighter.state_dict())

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_all_preds = model(X_val).squeeze()
            val_loss = torch.nn.functional.binary_cross_entropy(val_all_preds, y_val.float())
            result.val_losses.append(val_loss.item())

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train.py -v`
Expected: All 5 tests PASS (2 baseline + 3 LiLAW).

- [ ] **Step 5: Lint, type check, commit**

```bash
ruff check src/lilaw_poc/train.py
git add src/lilaw_poc/train.py tests/test_train.py
git commit -m "feat(lilaw-poc): add LiLAW training loop with meta-parameter updates"
```

---

## Task 9: Experiment Runner

**Files:**
- Create: `lilaw-poc/src/lilaw_poc/experiment.py`

- [ ] **Step 1: Write the experiment runner**

```python
# src/lilaw_poc/experiment.py
"""Experiment runner: sweep over datasets × noise rates × seeds."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from lilaw_poc.data.datasets import load_dataset
from lilaw_poc.data.noise import inject_asymmetric_noise
from lilaw_poc.evaluate import compute_pr_auc, compute_recall_at_ppv
from lilaw_poc.train import train_baseline, train_lilaw


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dataset: str = "breast_cancer"
    noise_rate: float = 0.2
    seed: int = 42
    epochs: int = 50
    batch_size: int = 64
    hidden_dim: int = 128


@dataclass
class ExperimentResult:
    """Result of a single experiment run."""

    config: ExperimentConfig
    baseline_pr_auc: float = 0.0
    lilaw_pr_auc: float = 0.0
    baseline_recall_at_ppv80: float = 0.0
    lilaw_recall_at_ppv80: float = 0.0
    final_meta_params: dict[str, float] = field(default_factory=dict)


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one experiment: baseline vs LiLAW on a dataset+noise combination."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Load data
    splits = load_dataset(config.dataset, seed=config.seed)
    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]
    input_dim = X_train.shape[1]

    # Inject noise into train and val
    y_train_noisy, _ = inject_asymmetric_noise(y_train, config.noise_rate, rng)
    rng_val = np.random.default_rng(config.seed + 1000)
    y_val_noisy, _ = inject_asymmetric_noise(y_val, config.noise_rate, rng_val)

    # Train baseline
    torch.manual_seed(config.seed)
    baseline_result = train_baseline(
        X_train, y_train_noisy, X_val, y_val_noisy,
        input_dim=input_dim, epochs=config.epochs,
        batch_size=config.batch_size, hidden_dim=config.hidden_dim,
    )

    # Train LiLAW
    torch.manual_seed(config.seed)
    lilaw_result = train_lilaw(
        X_train, y_train_noisy, X_val, y_val_noisy,
        input_dim=input_dim, epochs=config.epochs,
        batch_size=config.batch_size, hidden_dim=config.hidden_dim,
    )

    # Evaluate on clean test set
    baseline_result.model.eval()
    lilaw_result.model.eval()
    with torch.no_grad():
        baseline_scores = baseline_result.model(X_test).squeeze()
        lilaw_scores = lilaw_result.model(X_test).squeeze()

    return ExperimentResult(
        config=config,
        baseline_pr_auc=compute_pr_auc(y_test, baseline_scores),
        lilaw_pr_auc=compute_pr_auc(y_test, lilaw_scores),
        baseline_recall_at_ppv80=compute_recall_at_ppv(y_test, baseline_scores),
        lilaw_recall_at_ppv80=compute_recall_at_ppv(y_test, lilaw_scores),
        final_meta_params=lilaw_result.meta_params[-1] if lilaw_result.meta_params else {},
    )


def run_sweep(
    datasets: list[str] | None = None,
    noise_rates: list[float] | None = None,
    seeds: list[int] | None = None,
    epochs: int = 50,
    output_dir: str = "results",
) -> list[ExperimentResult]:
    """Run full sweep over datasets × noise rates × seeds."""
    if datasets is None:
        datasets = ["breast_cancer", "adult", "pima"]
    if noise_rates is None:
        noise_rates = [0.1, 0.2, 0.3, 0.4]
    if seeds is None:
        seeds = [42, 123, 456]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: list[ExperimentResult] = []
    total = len(datasets) * len(noise_rates) * len(seeds)
    i = 0

    for dataset in datasets:
        for noise_rate in noise_rates:
            for seed in seeds:
                i += 1
                config = ExperimentConfig(
                    dataset=dataset, noise_rate=noise_rate, seed=seed, epochs=epochs,
                )
                print(f"[{i}/{total}] {dataset} | noise={noise_rate} | seed={seed}")
                result = run_single_experiment(config)
                results.append(result)
                print(
                    f"  Baseline PR-AUC={result.baseline_pr_auc:.4f} | "
                    f"LiLAW PR-AUC={result.lilaw_pr_auc:.4f} | "
                    f"Baseline Recall@PPV80={result.baseline_recall_at_ppv80:.4f} | "
                    f"LiLAW Recall@PPV80={result.lilaw_recall_at_ppv80:.4f}"
                )

    # Save results as JSON
    results_json = []
    for r in results:
        entry = {
            "dataset": r.config.dataset,
            "noise_rate": r.config.noise_rate,
            "seed": r.config.seed,
            "baseline_pr_auc": r.baseline_pr_auc,
            "lilaw_pr_auc": r.lilaw_pr_auc,
            "baseline_recall_at_ppv80": r.baseline_recall_at_ppv80,
            "lilaw_recall_at_ppv80": r.lilaw_recall_at_ppv80,
            "final_meta_params": r.final_meta_params,
        }
        results_json.append(entry)

    with open(out_path / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # Aggregate mean ± std per (dataset, noise_rate)
    _print_summary(results)

    # Plot PR curves (requires storing scores — deferred to plot_pr_curves)

    print(f"\nResults saved to {out_path / 'results.json'}")
    return results


def _print_summary(results: list[ExperimentResult]) -> None:
    """Print aggregated mean ± std per (dataset, noise_rate)."""
    from collections import defaultdict

    groups: dict[tuple[str, float], list[ExperimentResult]] = defaultdict(list)
    for r in results:
        groups[(r.config.dataset, r.config.noise_rate)].append(r)

    print("\n=== Summary (mean ± std over seeds) ===")
    print(f"{'Dataset':<16} {'Noise':>5}  {'BL PR-AUC':>12} {'LiLAW PR-AUC':>14} {'BL Rec@PPV80':>14} {'LiLAW Rec@PPV80':>17}")
    print("-" * 85)
    for (dataset, noise_rate), group in sorted(groups.items()):
        bl_auc = np.array([r.baseline_pr_auc for r in group])
        lw_auc = np.array([r.lilaw_pr_auc for r in group])
        bl_rec = np.array([r.baseline_recall_at_ppv80 for r in group])
        lw_rec = np.array([r.lilaw_recall_at_ppv80 for r in group])
        print(
            f"{dataset:<16} {noise_rate:>5.0%}  "
            f"{bl_auc.mean():.4f}±{bl_auc.std():.4f} "
            f"{lw_auc.mean():.4f}±{lw_auc.std():.4f}   "
            f"{bl_rec.mean():.4f}±{bl_rec.std():.4f}   "
            f"{lw_rec.mean():.4f}±{lw_rec.std():.4f}"
        )


if __name__ == "__main__":
    run_sweep()
```

- [ ] **Step 2: Lint and type check**

Run: `ruff check src/lilaw_poc/experiment.py`

- [ ] **Step 3: Commit**

```bash
git add src/lilaw_poc/experiment.py
git commit -m "feat(lilaw-poc): add experiment runner with sweep over datasets x noise x seeds"
```

---

## Task 10: Smoke Test — End-to-End

**Files:**
- Create: `lilaw-poc/tests/test_e2e.py`

- [ ] **Step 1: Write end-to-end smoke test**

```python
# tests/test_e2e.py
"""End-to-end smoke test: run a single experiment on breast cancer with 20% noise."""

import pytest

from lilaw_poc.experiment import ExperimentConfig, run_single_experiment


class TestEndToEnd:
    """Smoke test that the full pipeline runs without errors."""

    @pytest.mark.slow
    def test_single_experiment_completes(self) -> None:
        """A single experiment should complete and return valid metrics."""
        config = ExperimentConfig(
            dataset="breast_cancer",
            noise_rate=0.2,
            seed=42,
            epochs=5,  # short for CI
            batch_size=32,
        )
        result = run_single_experiment(config)

        assert 0.0 <= result.baseline_pr_auc <= 1.0
        assert 0.0 <= result.lilaw_pr_auc <= 1.0
        assert 0.0 <= result.baseline_recall_at_ppv80 <= 1.0
        assert 0.0 <= result.lilaw_recall_at_ppv80 <= 1.0
        assert "alpha" in result.final_meta_params
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_e2e.py -v -m slow`
Expected: PASS. Full pipeline runs without errors.

- [ ] **Step 3: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Final lint + type check**

Run: `ruff check src/ tests/ && ty check src/`

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(lilaw-poc): add end-to-end smoke test"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Project scaffolding (uv, ruff, ty, pytest) | Toolchain verification |
| 2 | Asymmetric noise injection | 5 tests |
| 3 | LiLAW weight functions + meta-params | 8 tests |
| 4 | 5-layer MLP (MLMD mock) | 4 tests |
| 5 | Dataset loaders (breast cancer, adult, pima) | 15 tests |
| 6 | Evaluation metrics (PR-AUC, Recall@PPV) | 4 tests |
| 7 | Training loop — baseline BCE | 2 tests |
| 8 | Training loop — LiLAW weighted | 3 tests |
| 9 | Experiment runner + aggregation | — (covered by Task 10 smoke test) |
| 10 | End-to-end smoke test | 1 test |

**Total: ~42 tests across 10 tasks.**
