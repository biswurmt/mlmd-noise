# Task 7: Training Loop — Baseline

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
