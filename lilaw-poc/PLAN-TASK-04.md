# Task 4: MLP Model (MLMD Mock)

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
