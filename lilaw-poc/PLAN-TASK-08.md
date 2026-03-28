# Task 8: Training Loop — LiLAW

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
