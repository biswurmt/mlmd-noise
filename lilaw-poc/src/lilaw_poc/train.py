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
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    input_dim: int,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-4,
    momentum: float = 0.9,
    weight_decay: float = 1e-6,
    hidden_dim: int = 128,,
    device: torch.device | None = None,,
    device: torch.device | None = None,
) -> TrainResult:
    """Train an MLMDNet with vanilla BCE loss.

    Args:
        x_train: Training features.
        y_train: Training labels (long, 0 or 1).
        x_val: Validation features.
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
    device = device if device is not None else torch.device("cpu")
    device = device if device is not None else torch.device("cpu")
    model = MLMDNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.BCELoss()

    # Move data to device once
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train.float()), batch_size=batch_size, shuffle=True
    )
    result = TrainResult(model=model)

    for _epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(x_batch).squeeze()
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        result.train_losses.append(epoch_loss / n_batches)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_preds = model(x_val).squeeze()
            val_loss = criterion(val_preds, y_val.float())
            result.val_losses.append(val_loss.item())

    return result


def train_lilaw(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
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
    delta_init: float = 6.0,,
    device: torch.device | None = None,
) -> TrainResult:
    """Train an MLMDNet with LiLAW-weighted BCE loss.

    Alternates between:
      1. Training update: freeze meta-params, update model on weighted training loss.
      2. Meta update: freeze model, update meta-params on weighted validation loss.

    Args:
        x_train: Training features.
        y_train: Training labels.
        x_val: Validation features (may be noisy — LiLAW handles this).
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
    device = device if device is not None else torch.device("cpu")
    device = device if device is not None else torch.device("cpu")
    model = MLMDNet(input_dim=input_dim, hidden_dim=hidden_dim).to(device).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    weighter = LiLAWWeighter(
        alpha_init=alpha_init, beta_init=beta_init, delta_init=delta_init
    )

    # Move data to device once
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # Move data to device once
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train.float()), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(x_val, y_val.float()), batch_size=batch_size, shuffle=True
    )
    val_iter = iter(val_loader)

    result = TrainResult(model=model)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        use_lilaw = epoch >= warmup_epochs

        for x_batch, y_batch in train_loader:
            # --- Step 1: Training update (freeze meta-params, update model) ---
            if use_lilaw:
                weighter.set_requires_grad(requires_grad=False)
            optimizer.zero_grad()

            preds = model(x_batch).squeeze()
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
                    x_val_batch, y_val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    x_val_batch, y_val_batch = next(val_iter)

                weighter.set_requires_grad(requires_grad=True)
                weighter.zero_grad()

                with torch.no_grad():
                    val_preds = model(x_val_batch).squeeze()

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
            val_all_preds = model(x_val).squeeze()
            val_loss = torch.nn.functional.binary_cross_entropy(val_all_preds, y_val.float())
            result.val_losses.append(val_loss.item())

    return result
