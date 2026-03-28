"""Multi-class training loops: baseline CE and LiLAW-weighted CE for MedMNIST."""

import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as func
from torch import nn
from torch.utils.data import DataLoader

from lilaw_poc.lilaw import LiLAWWeighter


@dataclass
class TrainConfig:
    """Training hyperparameters matching the LiLAW paper's MedMNIST setup."""

    epochs: int = 100
    lr: float = 0.0001
    weight_decay: float = 0.0
    lr_milestones: list[int] = field(default_factory=lambda: [50, 75])
    lr_gamma: float = 0.1
    warmup_epochs: int = 1
    meta_lr: float = 0.005
    meta_wd: float = 0.0001
    alpha_init: float = 10.0
    beta_init: float = 2.0
    delta_init: float = 6.0
    early_stopping_patience: int = 10


@dataclass
class TrainResult:
    """Container for training outputs."""

    model: nn.Module
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    val_accs: list[float] = field(default_factory=list)
    meta_params: list[dict[str, float]] = field(default_factory=list)
    best_epoch: int = 0


def _move_weighter_to_device(weighter: LiLAWWeighter, device: torch.device) -> None:
    """Move LiLAWWeighter's meta-parameters to the specified device."""
    weighter.alpha = torch.tensor(weighter.alpha.item(), device=device, requires_grad=True)
    weighter.beta = torch.tensor(weighter.beta.item(), device=device, requires_grad=True)
    weighter.delta = torch.tensor(weighter.delta.item(), device=device, requires_grad=True)


def _eval_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute Top-1 accuracy on a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            logits = model(images)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / max(total, 1)


def train_baseline_mc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """Train with vanilla CrossEntropyLoss + Adam + MultiStepLR.

    Includes early stopping on validation accuracy.

    Args:
        model: ResNet-18 (or any model) outputting logits.
        train_loader: Training DataLoader (may have noisy labels).
        val_loader: Validation DataLoader (clean labels for early stopping).
        config: Training hyperparameters.
        device: Torch device (cuda or cpu).

    Returns:
        TrainResult with best model restored.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )

    result = TrainResult(model=model)
    best_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        # Training
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()

            optimizer.zero_grad()
            logits = model(images)
            loss = func.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        result.train_losses.append(epoch_loss / max(n_batches, 1))

        # Validation
        val_acc = _eval_accuracy(model, val_loader, device)
        result.val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} (best={result.best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.model = model
    return result


def train_lilaw_mc(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """Train with LiLAW-weighted CrossEntropyLoss.

    Alternates between:
      1. Training update: freeze meta-params, update model on weighted training loss.
      2. Meta update: freeze model, update meta-params on weighted validation loss.

    Args:
        model: ResNet-18 (or any model) outputting logits.
        train_loader: Training DataLoader (may have noisy labels).
        val_loader: Validation DataLoader (clean labels for early stopping).
        config: Training hyperparameters.
        device: Torch device (cuda or cpu).

    Returns:
        TrainResult with best model, loss history, and meta-parameter trajectory.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=config.lr_milestones, gamma=config.lr_gamma
    )
    weighter = LiLAWWeighter(
        alpha_init=config.alpha_init, beta_init=config.beta_init, delta_init=config.delta_init
    )
    _move_weighter_to_device(weighter, device)

    val_iter = iter(val_loader)

    result = TrainResult(model=model)
    best_acc = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        use_lilaw = epoch >= config.warmup_epochs

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            labels = labels.long()

            # --- Step 1: Training update (freeze meta-params, update model) ---
            if use_lilaw:
                weighter.set_requires_grad(requires_grad=False)

            optimizer.zero_grad()
            logits = model(images)
            ce_per_sample = func.cross_entropy(logits, labels, reduction="none")

            if use_lilaw:
                probs = func.softmax(logits.detach(), dim=1)
                s_label = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
                s_max = probs.max(dim=1).values
                _, _, _, w_total = weighter.compute_weights(s_label, s_max)
                loss = (w_total.detach() * ce_per_sample).mean()
            else:
                loss = ce_per_sample.mean()

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

            # --- Step 2: Meta update (freeze model, update meta-params) ---
            if use_lilaw:
                try:
                    val_images, val_labels = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_images, val_labels = next(val_iter)

                val_images, val_labels = val_images.to(device), val_labels.to(device)
                if val_labels.dim() > 1:
                    val_labels = val_labels.squeeze(-1)
                val_labels = val_labels.long()

                weighter.set_requires_grad(requires_grad=True)
                weighter.zero_grad()

                with torch.no_grad():
                    val_logits = model(val_images)

                val_ce = func.cross_entropy(val_logits, val_labels, reduction="none")
                val_probs = func.softmax(val_logits, dim=1)
                s_label_val = val_probs.gather(1, val_labels.unsqueeze(1)).squeeze(1)
                s_max_val = val_probs.max(dim=1).values
                _, _, _, w_val = weighter.compute_weights(s_label_val, s_max_val)
                meta_loss = (w_val * val_ce).mean()
                meta_loss.backward()

                weighter.meta_step(lr=config.meta_lr, wd=config.meta_wd)

        scheduler.step()
        result.train_losses.append(epoch_loss / max(n_batches, 1))
        result.meta_params.append(weighter.state_dict())

        # Validation accuracy (clean labels)
        val_acc = _eval_accuracy(model, val_loader, device)
        result.val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            result.best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.early_stopping_patience:
                print(f"  Early stopping at epoch {epoch} (best={result.best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    result.model = model
    return result
