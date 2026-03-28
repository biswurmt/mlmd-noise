"""MedMNIST experiment runner: sweep over datasets x noise rates x seeds."""

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as func

from lilaw_poc.medmnist.datasets import (
    NoisyLabelDataset,
    get_labels,
    get_medmnist_loaders,
)
from lilaw_poc.medmnist.evaluate import compute_accuracy, compute_auroc
from lilaw_poc.medmnist.model import build_resnet18
from lilaw_poc.medmnist.noise import inject_symmetric_noise
from lilaw_poc.medmnist.train import TrainConfig, train_baseline_mc, train_lilaw_mc


@dataclass
class ExperimentConfig:
    """Configuration for a single MedMNIST experiment run."""

    dataset: str = "bloodmnist"
    noise_rate: float = 0.0
    seed: int = 42
    epochs: int = 100
    batch_size: int = 128


@dataclass
class ExperimentResult:
    """Result of a single MedMNIST experiment run."""

    config: ExperimentConfig
    baseline_acc: float = 0.0
    lilaw_acc: float = 0.0
    baseline_auroc: float = 0.0
    lilaw_auroc: float = 0.0
    final_meta_params: dict[str, float] = field(default_factory=dict)
    baseline_best_epoch: int = 0
    lilaw_best_epoch: int = 0
    elapsed_seconds: float = 0.0


def _evaluate_model(
    model: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate model on test set, returning (accuracy, auroc)."""
    model.eval()
    all_labels = []
    all_logits = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            if labels.dim() > 1:
                labels = labels.squeeze(-1)
            logits = model(images)
            all_labels.append(labels)
            all_logits.append(logits.cpu())

    y_true = torch.cat(all_labels)
    logits = torch.cat(all_logits)
    probs = func.softmax(logits, dim=1)

    acc = compute_accuracy(y_true, logits)
    auroc = compute_auroc(y_true, probs, num_classes)
    return acc, auroc


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """Run one experiment: baseline CE vs LiLAW-CE on a MedMNIST dataset."""
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Seed everything
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    rng = np.random.default_rng(config.seed)

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = get_medmnist_loaders(
        config.dataset, batch_size=config.batch_size
    )

    # Inject symmetric noise into training labels only
    if config.noise_rate > 0.0:
        train_labels = get_labels(train_loader.dataset)
        noisy_labels, flip_mask = inject_symmetric_noise(
            train_labels, config.noise_rate, num_classes, rng
        )
        n_flipped = flip_mask.sum().item()
        eff = n_flipped / len(train_labels)
        print(
            f"  Injected {config.noise_rate:.0%} symmetric noise: "
            f"{n_flipped}/{len(train_labels)} changed ({eff:.1%} effective)"
        )

        noisy_train_ds = NoisyLabelDataset(train_loader.dataset, noisy_labels)
        train_loader = torch.utils.data.DataLoader(
            noisy_train_ds,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=train_loader.num_workers,
            pin_memory=True,
            persistent_workers=train_loader.num_workers > 0,
        )

    train_config = TrainConfig(epochs=config.epochs)

    # Train baseline
    print("  Training baseline CE...")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    baseline_model = build_resnet18(num_classes)
    baseline_result = train_baseline_mc(
        baseline_model, train_loader, val_loader, train_config, device
    )

    # Train LiLAW
    print("  Training LiLAW CE...")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    lilaw_model = build_resnet18(num_classes)
    lilaw_result = train_lilaw_mc(lilaw_model, train_loader, val_loader, train_config, device)

    # Evaluate on clean test set
    bl_acc, bl_auroc = _evaluate_model(baseline_result.model, test_loader, num_classes, device)
    lw_acc, lw_auroc = _evaluate_model(lilaw_result.model, test_loader, num_classes, device)

    elapsed = time.time() - t0
    return ExperimentResult(
        config=config,
        baseline_acc=bl_acc,
        lilaw_acc=lw_acc,
        baseline_auroc=bl_auroc,
        lilaw_auroc=lw_auroc,
        final_meta_params=lilaw_result.meta_params[-1] if lilaw_result.meta_params else {},
        baseline_best_epoch=baseline_result.best_epoch,
        lilaw_best_epoch=lilaw_result.best_epoch,
        elapsed_seconds=elapsed,
    )


def run_sweep(
    datasets: list[str] | None = None,
    noise_rates: list[float] | None = None,
    seeds: list[int] | None = None,
    epochs: int = 100,
    output_dir: str = "results/medmnist",
) -> list[ExperimentResult]:
    """Run full sweep over datasets x noise rates x seeds."""
    if datasets is None:
        datasets = ["bloodmnist"]
    if noise_rates is None:
        noise_rates = [0.0, 0.2, 0.4, 0.6, 0.8]
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
                print(f"\n[{i}/{total}] {dataset} | noise={noise_rate:.0%} | seed={seed}")
                result = run_single_experiment(config)
                results.append(result)
                print(
                    f"  Baseline: acc={result.baseline_acc:.4f} auroc={result.baseline_auroc:.4f} "
                    f"(best@{result.baseline_best_epoch})"
                )
                print(
                    f"  LiLAW:    acc={result.lilaw_acc:.4f} auroc={result.lilaw_auroc:.4f} "
                    f"(best@{result.lilaw_best_epoch})"
                )
                print(f"  Elapsed: {result.elapsed_seconds:.1f}s")

    # Save results as JSON
    results_json = []
    for r in results:
        entry = {
            "dataset": r.config.dataset,
            "noise_rate": r.config.noise_rate,
            "seed": r.config.seed,
            "baseline_acc": r.baseline_acc,
            "lilaw_acc": r.lilaw_acc,
            "baseline_auroc": r.baseline_auroc,
            "lilaw_auroc": r.lilaw_auroc,
            "final_meta_params": r.final_meta_params,
            "baseline_best_epoch": r.baseline_best_epoch,
            "lilaw_best_epoch": r.lilaw_best_epoch,
            "elapsed_seconds": r.elapsed_seconds,
        }
        results_json.append(entry)

    with open(out_path / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    _print_summary(results)

    print(f"\nResults saved to {out_path / 'results.json'}")
    return results


def _print_summary(results: list[ExperimentResult]) -> None:
    """Print aggregated mean +/- std per (dataset, noise_rate)."""
    groups: dict[tuple[str, float], list[ExperimentResult]] = defaultdict(list)
    for r in results:
        groups[(r.config.dataset, r.config.noise_rate)].append(r)

    print("\n=== Summary (mean +/- std over seeds) ===")
    print(
        f"{'Dataset':<14} {'Noise':>5}  {'BL Acc':>12} {'LiLAW Acc':>12} "
        f"{'BL AUROC':>12} {'LiLAW AUROC':>12}"
    )
    print("-" * 75)
    for (dataset, noise_rate), group in sorted(groups.items()):
        bl_acc = np.array([r.baseline_acc for r in group])
        lw_acc = np.array([r.lilaw_acc for r in group])
        bl_auc = np.array([r.baseline_auroc for r in group])
        lw_auc = np.array([r.lilaw_auroc for r in group])
        print(
            f"{dataset:<14} {noise_rate:>5.0%}  "
            f"{bl_acc.mean():.4f}±{bl_acc.std():.4f} "
            f"{lw_acc.mean():.4f}±{lw_acc.std():.4f} "
            f"{bl_auc.mean():.4f}±{bl_auc.std():.4f} "
            f"{lw_auc.mean():.4f}±{lw_auc.std():.4f}"
        )


if __name__ == "__main__":
    run_sweep()
