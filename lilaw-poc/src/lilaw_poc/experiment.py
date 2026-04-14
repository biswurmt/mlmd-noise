"""Experiment runner: sweep over datasets x noise rates x seeds."""

import json
from collections import defaultdict
from dataclasses import dataclass, field
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


def _predict_scores(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Run inference on the model's device and return CPU scores."""
    model_device = next(model.parameters()).device
    return model(x.to(model_device)).squeeze().cpu()


def run_single_experiment(config: ExperimentConfig, device: torch.device | None = None) -> ExperimentResult:
    """Run one experiment: baseline vs LiLAW on a dataset+noise combination."""
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    # Load data
    splits = load_dataset(config.dataset, seed=config.seed)
    x_train, y_train = splits["X_train"], splits["y_train"]
    x_val, y_val = splits["X_val"], splits["y_val"]
    x_test, y_test = splits["X_test"], splits["y_test"]
    input_dim = x_train.shape[1]

    # Inject noise into train and val
    y_train_noisy, _ = inject_asymmetric_noise(y_train, config.noise_rate, rng)
    rng_val = np.random.default_rng(config.seed + 1000)
    y_val_noisy, _ = inject_asymmetric_noise(y_val, config.noise_rate, rng_val)

    # Train baseline
    torch.manual_seed(config.seed)
    dev = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    baseline_result = train_baseline(
        x_train, y_train_noisy, x_val, y_val_noisy,
        input_dim=input_dim, epochs=config.epochs,
        batch_size=config.batch_size, hidden_dim=config.hidden_dim,
        device=dev,
    )

    # Train LiLAW
    torch.manual_seed(config.seed)
    lilaw_result = train_lilaw(
        x_train, y_train_noisy, x_val, y_val_noisy,
        input_dim=input_dim, epochs=config.epochs,
        batch_size=config.batch_size, hidden_dim=config.hidden_dim,
        device=dev,
    )

    # Evaluate on clean test set
    baseline_result.model.eval()
    lilaw_result.model.eval()
    with torch.no_grad():
        baseline_scores = _predict_scores(baseline_result.model, x_test)
        lilaw_scores = _predict_scores(lilaw_result.model, x_test)

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
    device: torch.device | None = None,
) -> list[ExperimentResult]:
    """Run full sweep over datasets x noise rates x seeds."""
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
                result = run_single_experiment(config, device=device)
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

    # Aggregate mean +/- std per (dataset, noise_rate)
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
        f"{'Dataset':<16} {'Noise':>5}  {'BL PR-AUC':>12} {'LiLAW PR-AUC':>14} "
        f"{'BL Rec@PPV80':>14} {'LiLAW Rec@PPV80':>17}"
    )
    print("-" * 85)
    for (dataset, noise_rate), group in sorted(groups.items()):
        bl_auc = np.array([r.baseline_pr_auc for r in group])
        lw_auc = np.array([r.lilaw_pr_auc for r in group])
        bl_rec = np.array([r.baseline_recall_at_ppv80 for r in group])
        lw_rec = np.array([r.lilaw_recall_at_ppv80 for r in group])
        print(
            f"{dataset:<16} {noise_rate:>5.0%}  "
            f"{bl_auc.mean():.4f}+/-{bl_auc.std():.4f} "
            f"{lw_auc.mean():.4f}+/-{lw_auc.std():.4f}   "
            f"{bl_rec.mean():.4f}+/-{bl_rec.std():.4f}   "
            f"{lw_rec.mean():.4f}+/-{lw_rec.std():.4f}"
        )


if __name__ == "__main__":
    run_sweep()
