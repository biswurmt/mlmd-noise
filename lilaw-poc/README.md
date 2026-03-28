# LiLAW PoC

Proof-of-concept implementation of **LiLAW: Lightweight Learnable Adaptive Weighting** for robust learning under asymmetric label noise. Based on [LiLAW: Lightweight Learnable Adaptive Weighting to Claim the Jungle of Noisy Labels](https://arxiv.org/abs/2502.01981).

## Quick Start

```bash
# Install
uv sync

# Run tests
pytest

# Run experiments
uv run python -m lilaw_poc.experiment --help
```

The main entry point is `scripts/run_sweep_runpod.sh`, which sweeps over datasets (Breast Cancer, Adult, Pima Indians) and noise rates (10%-40%).

## Architecture

```
src/lilaw_poc/
├── data/           — noise injection, dataset loaders
├── lilaw.py         — LiLAW weighting functions
├── model.py         — MLMD-mock model (5-layer MLP)
├── train.py         — training loops for baseline and LiLAW
├── experiment.py    — sweep runner
└── evaluate.py      — evaluation metrics
```

### Key Files

| File | Purpose |
|------|---------|
| `lilaw.py` | Weighting functions (Wα, Wβ, Wδ) and meta-parameter container |
| `train.py` | Two training loops: baseline BCE vs. LiLAW-weighted |
| `model.py` | MLMD mock (5-layer MLP, ReLU, sigmoid output) |
| `experiment.py` | Cross-product sweep over datasets × noise rates × seeds |

## Method Overview

LiLAW re-weights per-sample loss based on sample difficulty:

- **Wα (easy samples)**: Weight easy examples using a sigmoid on logit of true label
- **Wβ (hard samples)**: Down-weight hard samples to ignore label noise
- **Wδ (moderate samples)**: RBF kernel centered near decision boundary

Three meta-parameters (α, β, δ) are updated each batch via a meta-gradient on a validation batch. The method does not require a clean validation set.

## Documentation

- **[Design & Decisions](docs/design.md)** → design rationale, noise model, evaluation metrics.
- **[Experiments & Results](docs/experiments.md)** — empirical validation on Breast Cancer, Adult, and Pima datasets.
- **[Theory & Algorithm](docs/theory.md)** — mathematics of LiLAW weight functions, gradient updates.
- [Decision Log](docs/DECISIONS.md) — complete decision history and parameter choices.

## Reference

- LiLAW Paper: "Lightweight Learnable Adaptive Weighting to Claim the Jungle of Noisy Labels" (arXiv:2502.01981)
- MLMD context: see `../presentation-slide/` for the original proposal.
