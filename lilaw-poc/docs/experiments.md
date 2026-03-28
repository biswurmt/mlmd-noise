# Experiments

## Setup

**Datasets**: breast_cancer (sklearn), Adult (UCI), Pima Indians. Split: 55/15/30 train/val/test.
**Noise injection**: Asymmetric positive→negative flips at 10%, 20%, 30%, 40% rates.
**Model**: 5-layer MLP, BCE loss, SGD (lr=1e-4, momentum 0.9).

## Key Results

| Dataset | Noise | Baseline PR-AUC | LiLAW PR-AUC | Δ |
|---------|-------|------------------|---------------|---|
| adult   | 10%   | 0.692 ±0.02     | 0.702 ±0.02  | +1% |
| adult   | 40%   | 0.679 ±0.01     | 0.699 ±0.02  | +3% |
| breast_cancer | 20% | 0.804 ±0.22 | 0.826 ±0.20 | +3% |
| pima    | 10%   | 0.561 ±0.01     | 0.570 ±0.01  | +2% |
| pima    | 30%   | 0.515 ±0.01     | 0.510 ±0.01  | -1% |

Full results across all noise rates are in [DECISIONS.md](DECISIONS.md#d13-poc-sweep-results).

## Analysis

LiLAW provides 1-2% PR-AUC improvement on the Adult and Breast Cancer datasets. Pima, a small dataset (n=768), fails to benefit at noise rates above 20%, likely due to noisy meta-gradient signals.

Compared to the LiLAW paper's reported gains on CIFAR (e.g., +17% accuracy improvement), our findings are modest. The likely explanation is that:

1. Asymmetric noise reduces separability between hard and noisy samples.
2. On `adult`, baseline performance is already near 0.69 PR-AUC, leaving less headroom.

For more details, see the [DECISIONS.md](DECISIONS.md) decision log.
