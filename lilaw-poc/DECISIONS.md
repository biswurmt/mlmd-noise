# LiLAW PoC — Design & Method Decisions

This document captures all experimental, method, and technical decisions for the LiLAW proof-of-concept. It is the single source of truth for design rationale.

---

## Method: LiLAW Overview

Three learnable scalars (α, β, δ) weight per-sample loss by difficulty category (easy/moderate/hard) using sigmoid and RBF functions. Meta-updates on a validation batch follow each training batch — one extra forward+backward pass. No clean validation set required.

- **W_α** (easy samples): `σ(α·s[ỹ] - max(s))`. Gradient always non-negative → α decreases during training.
- **W_β** (hard samples): `σ(-(β·s[ỹ] - max(s)))`. Gradient always non-positive → β increases during training.
- **W_δ** (moderate samples): `exp(-(δ·s[ỹ] - max(s))² / 2)`. Gradient sign depends on whether `δ·s[ỹ] ≥ max(s)`.
- **Combined weight**: `W = W_α + W_β + W_δ`, applied multiplicatively to the base loss.

---

## Decisions

### D1: Noise Model — Asymmetric Positive→Negative Flip

We inject noise only in the positive→negative direction to simulate the real EHR problem: a test was medically necessary but performed at an external institution (unobserved), so the label appears negative. Flip rates: **10%, 20%, 30%, 40%**. Noise is injected into train+val splits only; test set remains clean.

### D2: Meta-Parameter Initialization

Start with paper defaults: α=10, β=2, δ=6. Meta-learning rates: lr=0.005, wd=0.0001 for all three. These were tuned for symmetric noise on image data; we accept them as-is for the PoC and may ablate later.

### D3: Base Model (MLMD Mock)

5-layer MLP with ReLU activations and sigmoid output, matching MLMD's architecture. BCE loss. SGD optimizer (lr=1e-4, momentum=0.9, wd=1e-6).

### D4: Evaluation Protocol

- **Clean held-out test set** — no noise injected, ground truth labels.
- **Primary metric: PR-AUC** — threshold-free ranking quality, robust to class imbalance from noise.
- **Secondary metric: Recall @ PPV ≥ 0.80** — "at MLMD's safety floor, how many necessary tests does LiLAW recover vs baseline?"
- **Diagnostic: Full PR curve** per (noise_rate, method) for visual comparison.
- **Not AUROC** — inflated by true negatives under class imbalance.
- **Not F1-optimal threshold** — F1 weights precision/recall equally, but clinically precision has a hard floor while recall is maximized above it.

### D5: PoC Datasets (Progressive Complexity)

1. `sklearn.datasets.load_breast_cancer()` — hyper-clean baseline, easily saturated by small MLP.
2. Adult Census Income — structural proxy with categorical/continuous feature mix matching triage data.
3. Pima Indians Diabetes — clean numeric clinical dataset predicting diabetes onset. **Alternative for later:** Diabetes 130-US Hospitals (real-world clinical data, messier preprocessing with ICD-9 codes and categorical features) if we need a harder proxy.

### D6: Comparison Protocol

For each (dataset, noise_rate), compare vanilla BCE baseline vs LiLAW-weighted BCE. Both use identical model architecture and hyperparameters.

### D7: Data Splitting

55% train / 15% val / 30% test, matching MLMD's split ratio. Val set receives noise injection (LiLAW does not require a clean val set). Test set stays clean.

### D8: Warmup

1-epoch warmup with vanilla BCE (no LiLAW weighting) before activating the meta-learning loop, per the paper default. **Alternative for later experiments:** longer warmup periods if initial results underperform, on the hypothesis that asymmetric noise may need more epochs for the model to establish a baseline loss landscape before meta-parameters begin adapting.

### D9: Reproducibility

3 random seeds per experiment, reporting mean ± std.

### D10: Engineering Guardrails

- **Package management:** `uv`
- **Formatting/linting:** `ruff`
- **Type checking:** `ty` (Astral)
- **Testing:** `pytest` — core tests for gradient flow through LiLAW loss, weight function bounds, noise injection correctness
