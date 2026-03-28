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

### D10: Training Step Weight Detachment

In the training step, `w_total.detach()` is applied before weighting the per-sample BCE loss. This means the model sees constant per-sample weights and cannot learn to shape its predictions in a way that accounts for how the LiLAW weight function responds. This matches Algorithm 1 from the paper: meta-parameters are frozen during the training update, and the model treats the current weights as fixed importance-sampling coefficients. The alternative (keeping `w_total` differentiable) would let the model's gradient be scaled by the weight function's curvature, potentially improving moderate-sample handling via the RBF shape. This is a candidate ablation for future experiments.

### D11: Verification — Gradient Sign Properties

Empirically verified (via `test_lilaw.py::test_alpha_gradient_non_negative` and `test_beta_gradient_non_positive`) that:
- **d(W_alpha)/d(alpha) >= 0**: Alpha gradient is non-negative, so SGD decreases alpha over training (down-weighting easy samples more over time).
- **d(W_beta)/d(beta) <= 0**: Beta gradient is non-positive, so SGD increases beta over training (down-weighting hard/noisy samples more over time).

These match the paper's Theorems for the expected adaptive behavior of LiLAW under noisy labels.

### D13: PoC Sweep Results and Calibration Against the Paper

**Observations (full sweep: 3 datasets × 4 noise rates × 3 seeds, 50 epochs, asymmetric positive→negative noise):**

| Dataset | Noise | Baseline PR-AUC | LiLAW PR-AUC | Baseline Rec@PPV80 | LiLAW Rec@PPV80 |
|---|---|---|---|---|---|
| adult | 10% | 0.6924±0.0192 | 0.7016±0.0173 | 0.2759±0.0903 | 0.2964±0.1010 |
| adult | 20% | 0.6908±0.0163 | 0.6994±0.0179 | 0.2953±0.0738 | 0.2943±0.0934 |
| adult | 30% | 0.6856±0.0144 | 0.6998±0.0192 | 0.2947±0.0516 | 0.2971±0.0880 |
| adult | 40% | 0.6790±0.0148 | 0.6993±0.0181 | 0.2901±0.0603 | 0.3109±0.0810 |
| breast_cancer | 10% | 0.8114±0.2101 | 0.8349±0.1931 | 0.6480±0.4517 | 0.6604±0.4604 |
| breast_cancer | 20% | 0.8044±0.2187 | 0.8260±0.2023 | 0.6449±0.4561 | 0.6604±0.4604 |
| breast_cancer | 30% | 0.7979±0.2187 | 0.8164±0.2054 | 0.6417±0.4538 | 0.6604±0.4604 |
| breast_cancer | 40% | 0.7937±0.2202 | 0.8073±0.2133 | 0.6417±0.4538 | 0.6573±0.4648 |
| pima | 10% | 0.5613±0.0140 | 0.5701±0.0133 | 0.1152±0.0869 | 0.0947±0.0686 |
| pima | 20% | 0.5377±0.0130 | 0.5438±0.0155 | 0.0823±0.0582 | 0.0700±0.0517 |
| pima | 30% | 0.5153±0.0141 | 0.5103±0.0130 | 0.0988±0.0698 | 0.0329±0.0466 |
| pima | 40% | 0.4980±0.0096 | 0.4904±0.0139 | 0.0412±0.0582 | 0.0329±0.0466 |

**Pattern:** LiLAW consistently improves PR-AUC by ~1–2% on `adult` across all noise rates, and shows a directional improvement on `breast_cancer` (large variance due to seed 456 collapsing). On `pima`, gains are marginal at low noise and turn negative at 30–40%.

**These gains are much smaller than the paper's headline numbers** (e.g. +17% Top-1 accuracy at 50% symmetric noise on CIFAR-100-M, +40% at 50% asymmetric noise). The following hypotheses explain the gap:

1. **Noise rate and baseline degradation**: The paper's large gains occur when the baseline collapses under heavy noise (40–90% symmetric, where CIFAR-100-M baseline drops to ~58% accuracy). Our 10–40% asymmetric noise barely degrades strong baselines — `adult` baseline holds at ~0.69 PR-AUC across all noise rates, leaving little room for recovery.

2. **Asymmetric vs. symmetric noise**: Our noise is unidirectional (positive→negative flips only). The paper's theoretical guarantees ("provably improves under diagonally dominant label noise") are formulated for symmetric noise. With asymmetric noise, hard-noisy samples are indistinguishable from hard-clean samples in loss space, making the weight signal weaker.

3. **Saturated baseline on easy datasets**: `breast_cancer` PR-AUC is ~0.95–0.97 at low noise in favorable seeds — mathematically bounded near ceiling. The paper demonstrates LiLAW on datasets where the task is genuinely hard even without noise.

4. **Tabular vs. image feature richness**: The easy/moderate/hard difficulty spectrum that LiLAW exploits is richer for complex visual features (CIFAR, MedMNIST) than for 30–108 tabular features. On tabular data, per-sample loss distributions are more uniform, so the weighting provides less signal.

5. **Pima negative results at high noise**: 768 samples is tiny. Meta-update validation batches become very noisy, corrupting the meta-gradient signal. The paper's Theorem requires sufficiently clean loss geometry for LiLAW to provably improve; at 30–40% asymmetric noise on a small dataset this condition is likely violated.

**Implication for MLMD**: The MLMD setting (large structured EHR features, moderate asymmetric noise from unobserved external tests) most closely resembles `adult`. A consistent ~1–2% PR-AUC lift is plausible and clinically meaningful at scale, but should not be expected to match the paper's image-data gains. The `pima` failure mode at high noise is a risk flag if MLMD noise rates exceed ~25%.

### D12: Engineering Guardrails

- **Package management:** `uv`
- **Formatting/linting:** `ruff`
- **Type checking:** `ty` (Astral)
- **Testing:** `pytest` — core tests for gradient flow through LiLAW loss, weight function bounds, noise injection correctness

---

## MedMNIST Replication Experiment

### D14: Symmetric Noise for MedMNIST Replication

The paper's MedMNIST experiments use symmetric noise (uniform random label replacement). We match this exactly to enable direct comparison with published results. This differs from the tabular PoC which used asymmetric positive→negative noise to simulate the EHR setting.

### D15: Clean Validation Set for Early Stopping

Noise is injected into the training set only. The validation set remains clean to enable meaningful early stopping. The paper emphasizes "no clean validation set required" for LiLAW's meta-update, but standard practice for their reported results uses clean validation for model selection.

### D16: Early Stopping on Validation Accuracy

Early stopping monitors validation Top-1 accuracy with patience=10 epochs. Best model state is restored after stopping. This is applied identically to both baseline and LiLAW methods for fair comparison.

### D17: Initial Sweep Scope

BloodMNIST (11,959 train samples, 8 classes) at noise rates 0%, 20%, 40%, 60%, 80% with 3 seeds (42, 123, 456). BloodMNIST chosen for fast iteration (~30-45 min on H100). PathMNIST can be added as a follow-up.

### D18: Code Isolation

All MedMNIST replication code lives in `src/lilaw_poc/medmnist/` subpackage. The existing binary PoC code is untouched. The `lilaw.py` core (weight functions + LiLAWWeighter) is reused without modification — it already operates on generic `s_label`/`s_max` tensors.

### D19: Pretrained Weights

ResNet-18 with ImageNet pretrained weights via `timm`. The paper specifies ImageNet-21K pretraining; `timm`'s default `resnet18` uses ImageNet-1K. If results diverge from the paper, switching to a 21K checkpoint is a candidate fix.

### D20: MedMNIST Sweep Results — BloodMNIST (5 noise rates × 3 seeds)

**Run:** 2026-03-28 on RunPod A100 SXM, 15 experiments, results in `lilaw-poc/results/medmnist/results.json`.

| Noise | Baseline Acc (mean) | LiLAW Acc (mean) | Delta |
|-------|---------------------|------------------|-------|
| 0%    | 96.39%              | 96.74%           | +0.35% |
| 20%   | 94.47%              | 94.13%           | −0.35% |
| 40%   | 92.94%              | 92.78%           | −0.16% |
| 60%   | 90.15%              | 89.62%           | −0.53% |
| 80%   | 82.09%              | 82.59%           | +0.50% |

**0% noise baseline**: 96.4% accuracy — well above the paper's ≥95% expectation for BloodMNIST.

**Key finding: LiLAW does not consistently outperform the baseline.** Differences are within ±1% at all noise rates, with LiLAW trailing at 20–60% noise and showing marginal gains only at 0% and 80%. The paper's claim that LiLAW significantly maintains accuracy under high label noise is **not replicated** on this run.

**Meta-parameter behavior:** `alpha` saturates near 10 (max) across all runs regardless of noise rate. `beta` and `delta` do increase monotonically with noise rate (~2.1→2.7 and ~6.1→7.2 respectively), confirming the weighting mechanism responds to noise. However, this adaptation does not translate to accuracy improvements.

**Hypotheses for the gap vs. paper:**

1. **Epoch budget**: 30 epochs with patience=10 may be insufficient for the meta-learning loop to converge. The paper's reported results likely use longer training. Alpha saturating at ~10 suggests the easy-sample suppression mechanism ran its course early; more epochs may allow beta/delta to provide signal.
2. **ImageNet-1K vs. 21K pretraining** (D19): The paper uses 21K; a stronger pretrained backbone could sharpen the easy/moderate/hard loss landscape.
3. **Symmetric noise scale**: At 80% symmetric noise, label information is nearly destroyed — this is where the paper's largest gains occur. Our results do show LiLAW slightly ahead at 80%, consistent with the paper's direction even if not magnitude.
4. **Meta-update batch size**: If the meta-batch is too small relative to class count (8 classes), the meta-gradient is noisy, weakening the update signal.

**Next steps:** Before concluding LiLAW does not work on image data, try: (a) longer training (60–100 epochs) with larger patience, (b) check meta-batch size, (c) PathMNIST as a harder task where the baseline degrades more sharply.
