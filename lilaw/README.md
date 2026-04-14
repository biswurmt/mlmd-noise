# LiLAW Patch for MLMD Training

Drop-in patch adding **LiLAW** (Lightweight Learnable Adaptive Weighting) to the
MLMD multimodal fusion training recipe.

Reference: *Lightweight Learnable Adaptive Weighting to Claim the Jungle of
Noisy Labels* — arXiv:2502.01981.

---

## Files

| File | Purpose |
|---|---|
| `lilaw.py` | `LiLAWWeighter` — meta-parameter container and weight functions |
| `lilaw_module.py` | `LiLAWFusionLightningModule` — drop-in for `FusionLightningModule` |
| `MLMD_TRAINING_GUIDE.md` | Reference guide for the existing training system |
| `Sample_training_for_students.ipynb` | Reference notebook (source of truth for the baseline) |

---

## Integration

### 1. Place the files

Copy `lilaw.py` and `lilaw_module.py` into the package alongside
`FusionLightningModule`. Adjust the import at the top of `lilaw_module.py` if
needed:

```python
# Default (same package, relative import):
from lilaw import LiLAWWeighter

# If placed in e.g. src/mlmd/train/:
from src.mlmd.train.lilaw import LiLAWWeighter
```

### 2. Swap the model class

In your training script, replace:

```python
from <wherever> import FusionLightningModule
# ...
models[mlmd_name]['model'] = FusionLightningModule(
    tabular_dim=tabular_dim,
    text_dim=text_dim,
    pos_weight=pos_weight[mlmd_name],
    lr=1e-3,
    use_pos_weight=True,
)
```

with:

```python
from lilaw_module import LiLAWFusionLightningModule
# ...
models[mlmd_name]['model'] = LiLAWFusionLightningModule(
    tabular_dim=tabular_dim,
    text_dim=text_dim,
    pos_weight=pos_weight[mlmd_name],
    lr=1e-3,
    use_pos_weight=True,
    warmup_epochs=1,
    meta_lr=0.005,
    meta_wd=0.0001,
    alpha_init=10.0,
    beta_init=2.0,
    delta_init=6.0,
)
```

Everything else — `Trainer` config, callbacks, dataloaders — is unchanged.

**`save_model()` requires one change:** replace
`FusionLightningModule.load_from_checkpoint(...)` with
`LiLAWFusionLightningModule.load_from_checkpoint(...)`.

---

## How It Works

Each training step runs two updates:

1. **Inner (model) update:** Per-sample BCE loss is computed on the training
   batch. Each sample's loss is multiplied by a LiLAW weight that reflects its
   difficulty — suspected false negatives (high loss, low confidence) receive
   lower weight, reducing their corrupting influence on the gradient.

2. **Outer (meta) update:** The three LiLAW scalar parameters (α, β, δ) are
   updated via SGD on a batch drawn from the validation dataloader. This is the
   bilevel step: meta-parameters adapt to make the weighting scheme work well
   on held-out data.

During `warmup_epochs`, the model trains with standard weighted BCE (identical
to the baseline) so that probabilities are meaningful before LiLAW activates.

### Weight functions

| Parameter | Weight function | Targets |
|---|---|---|
| α (alpha, init 10) | `sigmoid(α·s_label − s_max)` | Up-weights easy (correctly predicted) samples |
| β (beta, init 2) | `sigmoid(−(β·s_label − s_max))` | Down-weights hard (confidently wrong) samples |
| δ (delta, init 6) | `exp(−(δ·s_label − s_max)² / 2)` | RBF peak near the decision boundary (moderate samples) |

`s_label` = predicted probability for the observed label; `s_max` = max predicted
probability across the two classes. For binary classification:
`s_label = p·y + (1−p)·(1−y)`, `s_max = max(p, 1−p)`.

---

## Hyperparameters

| Parameter | Default | Notes |
|---|---|---|
| `warmup_epochs` | 1 | Epochs of standard BCE before LiLAW activates. Try 1–3. |
| `meta_lr` | 0.005 | SGD learning rate for α, β, δ. |
| `meta_wd` | 0.0001 | Weight decay on meta-parameters. |
| `alpha_init` | 10.0 | Paper defaults. |
| `beta_init` | 2.0 | |
| `delta_init` | 6.0 | |

### MLflow logging

In addition to the baseline metrics, the module logs `lilaw_alpha`,
`lilaw_beta`, and `lilaw_delta` each epoch. Monitoring these confirms the
meta-parameters are moving and gives a signal about what the weighter is
learning.

---

## Departures from the Paper

A few implementation choices differ from a literal reading of the paper's
Algorithm 1 (arXiv:2502.01981). These are intentional and standard practice,
but worth noting for review:

1. **Weight detachment in the inner step.** The paper's pseudocode lets
   gradients flow through W(s) to θ. This implementation detaches the weights
   (`w.detach() * bce`), so the model gradient comes only from the per-sample
   loss. This prevents the model from "gaming" the weighting and is standard
   in meta-learning for sample reweighting (the PoC does the same).

2. **No `pos_weight` in the meta-update.** The inner step uses the baseline's
   `pos_weight`-adjusted BCE; the outer (meta) step uses plain BCE. The
   meta-parameters should learn to steer the model toward good held-out
   performance under the natural loss, not the frequency-adjusted one.

3. **Warmup epochs.** The paper starts LiLAW from epoch 1, but their
   experiments use pretrained models. Since MLMD trains from scratch, a
   warmup period lets predictions become meaningful before LiLAW activates.

---

## Evaluation

Run evaluation identically to the baseline — `TestEvaluator` and
`plot_metrics()` are unchanged. The key metrics to compare against baseline:

- **Recall** (primary): the method aims to recover true positives that were
  suppressed by false-negative label noise.
- **Precision** (safety constraint): must not degrade.
- **PR AUC** at multiple operating thresholds.
