# LiLAW Briefing

**Paper:** Moturu, Muzammil, Goldenberg, Taati. *LiLAW: Lightweight Learnable Adaptive Weighting to Meta-Learn Sample Difficulty, Improve Noisy Training, Increase Fairness, and Effectively Use Synthetic Data.* ICML 2026.

**Note:** Anna Goldenberg is a co-author. This is your supervisor's lab's own method.

---

## Core Idea

Every training sample is either easy, moderate, or hard — and which category it falls into changes as training progresses. LiLAW learns to dynamically reweight each sample's loss based on its difficulty, so the model focuses on informative samples at the right time.

Difficulty is defined by two values: the model's confidence in the observed label (`s[y_tilde]`) and its peak confidence (`max(s)`):

- **Easy:** model agrees with label, high confidence
- **Moderate:** model is unconfident (agrees or disagrees)
- **Hard:** model disagrees with label, high confidence — the likely noisy samples

For asymmetric label noise like ours (false negatives), hard samples are precisely the ones where the model predicts "test warranted" but the EHR label says negative. LiLAW learns to down-weight these.

---

## The Mechanism

Three scalar parameters — alpha (easy), beta (hard), delta (moderate) — control per-sample weights. Each maps to a weight function:

- `W_alpha`: sigmoid, high when model confidently agrees with label
- `W_beta`: inverted sigmoid, high when model confidently disagrees
- `W_delta`: RBF, high when model is unconfident either way

Final weight: `W = W_alpha + W_beta + W_delta`. Weighted loss: `L_W = W * L`.

**Bilevel optimization:** After each training mini-batch, theta updates on weighted training loss. Then alpha/beta/delta update on a *validation* mini-batch. Crucially, the validation set does not need to be clean — a noisy val set works, because accuracy on noisy held-out data is an unbiased estimator of accuracy under the noisy distribution.

---

## Why It Fits Our Problem

Label imputation handles *deterministic* false negatives (diagnosis X always implies test Y). LiLAW handles *probabilistic* cases — e.g., chest pain *sometimes* warrants an ECG but not always. No manual noise annotation required; the model learns which labels to distrust from the data.

---

## Practical Properties

- **3 extra parameters only** — no additional models, no dataset pruning
- **1 extra forward/backward pass per batch** — minimal compute overhead
- **No clean validation set required** — critical for clinical data where clean labels are unavailable
- **Drop-in** — wraps any existing training loop

---

## Validation Scope

Tested on general and medical imaging (MedMNISTv2), time-series (ECG5000), synthetic datasets (pain detection, gait classification), and fairness (Adult dataset). SOTA on ECG5000 and synthetic benchmarks. Validated across multiple noise types, levels, architectures, and loss functions.

**Gap:** Not yet validated on real-world asymmetric label noise in clinical tabular EHR data — which is our setting. This is an explicit limitation we acknowledge.
