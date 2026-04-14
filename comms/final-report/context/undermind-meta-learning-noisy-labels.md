# Undermind Agent Prompt: Literature Review on Meta-Learning for Noisy Labels

## Purpose

Perform a comprehensive literature review on meta-learning approaches for learning with noisy labels. The output will inform the writing of a related work paragraph for:

> *"Reading Between the Lines: LLM-Based Label Imputation and Meta-Learning for Noisy Labels in Emergency Department Triage"*
> CSC2541H, University of Toronto — Deadline: April 14, 2026

The paragraph will appear in `comms/final-report/sections/02-related-work.tex`, under the heading `\paragraph{Meta-learning for noisy labels.}`.

---

## Why This Review Matters

The paper applies **LiLAW** (Lightweight Learnable Adaptive Weighting, Moturu et al. 2025, arXiv:2509.20786) to a clinical dataset with asymmetric label noise. Negative labels in the SickKids pediatric ED EHR are ambiguous: some are true negatives (test not indicated), others are false negatives (test was performed at an external provider and never recorded in SickKids' EHR due to lack of interoperability). LiLAW down-weights suspected false negatives during training using three meta-learned scalar parameters (alpha, beta, delta) and bilevel optimization against a held-out validation set.

The related work paragraph must situate LiLAW within the broader landscape of noisy-label learning — explaining what came before, how the field evolved, and exactly what LiLAW contributes that prior methods do not.

---

## What to Cover

Research and synthesize the following topics, roughly in historical/conceptual order.

### 1. Noise-Robust Loss Functions (pre-meta-learning era)

- Symmetric loss functions (e.g., MAE vs. cross-entropy) and their theoretical properties under label noise
- Loss correction approaches:
  - Patrini et al. CVPR 2017, "Making neural networks robust to label noise: a loss correction approach"
  - Generalized Cross Entropy — Zhang & Sabuncu, NeurIPS 2018
  - Symmetric Cross Entropy — Wang et al., ICCV 2019
- What are the theoretical guarantees and empirical weaknesses of these approaches?

### 2. Sample Selection and Curriculum Learning

- **MentorNet** (Jiang et al. 2018, arXiv:1712.05055): data-driven curriculum for corrupted labels; uses a separate mentor network trained on clean data to guide the student network
- **Co-teaching** (Han et al., NeurIPS 2018): two networks select small-loss samples to teach each other; avoids MentorNet's clean-data requirement
- **DivideMix** (Li et al. 2020, arXiv:2002.07394): GMM-based clean/noisy sample separation + semi-supervised learning
- **UNICON** (Karim et al. 2022, arXiv:2203.14542): uniform selection + contrastive learning
- **Early-Learning Regularization** (Liu et al. 2020, arXiv:2007.00151): prevents memorization of noisy labels via early stopping signals
- How do these methods differ in their assumptions about noise structure? What are the failure modes when noise is asymmetric (class-conditional) rather than symmetric?

### 3. Meta-Learning / Sample-Reweighting Approaches

- **L2RW / LRW** (Ren et al., ICML 2018, arXiv:1803.09050): learns per-sample weights via bilevel meta-gradient on a small clean validation set; requires a clean, unbiased meta-set; cited in project `references.bib` as `ren_learning_2018`
- **MAML** (Finn et al., ICML 2017, arXiv:1703.03400): model-agnostic meta-learning via gradient-through-gradient; foundational bilevel optimization formulation
- **Meta-Weight-Net / MWNet** (Shu et al., NeurIPS 2019, "Meta-weight-net: Learning an explicit mapping for sample weighting"): learns a weighting function (small MLP) mapping loss value to weight, also via bilevel optimization; still requires a clean validation set
- **MOLERE** (Jain et al. 2024, arXiv:2403.12236): meta-learning on hard validation samples; three-network architecture
- **Confident Learning** (Northcutt et al. 2019, arXiv:1911.00068): estimates the joint distribution of noisy and true labels; prunes label errors before training; no bilevel optimization; cited in project `references.bib` as `northcutt_confident_learning_2019`

For each method, assess: (a) clean validation set requirements, (b) computational overhead, (c) scalability to large datasets, (d) asymmetric noise handling.

### 4. Label Noise in Clinical/EHR Settings

- Yang et al. 2024 (BMC Medical Informatics, DOI: 10.1186/s12911-024-02581-5, cited as `yang_addressing_2024`): adapts CV noise-robust methods to EHR tabular data; cross-domain transfer results
- What are the unique challenges of asymmetric/PU-style noise in EHR data versus symmetric label noise in image benchmarks?

### 5. LiLAW's Position in the Landscape

Drawing on the LiLAW paper (Moturu, Muzammil, Goldenberg, Taati, arXiv:2509.20786, cited as `moturu_lilaw_2025`):

- **What LiLAW does**: three scalar parameters (alpha for easy samples, delta for moderate, beta for hard), adaptive weighting via bilevel meta-gradient on a validation mini-batch after each training mini-batch; uses only softmax output disagreement and confidence as signal
- **What LiLAW does NOT need**: a clean/unbiased validation set; additional auxiliary models; per-sample weight parameters
- **Comparison to prior work**:
  - vs. L2RW: per-sample weights scale O(N) with dataset size; LiLAW uses 3 global scalars
  - vs. MentorNet/Co-teaching: no auxiliary mentor/peer networks
  - vs. MWNet: no weight-function MLP; much lower parameter count
  - vs. MOLERE: one network, not three
- **Core novelty**: extreme parameter efficiency (3 scalars total) + adaptive without requiring clean validation data + differentiable weighting geometry based on agreement/confidence in softmax space
- **Demonstrated results**: state-of-the-art on MedMNIST variants, SynPAIN, GAITGen, ECG5000

---

## Sources to Prioritize

Read at minimum the following, listed with their arXiv IDs or DOIs:

| # | Key | Paper | Where to find |
|---|-----|--------|--------------|
| 1 | `moturu_lilaw_2025` | LiLAW (Moturu et al. 2025) | arXiv:2509.20786 |
| 2 | `ren_learning_2018` | L2RW: "Learning to Reweight Examples for Robust Deep Learning" (Ren et al. 2018) | arXiv:1803.09050 |
| 3 | — | MAML: "Model-Agnostic Meta-Learning" (Finn et al. 2017) | arXiv:1703.03400 |
| 4 | — | Co-teaching (Han et al. NeurIPS 2018) | NeurIPS 2018 proceedings |
| 5 | — | MentorNet (Jiang et al. 2018) | arXiv:1712.05055 |
| 6 | — | DivideMix (Li et al. 2020) | arXiv:2002.07394 |
| 7 | `northcutt_confident_learning_2019` | Confident Learning (Northcutt et al. 2019) | arXiv:1911.00068 |
| 8 | — | MOLERE (Jain et al. 2024) | arXiv:2403.12236 |
| 9 | `yang_addressing_2024` | EHR label noise (Yang et al. 2024) | DOI: 10.1186/s12911-024-02581-5 |
| 10 | — | Generalized Cross Entropy (Zhang & Sabuncu 2018) | NeurIPS 2018 |
| 11 | — | Meta-Weight-Net (Shu et al. NeurIPS 2019) | search "Meta-weight-net: Learning an explicit mapping for sample weighting" |
| 12 | — | Loss correction (Patrini et al. CVPR 2017) | search "Making neural networks robust to label noise: a loss correction approach" |

---

## Output Format

Return a single document structured as follows:

### 1. Conceptual Landscape (~400 words)
Narrative describing the arc from noise-robust losses → curriculum/sample selection → meta-learning reweighting. Identify the key motivating weakness at each transition.

### 2. Method Comparison Table
For each of the ~10 key methods, provide:
- Method name, year, venue
- Core mechanism (1–2 sentences)
- Requires clean validation set? (Y/N)
- Requires auxiliary model? (Y/N)
- Handles asymmetric noise? (Y / N / Partial)
- Computational overhead vs. standard training
- Key weakness/limitation

### 3. LiLAW Positioning Summary (~200 words)
Where exactly does LiLAW sit? What gap does it fill? Be specific about the three design choices (parameter count, no clean-set requirement, softmax-geometry-based weighting) and how each addresses a concrete limitation of prior work.

### 4. Draft Related Work Paragraph (~250–300 words)
A polished academic paragraph for direct inclusion in `02-related-work.tex`. Style rules (non-negotiable):

- **No em dashes** — use commas, parentheses, or separate sentences instead
- **Quantify claims** where possible (e.g., "L2RW assigns per-sample weights, introducing O(N) additional parameters per mini-batch")
- **Define acronyms** on first use
- No unquantified hedges ("significant improvement" requires a number)
- Final sentence must connect LiLAW's design to the SickKids asymmetric noise problem (false negatives from external provider transfers, not symmetric benchmark noise)
- Use the following citation keys from the project bib where applicable:
  - `\cite{ren_learning_2018}` — L2RW
  - `\cite{moturu_lilaw_2025}` — LiLAW
  - `\cite{northcutt_confident_learning_2019}` — Confident Learning
  - `\cite{yang_addressing_2024}` — EHR label noise
  - For papers not yet in the project bib (MWNet, MentorNet, Co-teaching, DivideMix), provide the full citation details so they can be added

### 5. Gaps and Caveats
- Papers you were unable to fully access
- Any claims in LiLAW's related work positioning that conflict with what you found in primary sources
- Suggestions for additional citations worth adding to `references.bib`
