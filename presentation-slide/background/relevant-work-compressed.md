# Relevant Work — Background Research

## LLMs + Knowledge Graphs for Clinical Supervision

**Yao et al. (2025)** benchmark LLMs imputing missing disease-to-treatment edges in medical KGs. LLMs propose plausible links but recall is limited and many edges conflict with clinical guidelines — motivating KG-grounded evaluation and hybrid vetting rather than naïve adoption.

**Li et al. (COLING 2025)** address LLM hallucinations in clinical extraction via *knowledge-conditioned* prompting: they distill an internal rule base, align it with SNOMED, and use it to condition the LLM. This improves F1 on lung lesion fields by ~13 points over vanilla in-context learning, showing explicit knowledge grounding substantially reduces structural errors.

**Cao et al. (JAMIA Open 2026)** demonstrate LLM+KG pipelines at production scale (110k trial documents, 60k EMRs), maintaining a live medical KG for semantic querying and evidence synthesis. Viable, but requires strong engineering and careful graph curation.

**Han et al. — Denoise2Impute (2026)** reframe unknown missingness in EHR diagnosis codes as a denoising problem. A SetTransformer distinguishes true negatives from unobserved positives using co-occurrence patterns between codes (e.g., obesity + high glucose suggesting missing diabetes). The clearest precedent for systematically separating true negatives from missing positives in EHR labels, though at the diagnosis-code level rather than triage/test labels.

**Takeaway:** LLMs and KGs are increasingly used to construct or complete clinical supervision signals, but hallucinations, guideline misalignment, and subtle missingness mechanisms remain major challenges.

---

## Noise-Robust Training Under Asymmetric Label Noise

**Yang et al. (2024)** provide the first systematic evaluation of CV label-noise techniques on tabular EHR classifiers, simulating asymmetric false-positive and false-negative rates. Label smoothing, MixUp, and Neighbor Consistency Regularization — alone and in combination — significantly improve robustness, confirming CV-inspired methods can transfer to EHR tasks.

**Ren et al. (ICML 2018) — Learning to Reweight Examples** meta-learns per-sample weights by backpropagating through a small clean validation set. Foundational reference for principled sample reweighting under label noise.

**Northcutt et al. (2021) — Confident Learning** estimates per-example label error probabilities from model predictions, enabling pruning or relabeling of likely-mislabeled points. Standard reference for data cleaning under noisy labels.

**Moturu et al. — LiLAW (2025–26)** extends meta-reweighting with only three global learnable parameters (alpha, beta, delta), mapping per-sample loss to difficulty-based weights via bilevel optimization. Retains robustness benefits of complex meta-weight networks while being cheap to train and easy to drop into existing pipelines.

---

## Where This Leaves Our Work

Existing LLM+KG work targets disease-to-treatment edges, free-text extraction, or diagnosis-code missingness — not triage/test labels corrupted by unobserved external tests. Noise-robust methods assume synthetic class-conditional noise or coarse EHR noise models, not our specific asymmetric false-negative mechanism.

Our work combines **KG-guided label imputation** (inspired by Yao, Denoise2Impute) with **LiLAW-style meta-reweighting**, closing the loop between how labels are constructed and how the model trains under residual noise — a regime the current literature only touches indirectly.
