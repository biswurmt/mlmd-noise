# Report Collaboration Guidelines

## Project

**Reading Between the Lines: LLM-Based Label Imputation and Meta-Learning for Noisy Labels in Emergency Department Triage**
CSC2541H, University of Toronto. Deadline: April 14, 2026.

**Research question:** Can label imputation via LLM-guided diagnosis-to-test mapping, combined with LiLAW meta-learning, improve test-prediction recall without sacrificing precision in a clinical MLMD system where negative labels are systematically unreliable?

---

## Writing Standards

Follow the full writing process in `context/scientific-communication-guide.md`. It is the authoritative style reference. Use the multi-pass drafting workflow (Draft → Edit_1_Structure → Edit_2_Ideas → Edit_3_Style → Edit_4_AntiTropes → Final_Output) for all section drafting.

**Project-specific additions:**
- Medical accuracy is non-negotiable. ICD-10 codes are real clinical artifacts. External provider transfers are the specific mechanism of label noise. Do not simplify or generalize these.
- Every result must connect to test-ordering efficiency (recall) or patient safety (precision). Avoid framing findings as abstract ML performance.
- The system is decision-support, not automation. Clinicians retain final authority.

---

## Project Context

### Clinical Setting

SickKids Paediatric ED uses a Machine Learning Medical Directive (MLMD) system that predicts whether specific diagnostic tests are needed at triage. When the model predicts a test is likely needed, that test is ordered in parallel with the patient's wait time (parallelized workflow). When the model predicts it is not needed, the test can only be ordered after physician assessment (serialized workflow). High recall directly translates to time saved; missed tests push patients back into the serialized queue.

### The Label Noise Problem

Positive labels (test was ordered at SickKids) are reliable ground truth. Negative labels are ambiguous: a recorded absence may be a true negative (test not clinically indicated) or a false negative (test performed at a different facility, not captured in SickKids' EHR due to lack of interoperability with external providers). This asymmetric noise artificially depresses recall — the model learns that certain clinical presentations do not require a test, when in fact the test was ordered elsewhere.

Five binary test-ordered labels. ~520,000 ED visits. Triage-time features: vitals, CTAS score, chief complaint, demographics. ICD-10 diagnosis codes assigned post-assessment are available for label refinement (not as model inputs).

### Methods

**Label Imputation:** An LLM generates a diagnosis-to-test knowledge graph mapping ICD-10 diagnosis codes to clinically indicated tests. When a patient has a post-assessment diagnosis that maps to a test but lacks a positive label for that test, the label is imputed positive. The graph is augmented with structured medical knowledge (clinical guidelines, literature co-occurrence counts) and audited iteratively before use. This handles the deterministic subset of false negatives, where a diagnosis clearly implies a test was needed.

**LiLAW (Lightweight Learnable Adaptive Weighting):** A meta-learning approach that learns three scalar parameters (alpha, beta, delta) to weight training samples by their loss magnitude. Uses bilevel optimization: model parameters update on weighted training batches; weighting parameters update on validation performance. This handles the probabilistic subset of false negatives that imputation cannot adjudicate — cases where clinical indication is ambiguous (e.g., chest pain sometimes warrants ECG, sometimes does not).

### Experimental Conditions

Four conditions, evaluated with the same metrics:
1. **Production baseline** — current deployed model on original noisy labels
2. **Imputation only** — retrained on label-imputed dataset, standard training
3. **LiLAW only** — original noisy labels, LiLAW-weighted training
4. **Combined** — label-imputed dataset, LiLAW-weighted training

### Metrics

Always report all of:
- **Recall** (primary): Fraction of needed tests the model predicts. Direct clinical efficiency measure.
- **Precision**: Fraction of predicted tests that are needed. Safety measure.
- **PR curves**: Report at multiple operating thresholds. Never single-threshold conclusions.
- **ECE (Expected Calibration Error)**: Confidence calibration. Relevant to clinician trust.
- **Label recovery rate**: Fraction of externally-treated cases successfully identified by imputation. Reported in Methods/Evaluation.

---

## Constraints

- Edit files in `sections/` only, unless explicitly told otherwise.
- Do not rewrite sections wholesale without direction. Refine and extend what exists.
- Do not introduce metrics, claims, or comparisons not grounded in the actual analysis.
- Do not change notation or terminology without discussion. Consistency across sections matters.

---

## Section Guidance

| Section | Word budget | Must accomplish |
|---|---|---|
| `00-abstract` | ~250 words | State problem, approach, key quantitative result, and clinical implication. Self-contained. |
| `01-introduction-clinical-motivation` | ~1–1.5 pages | Ground the reader in the SickKids clinical workflow. Make label noise concrete before naming it as a technical problem. Related work should identify gaps, not just list prior work. |
| `02-data-and-setting-methods` | ~1.5–2 pages | Describe all four experimental conditions. Explain the asymmetric noise structure. Detail LiLAW's bilevel update. Reader should be able to reproduce the setup. |
| `03-evaluation-and-results` | ~2–2.5 pages | Lead with the combined condition against baseline. Include PR curves. Error analysis required. Statistical uncertainty (confidence intervals or bootstrap) required. Explain clinical meaning of numbers. |
| `04-discussion-and-limitations` | ~0.75 pages | Summarize what worked and what didn't, with honest attribution of limitations. Distinguish decision-support role clearly. |
| `05-future-work` | ~0.5 pages | One focused paragraph. Potential impact at scale, concrete next steps. |

---

## Reference Materials (read-only)

- `context/project-background-short.md` — authoritative project description
- `context/scientific-communication-guide.md` — authoritative style guide
- `context/final-report-guidance.pdf` — course submission requirements
- `context/milestone-one/` — prior milestone for structural reference
