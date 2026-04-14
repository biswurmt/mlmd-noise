# Background: MLMD Noisy Labels Project

## Problem

The pediatric ED at SickKids uses a machine learning medical directive (MLMD) system to predict and authorize diagnostic tests at triage, parallelizing test processing with patient wait time. However, label noise limits performance. Positive labels (tests ordered) are reliable, but negative labels are ambiguous: they may be true negatives (test unnecessary) or false negatives (test performed externally, not recorded in SickKids EHR due to lack of integration with external providers).

The model learns spurious absences—e.g., "chest pain doesn't warrant ECG"—artificially depressing recall and forfeiting efficiency gains. While high precision maintains patient safety, each missed test returns a patient to the slower, serialized workflow.

## Data

SickKids EHR dataset: ~520,000 ED visits with triage-time features (vitals, CTAS score, chief complaint, demographics). Five binary test-ordered labels. Post-assessment ICD-10 diagnosis codes (not model inputs, unavailable at triage) enable label refinement. Asymmetric noise: positives are ground truth; negatives conflate true negatives with unobserved external tests. Data access limited to on-premise infrastructure.

## Methods

**Label Imputation.** Use LLM to generate diagnosis-to-test maps linking ICD-10 codes to clinically indicated tests. When a patient has a diagnosis but lacks a corresponding test label, impute positive. Augment with structured medical knowledge (clinical guidelines, knowledge graphs) for in-context learning to reduce hallucination risk. Iteratively audit rules using evidence grounding (literature co-occurrence counts) and LLM verification before building the final graph.

**Robust Training.** Implement LiLAW (Lightweight Learnable Adaptive Weighting), a meta-learning method that learns three scalar parameters (alpha, beta, delta) to weight training samples by loss magnitude. Bilevel optimization: model params update on weighted training batches; weighting params update on validation set. This down-weights suspected false negatives without explicit noise annotations, handling probabilistic cases label imputation cannot adjudicate (e.g., chest pain sometimes warrants ECG).

## Evaluation

Compare four conditions: (1) production baseline, (2) imputed labels only, (3) LiLAW training only, (4) combined. Metrics: recall (clinical efficiency), precision (safety), PR curves at multiple operating points, Expected Calibration Error (ECE). Three-phase validation: audit diagnosis-to-test map against guidelines and clinician review; quantitatively benchmark map's ability to recover masked labels; evaluate fully retrained system through blinded clinical review.

## Limitations & Constraints

High transfer rates from external providers generate label noise. Data shift from changing referral patterns. LLM hallucination risk (audit pipeline mitigates). LiLAW unproven on real clinical asymmetric noise. Neural network opacity may limit clinician trust. Deployment requires on-premise access, limiting experiment throughput. Out of scope: complete ground truth recovery, new architectures, external system integration, clinical deployment.

## Deliverables

(1) Diagnosis-to-test knowledge graph; (2) label-imputed training dataset; (3) retrained MLMD model; (4) evaluation report. Stretch goals: (5) LiLAW training recipe; (6) LiLAW-retrained model; (7) comparative evaluation.
