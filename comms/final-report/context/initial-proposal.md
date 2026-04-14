# 

# 

# 

# **Project Proposal**

| Date: | 6 February 2026 |
| :---- | :---- |
| **Course:** | LMP2392H: Healthcare and Medicine Basics for AI experts |
| **Instructors:** | Anna Goldenberg, Susan Poutanen |
| **Project title:** | Reading Between the Lines: LLM-Based Label Imputation and Meta-Learning for Noisy Labels in Emergency Department Triage |
| **Project advisor:** | Ismail Akrout |
| **Project team:** | Alice Chua, MScAC Tyler Biswurm, MScAC |

## Problem statement and clinical motivation

Redundant wait times constrain patient throughput in the pediatric Emergency Department (ED) at The Hospital for Sick Children (SickKids). Under standard care, patients queue twice: first for an initial physician assessment to order diagnostics, then again for results. Reducing these delays is clinically significant. Faster diagnostic turnaround enables earlier treatment decisions, shortens time to disposition, and reduces ED crowding—factors that directly affect patient outcomes and experience.

The hospital has implemented a machine learning medical directive (MLMD) system that uses triage data to predict and authorize relevant tests proactively, parallelizing diagnostic processing with the initial wait. However, the model's performance is limited by significant label noise. Positive labels in the SickKids Electronic Health Record (EHR) are reliable, but negative labels are ambiguous. A negative label may indicate that a test was medically unnecessary (a true negative), or it may reflect that the test was performed at an external institution (a false negative). The SickKids EHR is not integrated with external providers, so externally-performed tests do not appear in the training data. The algorithm mistakes these absences for clinical decisions, learning spurious patterns—for instance, that chest pain doesn't warrant an ECG.

This label ambiguity has pushed the MLMD toward a conservative decision boundary that artificially depresses recall. While the current system maintains high precision and patient safety, each relevant test the model fails to authorize returns a patient to the slower, serialized workflow, forfeiting substantial efficiency gains.

## Data and setting

The dataset consists of real clinical data from ED visits at SickKids. Input features include triage-time measurements (vital signs, CTAS acuity score, chief complaint, demographics) collected at presentation. Labels are asymmetrically noisy: positive labels are ground truth, while negative labels conflate true negatives with false negatives from unobserved external tests. Post-assessment ICD-10 diagnosis codes are available for label refinement strategies but are not model inputs, as they represent information unavailable at triage time. Data access requires institutional onboarding and is restricted to on-premise compute infrastructure.

## Methods

We address label noise through two complementary strategies. First, we clean the data by using a large language model (LLM) to identify and correct false negatives. Second, we make training more robust by implementing LiLAW, a meta-learning method. Better labels reduce noise at the source and robust training manages the residual uncertainty.

**Label imputation.**  We use an LLM to generate a diagnosis-to-test map linking ICD-10 codes to clinically indicated tests \[2\]. LLMs encode medical knowledge from training on clinical literature, enabling semantic inference about diagnosis-test relationships. When a patient's record contains a diagnosis but lacks a corresponding test, we impute a positive label. We also use the LLM and various prompt based techniques to scan free-text triage notes for mentions of external tests—for example, a note reading "US+ appy" indicates an ultrasound was performed \[3\]. We may augment the LLM's reasoning with structured medical knowledge sources (e.g., clinical knowledge graphs like ClinGraph, published guidelines like ACR criteria, or datasets like MIMIC-IV) for in-context learning to reduce hallucination risk \[4\]. If LLM hallucination proves intractable, we fall back to knowledge-graph-based approaches \[5\] that rely on conservative, validated diagnosis-test associations.

Meta-learning.  We implement LiLAW (Lightweight Learnable Adaptive Weighting) \[1\] to make training robust to label noise. LiLAW learns three scalar parameters (α, β, δ) that weight training samples based on their loss magnitude during training. Through bilevel optimization, model parameters update on weighted training batches while weighting parameters update on a validation set. This allows the model to down-weight suspected false negatives (high-loss samples where the model predicts a test but the label indicates negative) without requiring explicit noise annotations. LiLAW complements our deterministic label imputation strategy by handling probabilistic cases (e.g., chest pain sometimes warrants an ECG, but not always) in the absence of explicit probability annotations.

## Evaluation strategy

We evaluate the production baseline against three retraining strategies—imputed labels, LiLAW weighting, and a combined approach—using a held-out test set. Performance is assessed using recall as a proxy for clinical efficiency (minimizing serialized waiting) and precision as a measure of safety (minimizing unnecessary interventions). We report precision-recall curves at multiple operating points to characterize tradeoffs and prioritize recall improvements at the system's current precision threshold. We also report Expected Calibration Error (ECE).

Our validation strategy proceeds in three phases. First, we audit the intermediate LLM-generated diagnosis-to-test map through structured clinician review and by cross-referencing it against standard-of-care guidelines. Second, we quantitatively benchmark the map by measuring its ability to recover masked positive labels in held-out cases. Finally, we evaluate the fully retrained system through a blinded clinical review of cases where the model authorizes previously unordered tests.

## Deployment context and limitations

The MLMD system operates at triage, authorizing diagnostic tests before initial physician assessment to parallelize test processing with queue time. The system’s deployment context imposes several limitations. SickKids sees high transfer rates from other providers. As a consequence, many external tests go unregistered in the EHR, generating label noise. Data shift risks arise from changing referral patterns. Overly aggressive predictions risk unnecessary testing (cost, radiation exposure) and overly conservative predictions forfeit efficiency gains. Neural network opacity may limit clinician trust. Large language models may hallucinate plausible but incorrect diagnosis-test associations. The proposed meta-learning approach remains unproven on real-world clinical data.

## Project scope and deliverables

We plan to deliver the following: (1) an LLM-generated “Diagnosis → Required Tests” map, (2) a relabeled training dataset using the map, (3) a retrained MLMD model, and (4) an evaluation report comparing the new model against its predecessor. As a stretch goal, we also hope to deliver: (4) a LILAW-augmented training recipe, (5) an accordingly retrained MLMD model, and (6) another evaluation report assessing the performance gains of the retrained model.

The following are out of scope for this project: Complete ground truth recovery, new model architectures, integrating the MLMD with an external system, clinical deployment, and clinical trial administration.

## 

## Relevant work

1. A. Moturu, A. Goldenberg, and B. Taati, “LiLAW: Lightweight Learnable Adaptive Weighting to Meta-Learn Sample Difficulty and Improve Noisy Training,” arXiv preprint arXiv:2509.20786, Sep. 2025, doi: 10.48550/arXiv.2509.20786.

2. Yao, X., Sannabhadti, A., Wiberg, H., Shehadeh, K. S., & Padman, R. (2025). Can LLMs Support Medical Knowledge Imputation? An Evaluation-Based Perspective. arXiv preprint arXiv:2503.22954.

3. Islam, K. M. S., Nipu, A. S., Wu, J., & Madiraju, P. (2025). LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs. arXiv preprint arXiv:2505.08704.

4. Li, D., Kadav, A., Gao, A., Li, R., & Bourgon, R. (2025). Automated Clinical Data Extraction with Knowledge Conditioned LLMs. In Proceedings of the 31st International Conference on Computational Linguistics: Industry Track (pp. 149–162). Association for Computational Linguistics.

5. Cao, S., Li, R., Wu, R., Liu, R., Duprey, A., & Zhao, J. (2026). Real-time clinical analytics at scale: a platform built on large language models-powered knowledge graphs. JAMIA Open, 9(1), ooaf167. https://doi.org/10.1093/jamiaopen/ooaf167