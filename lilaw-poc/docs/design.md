# Design & Summary

This document summarises key decisions; for a full log, see [DECISIONS.md](DECISIONS.md).

## High-Level Summary

**Goal**: Evaluate whether LiLAW weighting can improve robustness to asymmetric label noise.

**Asymmetric noise:** We inject noise only in the positive-to-negative direction to mimic the clinical reality where tests at external hospitals go unobserved.

**Datasets**: Breast Cancer (sklearn), Adult (UCI), Pima Indians Diabetes.

**Model:** 5-layer MLP, BCE loss, SGD optimizer (lr=1e-4), momentum 0.9.

**Metrics**: PR-AUC (primary), Recall@PPV≥0.8 (secondary). No AUROC.

---

Full motivation and parameter choices are captured in [DECISIONS.md](DECISIONS.md).
