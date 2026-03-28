# LiLAW PoC Implementation Plan — Overview

> **For agentic workers:** Implement the plan task-by-task in order. Each task is self-contained with tests, implementation, and a commit. Check off steps (`- [ ]` → `- [x]`) as you go. Use subagents/parallel execution for independent tasks where your environment supports it.

**Goal:** Build a minimum viable PyTorch implementation of LiLAW (Lightweight Learnable Adaptive Weighting) that demonstrates recall recovery under asymmetric label noise on three tabular datasets, simulating the MLMD clinical system.

**Architecture:** Modular Python package (`lilaw_poc`) with separate modules for noise injection, LiLAW weighting, MLP model, dataset loading, evaluation, and experiment orchestration. The training loop alternates between a standard BCE update on a training batch and a meta-update of (α, β, δ) on a validation batch. An experiment runner sweeps over datasets × noise rates × seeds and produces PR curves + summary tables.

**Tech Stack:** Python 3.12+, PyTorch, scikit-learn, pandas, matplotlib, uv, ruff, ty, pytest

**Reference documents:**
- Design decisions: `lilaw-poc/DECISIONS.md`
- LiLAW paper source: `presentation-slide/background/lilaw/example_paper.tex`

---

## Paper Verification Checkpoints

**This implementation must be verified against the LiLAW paper at two mandatory checkpoints.** The paper source is at `presentation-slide/background/lilaw/example_paper.tex`.

### Checkpoint 1: Before Writing Code (after reading the plan, before Task 2)

Read the LiLAW paper and verify the following match between the plan and the paper:

- [ ] **Weight functions:** W_α (Eq. 2), W_β (Eq. 3), W_δ (Eq. 4) — confirm the sigmoid/RBF formulas in `lilaw.py` match the paper exactly.
- [ ] **Combined weight:** W = W_α + W_β + W_δ — confirm additive combination, applied multiplicatively to the loss.
- [ ] **Algorithm 1:** The alternating loop — confirm Step 1 (training update with frozen meta-params), Step 2 (meta-validation with frozen model), Step 3 (manual SGD on α, β, δ) match the plan's `train_lilaw`.
- [ ] **Binary adaptation:** Confirm how s_i[ỹ_i] and max(s_i) are adapted for binary sigmoid output (not multi-class softmax).
- [ ] **Meta-parameter initialization:** α=10, β=2, δ=6, lr=0.005, wd=0.0001.
- [ ] **Warmup:** 1 epoch of vanilla training before LiLAW activates.
- [ ] **Gradient properties:** ∇_α L_W ≥ 0 (α decreases), ∇_β L_W ≤ 0 (β increases), ∇_δ L_W can go either way.

If any discrepancy is found, fix the plan before proceeding.

### Checkpoint 2: After All Code Is Written (after Task 10, before claiming done)

Re-read the LiLAW paper and audit the implemented code against it:

- [ ] Re-verify all items from Checkpoint 1 against the **actual code** (not the plan).
- [ ] Confirm gradient flow: run a test that checks ∇_α L_W ≥ 0 and ∇_β L_W ≤ 0 empirically on a small batch.
- [ ] Confirm meta-parameter evolution direction: on a short training run, verify α decreases and β increases (as stated in the paper).
- [ ] Compare weight function outputs for a known easy/moderate/hard sample against hand-calculated expected values from the paper's formulas.

Document any discrepancies found and fixes applied in `lilaw-poc/DECISIONS.md` under a new heading "## Verification Notes".

---

## File Structure

```
lilaw-poc/
├── DECISIONS.md                    # Design rationale (exists)
├── PLAN-OVERVIEW.md                # This file
├── PLAN-TASK-01.md through PLAN-TASK-10.md
├── pyproject.toml                  # uv project, ruff/pytest/ty config
├── setup.sh                        # Environment bootstrap
├── src/
│   └── lilaw_poc/
│       ├── __init__.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── datasets.py         # Load + preprocess breast cancer, adult, pima
│       │   └── noise.py            # Asymmetric noise injection
│       ├── model.py                # 5-layer MLP (MLMD mock)
│       ├── lilaw.py                # LiLAW weight functions + meta-params
│       ├── train.py                # Training loops (baseline + LiLAW)
│       ├── evaluate.py             # PR-AUC, Recall@PPV>=0.80, PR curves
│       └── experiment.py           # Sweep over datasets × noise × seeds
├── tests/
│   ├── test_noise.py
│   ├── test_lilaw.py
│   ├── test_model.py
│   ├── test_datasets.py
│   ├── test_evaluate.py
│   ├── test_train.py
│   └── test_e2e.py
└── results/                        # Generated experiment outputs (gitignored)
```

---

## Task Summary

| Task | Description | Tests | File |
|------|-------------|-------|------|
| 1 | Project scaffolding (uv, ruff, ty, pytest) | Toolchain verification | PLAN-TASK-01.md |
| 2 | Asymmetric noise injection | 5 tests | PLAN-TASK-02.md |
| 3 | LiLAW weight functions + meta-params | 8 tests | PLAN-TASK-03.md |
| 4 | 5-layer MLP (MLMD mock) | 4 tests | PLAN-TASK-04.md |
| 5 | Dataset loaders (breast cancer, adult, pima) | 15 tests | PLAN-TASK-05.md |
| 6 | Evaluation metrics (PR-AUC, Recall@PPV) | 4 tests | PLAN-TASK-06.md |
| 7 | Training loop — baseline BCE | 2 tests | PLAN-TASK-07.md |
| 8 | Training loop — LiLAW weighted | 3 tests | PLAN-TASK-08.md |
| 9 | Experiment runner + aggregation | — | PLAN-TASK-09.md |
| 10 | End-to-end smoke test | 1 test | PLAN-TASK-10.md |

**Total: ~42 tests across 10 tasks.**
