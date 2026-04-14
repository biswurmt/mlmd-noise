# Final Report TODO
**Deadline: April 14, 2026**

---

## What Alice wrote

- [x] Introduction (clinical workflow, MLMD system, label noise mechanism)
- [x] Related Work: Label Imputation in EHRs
- [x] Related Work: Knowledge Graphs and LLMs in Clinical Decision Support
- [x] Methods: KG construction (RAG, API enrichment, multi-agent audit, imputation strategies)
- [x] Cohort selection and experimental setup (N=483,705, exclusions, feature space)
- [x] Results: Label imputation performance table (real numbers, all 4 pathways)
- [x] Results: Inference latency benchmarks
- [x] Discussion: Clinical translation and limitations (KG-focused)
- [x] Future Work

---

## What Tyler wrote / owns

- [x] Related Work: Meta-Learning for Noisy Labels (`02-related-work.tex`)
- [x] Methods: Robust Training via LiLAW (`04-methods.tex`)
- [x] Results: LiLAW proxy validation subsection (tabular PoC + MedMNIST replication, real numbers)
- [~] Results: LiLAW-only ablation narrative (drafted with PENDING markers; needs MLMD numbers for Table 2)
- [~] Results: Combined condition narrative (drafted with PENDING markers; needs MLMD numbers for Table 2)

---

## What's still missing (either/both)

**Blockers — needed before the report is complete:**
- [ ] All downstream model numbers (Table 2: Precision, Recall, F1, AUROC across 4 conditions)
- [ ] ECE values for all 4 conditions (required metric)
- [ ] PR curves figure (required; multi-threshold, all 4 conditions)
- [ ] KG validation stats: node/edge counts, audit verdict breakdown, mean confidence score
- [ ] Clinician chart review: sample size + accuracy rate for discovered labels

**Polish — once numbers are in:**
- [ ] Fill abstract result placeholders (`00-abstract.tex`)
- [ ] Update discussion with final quantitative conclusions (`07-discussion-and-limitations.tex`)
- [ ] Run `make pdf` and review layout, table overflow, figure placement
